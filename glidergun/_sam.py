import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import IO, TYPE_CHECKING, cast

import numpy as np
from rasterio.crs import CRS
from scipy.cluster.hierarchy import DisjointSet

from glidergun._grid import Grid, con, grid, standardize
from glidergun._types import FeatureCollection
from glidergun._utils import create_directory_for, get_geojson

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from glidergun._stack import Stack


@dataclass(frozen=True, repr=False, slots=True)
class SamResult:
    masks: list["SamMask"]
    source: "Stack"

    def mask(self, *labels: str) -> Grid:
        polygons = [(mask.to_polygon(), 1) for mask in self.masks if not labels or mask.label in labels]
        return grid(polygons, self.source.extent, self.source.crs, self.source.cell_size) == 1

    def highlight(self, *labels: str) -> "Stack":
        return self.source.each(lambda g: con(self.mask(*labels), g, g / 5)).type("uint8", 0)

    def to_geojson(self, crs: int | str | CRS | None = 4326, smooth_factor: float = 0.0) -> FeatureCollection:
        return get_geojson(
            (mask.to_polygon(crs, smooth_factor), {"label": mask.label, "score": mask.score}) for mask in self.masks
        )

    def save_geojson(self, file: str | IO[str], crs: int | str | CRS | None = 4326, smooth_factor: float = 0.0):
        data = self.to_geojson(crs, smooth_factor)
        if isinstance(file, str):
            create_directory_for(file)
            with open(file, "w") as f:
                json.dump(data, f)
        else:
            json.dump(data, file)


@dataclass(frozen=True, slots=True)
class SamMask:
    label: str
    score: float
    mask: Grid

    def to_polygon(self, crs: int | str | CRS | None = None, smooth_factor: float = 0.0):
        g = self.mask.set_nan(0)
        if crs:
            g = g.project(crs)
        polygons = [polygon for polygon, value in g.to_polygons(smooth_factor=smooth_factor) if value == 1]
        return max(polygons, key=lambda p: p.area)


@dataclass(frozen=True)
class Sam:
    def sam3(self, *prompt: str, model=None, confidence_threshold: float = 0.5, tile_size: int = 500):
        """Run Segment Anything Model 3 (SAM 3) over the stack with text prompts.

        Args:
            prompt: One or more text prompts used for segmentation.
            model: Optional pre-built SAM 3 model; built and cached if None.
            confidence_threshold: Minimum confidence to accept a predicted mask.

        Returns:
            SamResult: Collection of masks and an overview visualization stack.
        """
        self = cast("Stack", self)

        buffer = 0.1
        ios_threshold = 0.5

        w, h = self.cell_size * tile_size
        tiles = [e.adjust(-w * buffer, -h * buffer, w * buffer, h * buffer) for e in self.extent.tiles(w, h)]

        all_masks: list[SamMask] = []
        for i, tile in enumerate(tiles):
            logger.info(f"Processing tile {i + 1} of {len(tiles)}...")
            all_masks.extend(
                _execute_sam3(self.clip(tile), *prompt, model=model, confidence_threshold=confidence_threshold)
            )

        n = len(all_masks)
        ds = DisjointSet(range(n))
        for i, m in enumerate(all_masks):
            for j in range(i + 1, n):
                if _same_object(m, all_masks[j], ios_threshold):
                    ds.merge(i, j)

        groups = {}
        for i in range(n):
            groups.setdefault(ds[i], []).append(all_masks[i])

        masks = []
        for grouped in groups.values():
            first = grouped[0]
            if len(grouped) == 1:
                masks.append(first)
            else:
                label = first.label
                score = max(m.score for m in grouped)
                mask = sum(standardize(*[m.mask for m in grouped], extent="union")) > 0
                masks.append(SamMask(label=label, score=score, mask=mask))  # type: ignore

        return SamResult(masks=masks, source=self)


@lru_cache(maxsize=1)
def _build_sam3():
    from huggingface_hub import HfFolder, login
    from sam3.model_builder import build_sam3_image_model

    if not HfFolder.get_token():
        login()

    bpe_path = os.path.join(os.path.dirname(__file__), "assets", "bpe_simple_vocab_16e6.txt.gz")
    return build_sam3_image_model(bpe_path=bpe_path)


def _execute_sam3(stack: "Stack", *prompt: str, model=None, confidence_threshold: float = 0.5):
    from sam3.model.sam3_image_processor import Sam3Processor
    from torch import from_numpy

    if model is None:
        model = _build_sam3()

    processor = Sam3Processor(model, device=model.device, confidence_threshold=confidence_threshold)  # type: ignore
    rgb = np.stack([g.stretch(0, 255).type("uint8", 0).data for g in stack.grids[:3]], axis=-1)
    tensor = from_numpy(np.transpose(rgb, (2, 0, 1))).to(model.device)
    state = processor.set_image(tensor)

    for label in prompt:
        output = processor.set_text_prompt(label, state)
        for m, s in zip(output["masks"].cpu().numpy(), output["scores"].cpu().numpy(), strict=True):
            g = grid(m[0], extent=stack.extent, crs=stack.crs)
            yield SamMask(label=label, score=float(s), mask=g.clip(g.set_nan(0).data_extent))


def _same_object(m1: SamMask, m2: SamMask, ios_threshold: float) -> bool:
    try:
        if m1.label != m2.label:
            return False
        if not m1.mask.extent.intersects(m2.mask.extent):
            return False
        g1, g2 = standardize(m1.mask, m2.mask, extent="union")
        intersection_area = (g1 & g2).sum
        if intersection_area == 0:
            return False
        return intersection_area / min(g1.sum, g2.sum) > ios_threshold
    except Exception as ex:
        logging.warning(f"Error calculating intersection over smaller: {ex}")
        return False
