import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, cast

import numpy as np
from rasterio.crs import CRS
from shapely import Point, Polygon
from shapely.ops import unary_union

from glidergun._grid import Grid, con, grid
from glidergun._types import FeatureCollection
from glidergun._utils import get_geojson

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from glidergun._stack import Stack


@dataclass(frozen=True, repr=False, slots=True)
class SamResult:
    masks: list["SamMask"]
    source: "Stack"

    def to_points(self, crs: int | CRS | None = None) -> list[tuple[Point, str]]:
        return [(polygon.representative_point(), label) for polygon, label in self.to_polygons(crs)]

    def to_polygons(self, crs: int | CRS | None = None, smooth_factor: float = 0.0) -> list[tuple[Polygon, str]]:
        d: dict[str, list[SamMask]] = {}
        for m in self.masks:
            d.setdefault(m.label, []).append(m)
        return [
            (polygon, label)
            for label, masks in d.items()
            for polygon in unary_union([m.to_polygon(crs, smooth_factor) for m in masks]).geoms  # type: ignore
        ]

    def to_geojson(self, crs: int | CRS | None = 4326, smooth_factor: float = 0.0) -> FeatureCollection:
        return get_geojson((polygon, {"label": label}) for polygon, label in self.to_polygons(crs, smooth_factor))

    def highlight(self, *labels: str) -> "Stack":
        return self.source.each(lambda g: con(self.union_mask(*labels), g, g / 5)).type("uint8", 0)

    def union_mask(self, *labels: str) -> Grid:
        polygons = [(polygon, 1) for polygon, label in self.to_polygons() if not labels or label in labels]
        return grid(polygons, self.source.extent, self.source.crs, self.source.cell_size) == 1


@dataclass(frozen=True, slots=True)
class SamMask:
    label: str
    score: float
    mask: Grid

    def to_polygon(self, crs: int | CRS | None = None, smooth_factor: float = 0.0):
        g = self.mask.set_nan(0)
        if crs:
            g = g.project(crs)
        polygons = [polygon for polygon, value in g.to_polygons(smooth_factor=smooth_factor) if value == 1]
        return max(polygons, key=lambda p: p.area)


@dataclass(frozen=True)
class Sam:
    def sam3(self, *prompt: str, model=None, confidence_threshold: float = 0.5, tile_size: int = 1000):
        """Run Segment Anything Model 3 (SAM 3) over the stack with text prompts.

        Args:
            prompt: One or more text prompts used for segmentation.
            model: Optional pre-built SAM 3 model; built and cached if None.
            confidence_threshold: Minimum confidence to accept a predicted mask.

        Returns:
            SamResult: Collection of masks and an overview visualization stack.
        """
        self = cast("Stack", self)

        w, h = self.cell_size * tile_size
        tiles = list(self.extent.tiles(w, h))
        masks = []

        for i, tile in enumerate(tiles):
            logger.info(f"Processing tile {i + 1} of {len(tiles)}...")
            s = self.clip(*tile.buffer(w / 4, h / 4, w / 4, h / 4))
            masks.extend(_execute_sam3(s, *prompt, model=model, confidence_threshold=confidence_threshold))

        return SamResult(masks=masks, source=self)


@lru_cache(maxsize=1)
def _build_sam3():
    import os
    import site
    import urllib.request

    from huggingface_hub import HfFolder, login
    from sam3.model_builder import build_sam3_image_model

    dir = f"{site.getsitepackages()[0]}/assets"
    file = "bpe_simple_vocab_16e6.txt.gz"

    if not os.path.exists(f"{dir}/{file}"):
        os.makedirs(dir, exist_ok=True)
        url = f"https://raw.githubusercontent.com/facebookresearch/sam3/refs/heads/main/assets/{file}"
        urllib.request.urlretrieve(url, f"{dir}/{file}")

    if not HfFolder.get_token():
        login()

    return build_sam3_image_model()


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
            yield SamMask(label=label, score=float(s), mask=g.clip(*g.set_nan(0).data_extent))
