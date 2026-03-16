import logging
from dataclasses import dataclass

import numpy as np
from rasterio.crs import CRS

from glidergun.geojson import FeatureCollection
from glidergun.grid import Grid, con, grid, standardize
from glidergun.stack import Stack

logger = logging.getLogger(__name__)


@dataclass(frozen=True, repr=False, slots=True)
class SamResult:
    masks: list["SamMask"]
    source: Stack

    def mask(self, *labels: str) -> Grid:
        polygons = [(mask.to_polygon(), 1) for mask in self.masks if not labels or mask.label in labels]
        return grid(polygons, self.source.extent, self.source.crs, self.source.cell_size) == 1

    def highlight(self, *labels: str) -> Stack:
        return self.source.each(lambda g: con(self.mask(*labels), g, g / 5)).type("uint8", 0)

    def to_geojson(self):
        return FeatureCollection((m.to_polygon(4326), {"label": m.label, "score": m.score}) for m in self.masks)


@dataclass(frozen=True, slots=True)
class SamMask:
    label: str
    score: float
    mask: Grid

    def to_polygon(self, crs: int | str | CRS | None = None):
        g = self.mask.set_nan(0)
        if crs:
            g = g.project(crs)
        polygons = [polygon for polygon, value in g.to_polygons() if value == 1]
        return max(polygons, key=lambda p: p.area)


def sam(
    stack: Stack,
    *prompt: str,
    checkpoint_path: str | None = None,
    hf_token: str | None = None,
    confidence_threshold: float = 0.5,
    tile_size: int = 1024,
):
    """Run Segment Anything Model 3 (SAM 3) over the stack with text prompts.

    Args:
        prompt: One or more text prompts used for segmentation.
        checkpoint_path: Optional path to a pre-trained SAM 3 checkpoint.
        hf_token: Optional Hugging Face token.
        confidence_threshold: Minimum confidence to accept a predicted mask.

    Returns:
        SamResult: Collection of masks and an overview visualization stack.
    """
    from scipy.cluster.hierarchy import DisjointSet

    buffer = 0.1
    ios_threshold = 0.5

    w, h = stack.cell_size * tile_size
    tiles = [e.adjust(-w * buffer, -h * buffer, w * buffer, h * buffer) for e in stack.extent.tiles(w, h)]

    all_masks: list[SamMask] = []
    for i, tile in enumerate(tiles):
        logger.info(f"Processing tile {i + 1} of {len(tiles)}...")
        all_masks.extend(
            _execute_sam(
                stack.clip(tile),
                *prompt,
                checkpoint_or_hf_token=checkpoint_path or hf_token,
                confidence_threshold=confidence_threshold,
            )
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

    return SamResult(masks=sorted(masks, key=lambda m: m.score, reverse=True), source=stack)


def _execute_sam(
    stack: Stack, *prompt: str, checkpoint_or_hf_token: str | None = None, confidence_threshold: float = 0.5
):
    from sam_prompt.model_builder import build_prompt_function

    rgb = np.stack([g.stretch(0, 255).type("uint8", 0).data for g in stack.grids[:3]], axis=-1)
    evaluate = build_prompt_function(rgb, confidence_threshold, checkpoint_or_hf_token)

    for label in prompt:
        for m, s in evaluate(label):
            g = grid(m, extent=stack.extent, crs=stack.crs)
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
