import logging
import warnings
from dataclasses import dataclass
from types import SimpleNamespace
from typing import overload

import rasterio
from rasterio.crs import CRS
from rasterio.errors import NotGeoreferencedWarning
from rasterio.transform import Affine, array_bounds

from glidergun._grid import Grid, grid
from glidergun._stack import Stack, stack
from glidergun._types import Extent

logger = logging.getLogger(__name__)


@dataclass
class Profile:
    count: int
    crs: CRS
    height: int
    width: int
    transform: Affine


class Mosaic:
    def __init__(self, *files: str, blend: bool = False) -> None:
        assert files, "No files provided"
        profiles = list(self._read_profiles(*files))
        count_set = {p.count for _, p in profiles}
        crs_set = {p.crs for _, p in profiles}
        assert len(count_set) == 1, "Inconsistent number of bands"
        assert len(crs_set) == 1, "Inconsistent CRS"
        self.crs = crs_set.pop()
        self.files: dict[str, Extent] = {f: Extent(*array_bounds(p.height, p.width, p.transform)) for f, p in profiles}
        self.blend = blend
        self.extent = Extent(
            min(e.xmin for e in self.files.values()),
            min(e.ymin for e in self.files.values()),
            max(e.xmax for e in self.files.values()),
            max(e.ymax for e in self.files.values()),
        )

    def _read_profiles(self, *files: str):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=NotGeoreferencedWarning)
            for f in files:
                with rasterio.open(f) as dataset:
                    yield f, SimpleNamespace(**dataset.profile)

    def _read(self, extent: tuple[float, float, float, float] | list[float], index: int):
        for i, (f, e) in enumerate(self.files.items()):
            logger.info(f"Processing tile {i + 1} of {len(self.files)}...")
            try:
                if e.intersects(extent):
                    yield grid(f, extent, index=index)
            except Exception as ex:
                logger.warning(f"Failed to read {f} for extent {extent}: {ex}")

    def tiles(
        self,
        width: float,
        height: float,
        clip_extent: tuple[float, float, float, float] | list[float] | None = None,
    ):
        """Iterate over clipped mosaicked tiles of the requested size.

        Args:
            width: Tile width in coordinate units.
            height: Tile height in coordinate units.
            clip_extent: Optional extent to clip tiles to; defaults to full.

        Yields:
            Grid: Each tile as a `Grid`.
        """
        extent = Extent(*clip_extent) if clip_extent else self.extent
        for e in extent.tiles(width, height):
            g = self.clip(e)
            assert g
            yield g

    @overload
    def clip(self, extent: tuple[float, float, float, float] | list[float], index: int = 1) -> Grid | None: ...

    @overload
    def clip(self, extent: tuple[float, float, float, float] | list[float], index: tuple[int, ...]) -> Stack | None: ...

    def clip(self, extent: tuple[float, float, float, float] | list[float], index=None):
        try:
            if not index or isinstance(index, int):
                return mosaic(*(g for g in self._read(extent, index or 1) if g), blend=self.blend)
            return stack(*(self.clip(extent, index=i) for i in index))
        except Exception as ex:
            logger.warning(f"Failed to clip mosaic for extent {extent} and index {index}: {ex}")
            return None


@overload
def mosaic(*items: str, blend: bool = False) -> Mosaic: ...


@overload
def mosaic(*items: Grid, blend: bool = False) -> Grid: ...


@overload
def mosaic(*items: Stack, blend: bool = False) -> Stack: ...


def mosaic(*items, blend: bool = False):
    g = items[0]
    if isinstance(g, str):
        return Mosaic(*items, blend=blend)
    return g.mosaic(*items[1:], blend=blend)
