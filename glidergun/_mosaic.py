from typing import List, Tuple, Union, overload

from rasterio.io import MemoryFile

from glidergun._grid import Grid, grid
from glidergun._stack import Stack, stack


class Mosaic:
    def __init__(self, *files: Union[str, MemoryFile, Grid]) -> None:
        self.files = list(files)

    def _read(self, extent: Tuple[float, float, float, float], index: int):
        for f in self.files:
            try:
                yield f if isinstance(f, Grid) else grid(f, extent, index=index)
            except ValueError:
                pass

    @overload
    def clip(
        self, xmin: float, ymin: float, xmax: float, ymax: float, index: int = 1
    ) -> Grid: ...

    @overload
    def clip(
        self, xmin: float, ymin: float, xmax: float, ymax: float, index: Tuple[int, ...]
    ) -> Stack: ...

    def clip(self, xmin: float, ymin: float, xmax: float, ymax: float, index=None):
        if not index or isinstance(index, int):
            grids: List[Grid] = [
                g for g in self._read((xmin, ymin, xmax, ymax), index or 1) if g
            ]
            return mosaic(*grids)
        return stack(*(self.clip(xmin, ymin, xmax, ymax, index=i) for i in index))


@overload
def mosaic(*grids: str) -> Mosaic:
    pass


@overload
def mosaic(*grids: Grid) -> Grid:
    pass


def mosaic(*grids):
    g = grids[0]
    if isinstance(g, str):
        return Mosaic(*grids)
    return g.mosaic(*grids[1:])
