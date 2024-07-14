from typing import Tuple, Union
from rasterio.io import MemoryFile

from glidergun._grid import Grid, grid, mosaic


class Mosaic:
    def __init__(self, *files: Union[str, MemoryFile, Grid]) -> None:
        self.files = list(files)

    def _read(self, extent: Tuple[float, float, float, float]):
        for f in self.files:
            try:
                yield f if isinstance(f, Grid) else grid(f, extent=extent)
            except ValueError:
                pass

    def clip(self, xmin: float, ymin: float, xmax: float, ymax: float):
        return mosaic(*self._read((xmin, ymin, xmax, ymax)))
