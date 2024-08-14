from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, NamedTuple, Protocol

from numpy import ndarray
from rasterio.crs import CRS
from rasterio.transform import Affine

if TYPE_CHECKING:
    from glidergun._grid import Grid


@dataclass(frozen=True)
class GridCore:
    data: ndarray
    transform: Affine
    crs: CRS


class Extent(NamedTuple):
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def intersect(self, extent: "Extent"):
        return Extent(*[f(x) for f, x in zip((max, max, min, min), zip(self, extent))])

    def union(self, extent: "Extent"):
        return Extent(*[f(x) for f, x in zip((min, min, max, max), zip(self, extent))])

    __and__ = intersect
    __rand__ = __and__
    __or__ = union
    __ror__ = __or__


class CellSize(NamedTuple):
    x: float
    y: float

    def __mul__(self, n: object):
        if not isinstance(n, (float, int)):
            return NotImplemented
        return CellSize(self.x * n, self.y * n)

    def __rmul__(self, n: object):
        if not isinstance(n, (float, int)):
            return NotImplemented
        return CellSize(self.x * n, self.y * n)

    def __truediv__(self, n: float):
        return CellSize(self.x / n, self.y / n)


class Point(NamedTuple):
    x: float
    y: float
    value: float


class Scaler(Protocol):
    fit: Callable
    transform: Callable
    fit_transform: Callable


class StatsResult(NamedTuple):
    statistic: "Grid"
    pvalue: "Grid"
