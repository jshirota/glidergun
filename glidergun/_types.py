from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, NamedTuple, Protocol, TypedDict

from numpy import ndarray
from pyparsing import Any
from rasterio.crs import CRS
from rasterio.transform import Affine


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

    @property
    def is_valid(self):
        e = 1e-9
        return self.xmax - self.xmin > e and self.ymax - self.ymin > e

    def assert_valid(self):
        assert self.is_valid, f"Invalid extent: {self}"

    def intersects(self, xmin: float, ymin: float, xmax: float, ymax: float):
        return self.xmin < xmax and self.xmax > xmin and self.ymin < ymax and self.ymax > ymin

    def intersect(self, extent: "Extent"):
        return Extent(*[f(x) for f, x in zip((max, max, min, min), zip(self, extent, strict=False), strict=False)])

    def union(self, extent: "Extent"):
        return Extent(*[f(x) for f, x in zip((min, min, max, max), zip(self, extent, strict=False), strict=False)])

    def tiles(self, width: float, height: float):
        xmin = self.xmin
        while xmin < self.xmax:
            xmax = xmin + width
            ymin = self.ymin
            while ymin < self.ymax:
                ymax = ymin + height
                extent = Extent(xmin, ymin, xmax, ymax) & self
                if extent.is_valid:
                    yield extent
                ymin = ymax
            xmin = xmax

    def buffer(self, width: float, height: float):
        return Extent(self.xmin - width, self.ymin - height, self.xmax + width, self.ymax + height)

    def __repr__(self):
        return show(self)

    __and__ = intersect
    __rand__ = __and__
    __or__ = union
    __ror__ = __or__


class CellSize(NamedTuple):
    x: float
    y: float

    def __mul__(self, n: object):
        if not isinstance(n, (float | int)):
            return NotImplemented
        return CellSize(self.x * n, self.y * n)

    def __rmul__(self, n: object):
        if not isinstance(n, (float | int)):
            return NotImplemented
        return CellSize(self.x * n, self.y * n)

    def __truediv__(self, n: float):
        return CellSize(self.x / n, self.y / n)

    def __repr__(self):
        return show(self)


class Feature(TypedDict):
    type: Literal["Feature"]
    geometry: dict[str, Any]
    properties: dict[str, Any]


class FeatureCollection(TypedDict):
    type: Literal["FeatureCollection"]
    features: list[Feature]


class PointValue(NamedTuple):
    x: float
    y: float
    value: float

    def __repr__(self):
        return show(self)


class Scaler(Protocol):
    fit: Callable
    transform: Callable
    fit_transform: Callable


def show(obj: tuple[float, ...]) -> str:
    return f"({', '.join(str(round(n, 6)) for n in obj)})"
