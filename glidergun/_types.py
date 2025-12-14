import io
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, NamedTuple, Protocol, TypedDict

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from pyparsing import Any
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.warp import transform


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
    def width(self):
        return self.xmax - self.xmin

    @property
    def height(self):
        return self.ymax - self.ymin

    @property
    def is_valid(self):
        e = 1e-9
        return self.xmax - self.xmin > e and self.ymax - self.ymin > e

    def assert_valid(self):
        assert self.is_valid, f"Invalid extent: {self}"

    def contains(self, extent: tuple[float, float, float, float] | list[float]):
        xmin, ymin, xmax, ymax = extent
        return self.xmin <= xmin and self.xmax >= xmax and self.ymin <= ymin and self.ymax >= ymax

    def intersects(self, extent: tuple[float, float, float, float] | list[float]):
        xmin, ymin, xmax, ymax = extent
        return self.xmin < xmax and self.xmax > xmin and self.ymin < ymax and self.ymax > ymin

    def intersect(self, extent: tuple[float, float, float, float] | list[float]):
        return Extent(*[f(x) for f, x in zip((max, max, min, min), zip(self, extent, strict=False), strict=False)])

    def union(self, extent: tuple[float, float, float, float] | list[float]):
        return Extent(*[f(x) for f, x in zip((min, min, max, max), zip(self, extent, strict=False), strict=False)])

    def project(self, from_crs: int | str | CRS, to_crs: int | str | CRS):
        from glidergun._utils import get_crs

        projected = transform(get_crs(from_crs), get_crs(to_crs), [self.xmin, self.xmax], [self.ymin, self.ymax])
        [xmin, xmax], [ymin, ymax], *_ = projected
        return Extent(xmin, ymin, xmax, ymax)

    def tiles(self, width: float, height: float):
        extents: list[Extent] = []
        xmin = self.xmin
        while xmin < self.xmax:
            xmax = xmin + width
            ymin = self.ymin
            while ymin < self.ymax:
                ymax = ymin + height
                extent = Extent(xmin, ymin, xmax, ymax) & self
                if extent.is_valid:
                    extents.append(extent)
                ymin = ymax
            xmin = xmax
        return extents

    def adjust(self, xmin: float = 0.0, ymin: float = 0.0, xmax: float = 0.0, ymax: float = 0.0):
        extent = Extent(self.xmin + xmin, self.ymin + ymin, self.xmax + xmax, self.ymax + ymax)
        if not extent.is_valid:
            raise ValueError(f"Adjusted extent is not valid: {extent}")
        return extent

    def buffer(self, distance: float):
        return self.adjust(-distance, -distance, distance, distance)

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


@dataclass(frozen=True)
class Chart:
    figure: Figure
    axes: Axes

    def _repr_png_(self):
        with io.BytesIO() as buffer:
            self.figure.savefig(buffer, format="png", bbox_inches="tight")
            plt.close(self.figure)
            buffer.seek(0)
            return buffer.read()


def show(obj: tuple[float, ...]) -> str:
    return f"({', '.join(str(round(n, 6)) for n in obj)})"
