import io
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import NamedTuple, Protocol

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from rasterio.crs import CRS
from rasterio.transform import Affine
from shapely import box

from glidergun.utils import get_crs


@dataclass(frozen=True)
class GridCore:
    data: ndarray
    transform: Affine
    crs: CRS


@dataclass(frozen=True)
class Extent:
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    crs: CRS

    def __init__(self, xmin: float, ymin: float, xmax: float, ymax: float, crs: int | str | CRS) -> None:
        object.__setattr__(self, "xmin", xmin)
        object.__setattr__(self, "ymin", ymin)
        object.__setattr__(self, "xmax", xmax)
        object.__setattr__(self, "ymax", ymax)
        object.__setattr__(self, "crs", get_crs(crs))

    def __iter__(self):
        yield self.xmin
        yield self.ymin
        yield self.xmax
        yield self.ymax

    def __len__(self):
        return 4

    def __getitem__(self, i):
        return (self.xmin, self.ymin, self.xmax, self.ymax)[i]

    @property
    def width(self):
        return self.xmax - self.xmin

    @property
    def height(self):
        return self.ymax - self.ymin

    @property
    def center(self):
        return (self.xmin + self.xmax) / 2, (self.ymin + self.ymax) / 2

    @property
    def is_valid(self):
        e = 1e-9
        return self.xmax - self.xmin > e and self.ymax - self.ymin > e

    def contains(self, extent: "BBox"):
        xmin, ymin, xmax, ymax = extent
        return self.xmin <= xmin and self.xmax >= xmax and self.ymin <= ymin and self.ymax >= ymax

    def intersects(self, extent: "BBox"):
        xmin, ymin, xmax, ymax = extent
        return self.xmin < xmax and self.xmax > xmin and self.ymin < ymax and self.ymax > ymin

    def intersect(self, extent: "BBox"):
        return Extent(
            *[f(x) for f, x in zip((max, max, min, min), zip(self, extent, strict=True), strict=True)], crs=self.crs
        )

    def union(self, extent: "BBox"):
        return Extent(
            *[f(x) for f, x in zip((min, min, max, max), zip(self, extent, strict=True), strict=True)], crs=self.crs
        )

    def project(self, crs: int | str | CRS):
        from glidergun.utils import project

        xmin, ymin = project(self.xmin, self.ymin, self.crs, crs)
        xmax, ymax = project(self.xmax, self.ymax, self.crs, crs)
        return Extent(xmin, ymin, xmax, ymax, crs)

    def tiles(self, width: float, height: float):
        x_edges = np.arange(self.xmin, self.xmax, width)
        y_edges = np.arange(self.ymin, self.ymax, height)
        x_max_edges = x_edges + width
        y_max_edges = y_edges + height
        extents: list[Extent] = []
        for xmin, xmax in zip(x_edges, x_max_edges, strict=True):
            for ymin, ymax in zip(y_edges, y_max_edges, strict=True):
                extent = Extent(float(xmin), float(ymin), float(xmax), float(ymax), self.crs) & self
                if extent.is_valid:
                    extents.append(extent)
        return extents

    def adjust(self, xmin: float = 0.0, ymin: float = 0.0, xmax: float = 0.0, ymax: float = 0.0):
        extent = Extent(self.xmin + xmin, self.ymin + ymin, self.xmax + xmax, self.ymax + ymax, self.crs)
        if not extent.is_valid:
            raise ValueError(f"Adjusted extent is not valid: {extent}")
        return extent

    def buffer(self, distance: float):
        return self.adjust(-distance, -distance, distance, distance)

    def buffer_by(self, factor: float):
        """Buffers the extent by a factor of its width and height."""
        x_shift = (self.width - self.width * factor) / 2
        y_shift = (self.height - self.height * factor) / 2
        return self.adjust(x_shift, y_shift, -x_shift, -y_shift)

    def to_polygon(self):
        return box(self.xmin, self.ymin, self.xmax, self.ymax)

    def __repr__(self):
        return show(self)

    __and__ = intersect
    __rand__ = __and__
    __or__ = union
    __ror__ = __or__


BBox = Extent | tuple[float, float, float, float] | list[float]


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


def show(obj: Iterable[float]) -> str:
    return f"({', '.join(str(round(n, 6)) for n in obj)})"
