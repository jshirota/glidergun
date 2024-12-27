from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional, Tuple, Union, cast

import numpy as np

from numpy import ndarray
from rasterio.crs import CRS
from scipy.interpolate import (
    CloughTocher2DInterpolator,
    LinearNDInterpolator,
    NearestNDInterpolator,
    RBFInterpolator,
)

from glidergun._literals import InterpolationKernel

if TYPE_CHECKING:
    from glidergun._grid import Grid


@dataclass(frozen=True)
class Interpolation:
    def interp_clough_tocher(
        self,
        cell_size: Union[Tuple[float, float], float, None] = None,
        fill_value: float = np.nan,
        tol: float = 0.000001,
        maxiter: int = 400,
        rescale: bool = False,
    ):
        def f(coords, values):
            return CloughTocher2DInterpolator(
                coords, values, fill_value, tol, maxiter, rescale
            )

        g = cast("Grid", self)
        return interpolate(f, g.to_points(), g.extent, g.crs, cell_size or g.cell_size)

    def interp_linear(
        self,
        cell_size: Union[Tuple[float, float], float, None] = None,
        fill_value: float = np.nan,
        rescale: bool = False,
    ):
        def f(coords, values):
            return LinearNDInterpolator(coords, values, fill_value, rescale)

        g = cast("Grid", self)
        return interpolate(f, g.to_points(), g.extent, g.crs, cell_size or g.cell_size)

    def interp_nearest(
        self,
        cell_size: Union[Tuple[float, float], float, None] = None,
        rescale: bool = False,
        tree_options: Any = None,
    ):
        def f(coords, values):
            return NearestNDInterpolator(coords, values, rescale, tree_options)

        g = cast("Grid", self)
        return interpolate(f, g.to_points(), g.extent, g.crs, cell_size or g.cell_size)

    def interp_rbf(
        self,
        cell_size: Union[Tuple[float, float], float, None] = None,
        neighbors: Optional[int] = None,
        smoothing: float = 0,
        kernel: InterpolationKernel = "thin_plate_spline",
        epsilon: float = 1,
        degree: Optional[int] = None,
    ):
        def f(coords, values):
            return RBFInterpolator(
                coords, values, neighbors, smoothing, kernel, epsilon, degree
            )

        g = cast("Grid", self)
        return interpolate(f, g.to_points(), g.extent, g.crs, cell_size or g.cell_size)


def interpolate(
    interpolator_factory: Callable[[ndarray, ndarray], Any],
    points: Iterable[Tuple[float, float, float]],
    extent: Tuple[float, float, float, float],
    crs: Union[int, CRS],
    cell_size: Union[Tuple[float, float], float],
):
    from glidergun._functions import create

    points = list(points)
    coords = np.array([p[:2] for p in points])
    values = np.array([p[2] for p in points])
    g = create(extent, crs, cell_size)
    interp = interpolator_factory(coords, values)
    xs = np.linspace(g.xmin, g.xmax, g.width)
    ys = np.linspace(g.ymax, g.ymin, g.height)
    array = np.array([[x0, y0] for x0 in xs for y0 in ys])
    data = interp(array).reshape((g.width, g.height)).transpose(1, 0)
    return g.local(lambda _: data)
