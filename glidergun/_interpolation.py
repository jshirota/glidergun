from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

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
        points: Sequence[tuple[float, float, float]] | None = None,
        cell_size: tuple[float, float] | float | None = None,
        fill_value: float = np.nan,
        tol: float = 0.000001,
        maxiter: int = 400,
        rescale: bool = False,
    ):
        """Interpolate with Clough-Tocher 2D method over input points.

        Args:
            points: Sequence of `(x, y, value)` points; defaults to grid points.
            cell_size: Target cell size (tuple or scalar); defaults to current.
            fill_value: Value for outside convex hull / invalid.
            tol: Tolerance for Delaunay triangulation.
            maxiter: Maximum iterations.
            rescale: Whether to rescale inputs for numerical stability.

        Returns:
            Grid: Interpolated grid.
        """

        def f(coords, values):
            return CloughTocher2DInterpolator(coords, values, fill_value, tol, maxiter, rescale)

        self = cast("Grid", self)
        if points is None:
            points = self.to_points()
        return interpolate(f, points, self.extent, self.crs, cell_size or self.cell_size)

    def interp_linear(
        self,
        points: Sequence[tuple[float, float, float]] | None = None,
        cell_size: tuple[float, float] | float | None = None,
        fill_value: float = np.nan,
        rescale: bool = False,
    ):
        """Linear interpolation using scattered points.

        Args:
            points: Sequence of `(x, y, value)` points; defaults to grid points.
            cell_size: Target cell size (tuple or scalar); defaults to current.
            fill_value: Value for outside convex hull / invalid.
            rescale: Whether to rescale inputs for numerical stability.

        Returns:
            Grid: Interpolated grid.
        """

        def f(coords, values):
            return LinearNDInterpolator(coords, values, fill_value, rescale)

        self = cast("Grid", self)
        if points is None:
            points = self.to_points()
        return interpolate(f, points, self.extent, self.crs, cell_size or self.cell_size)

    def interp_nearest(
        self,
        points: Sequence[tuple[float, float, float]] | None = None,
        cell_size: tuple[float, float] | float | None = None,
        rescale: bool = False,
        tree_options: Any = None,
    ):
        """Nearest-neighbor interpolation of scattered points.

        Args:
            points: Sequence of `(x, y, value)` points; defaults to grid points.
            cell_size: Target cell size (tuple or scalar); defaults to current.
            rescale: Whether to rescale inputs for numerical stability.
            tree_options: Options passed to KD-tree construction.

        Returns:
            Grid: Interpolated grid.
        """

        def f(coords, values):
            return NearestNDInterpolator(coords, values, rescale, tree_options)

        self = cast("Grid", self)
        if points is None:
            points = self.to_points()
        return interpolate(f, points, self.extent, self.crs, cell_size or self.cell_size)

    def interp_rbf(
        self,
        points: Sequence[tuple[float, float, float]] | None = None,
        cell_size: tuple[float, float] | float | None = None,
        neighbors: int | None = None,
        smoothing: float = 0,
        kernel: InterpolationKernel = "thin_plate_spline",
        epsilon: float = 1,
        degree: int | None = None,
    ):
        """Radial basis function interpolation of scattered points.

        Args:
            points: Sequence of `(x, y, value)` points; defaults to grid points.
            cell_size: Target cell size (tuple or scalar); defaults to current.
            neighbors: Number of nearest neighbors to use; None uses all.
            smoothing: Regularization (0 means exact interpolation).
            kernel: RBF kernel name (e.g., 'thin_plate_spline').
            epsilon: Kernel parameter (scale).
            degree: Polynomial degree for certain kernels.

        Returns:
            Grid: Interpolated grid.
        """

        def f(coords, values):
            return RBFInterpolator(coords, values, neighbors, smoothing, kernel, epsilon, degree)

        self = cast("Grid", self)
        if points is None:
            points = self.to_points()
        return interpolate(f, points, self.extent, self.crs, cell_size or self.cell_size)


def interpolate(
    interpolator_factory: Callable[[ndarray, ndarray], Any],
    points: Sequence[tuple[float, float, float]],
    extent: tuple[float, float, float, float] | list[float],
    crs: int | str | CRS,
    cell_size: tuple[float, float] | float,
):
    """Common interpolation routine.

    Builds an interpolator from scattered `points` and evaluates it on grid
    coordinates derived from `extent` and `cell_size`.

    Args:
        interpolator_factory: Callable producing an interpolator given coords and values.
        points: Sequence of `(x, y, value)`.
        extent: Target grid extent.
        crs: Target CRS.
        cell_size: Target cell size.

    Returns:
        Grid: Interpolated grid.
    """
    from glidergun._grid import grid

    g = grid(np.nan, extent, crs, cell_size)

    if len(points) == 0:
        return g

    coords = np.array([p[:2] for p in points])
    values = np.array([p[2] for p in points])
    g = grid(np.nan, extent, crs, cell_size)
    interp = interpolator_factory(coords, values)
    xs = np.linspace(g.xmin, g.xmax, g.width)
    ys = np.linspace(g.ymax, g.ymin, g.height)
    array = np.array([[x0, y0] for x0 in xs for y0 in ys])
    data = interp(array).reshape((g.width, g.height)).transpose(1, 0)
    return g.local(data)
