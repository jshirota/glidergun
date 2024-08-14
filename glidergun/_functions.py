
from typing import Any, Callable, Iterable, Optional, Tuple, Union

import numpy as np
from numpy import ndarray
from rasterio.crs import CRS
from rasterio.transform import Affine
from scipy.interpolate import (LinearNDInterpolator, NearestNDInterpolator,
                               RBFInterpolator)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from glidergun._grid import Extent, Grid, con, grid, standardize
from glidergun._literals import InterpolationKernel
from glidergun._types import CellSize
from glidergun._utils import get_crs


def create(
    extent: Tuple[float, float, float, float],
    crs: Union[int, CRS],
    cell_size: Union[Tuple[float, float], float],
    value: float = 1.1,
):
    cell_size = (
        CellSize(cell_size, cell_size)
        if isinstance(cell_size, (int, float))
        else CellSize(*cell_size)
    )
    xmin, ymin, xmax, ymax = extent
    width = int((xmax - xmin) / cell_size.x)
    height = int((ymax - ymin) / cell_size.y)
    xmin = xmax - cell_size.x * width
    ymax = ymin + cell_size.y * height
    transform = Affine(cell_size.x, 0, xmin, 0, -cell_size.y, ymax, 0, 0, 1)
    g = grid(np.ones((height, width), "uint8"), transform, get_crs(crs))
    return g if value == 1.1 else g * value


def interpolate(
    interpolator_factory: Callable[[ndarray, ndarray], Any],
    points: Iterable[Tuple[float, float, float]],
    crs: Union[int, CRS],
    cell_size: Union[Tuple[float, float], float],
):
    coord_array = []
    value_array = []

    for p in points:
        coord_array.append(p[:2])
        value_array.append(p[-1])

    coords = np.array(coord_array)
    values = np.array(value_array)
    x, y = coords.transpose(1, 0)
    xmin, ymin, xmax, ymax = x.min(), y.min(), x.max(), y.max()

    x_buffer = (xmax - xmin) * 0.1
    y_buffer = (ymax - ymin) * 0.1
    xmin, ymin, xmax, ymax = (
        xmin - x_buffer,
        ymin - y_buffer,
        xmax + x_buffer,
        ymax + y_buffer,
    )

    extent = Extent(xmin, ymin, xmax, ymax)
    g = create(extent, crs, cell_size)
    interp = interpolator_factory(coords, values)
    xs = np.linspace(xmin, xmax, g.width)
    ys = np.linspace(ymax, ymin, g.height)
    array = np.array([[x0, y0] for x0 in xs for y0 in ys])
    data = interp(array).reshape((g.width, g.height)).transpose(1, 0)

    return g.local(lambda _: data)


def interp_linear(
    points: Iterable[Tuple[float, float, float]],
    crs: Union[int, CRS],
    cell_size: Union[Tuple[float, float], float],
    fill_value: float = np.nan,
    rescale: bool = False,
):
    def f(coords, values):
        return LinearNDInterpolator(coords, values, fill_value, rescale)

    return interpolate(f, points, crs, cell_size)


def interp_nearest(
    points: Iterable[Tuple[float, float, float]],
    crs: Union[int, CRS],
    cell_size: Union[Tuple[float, float], float],
    rescale: bool = False,
    tree_options: Any = None,
):
    def f(coords, values):
        return NearestNDInterpolator(coords, values, rescale, tree_options)

    return interpolate(f, points, crs, cell_size)


def interp_rbf(
    points: Iterable[Tuple[float, float, float]],
    crs: Union[int, CRS],
    cell_size: Union[Tuple[float, float], float],
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

    return interpolate(f, points, crs, cell_size)


def distance(
    extent: Tuple[float, float, float, float],
    crs: Union[int, CRS],
    cell_size: Union[Tuple[float, float], float],
    *points: Tuple[float, float],
):
    g = create(extent, crs, cell_size)

    if len(points) == 0:
        raise ValueError("Distance function requires at least one point.")

    if len(points) > 1000:
        raise ValueError("Distance function only accepts up to 1000 points.")

    if len(points) > 1:
        grids = [distance(extent, crs, cell_size, p) for p in points]
        return minimum(*grids)

    point = list(points)[0]
    w = int((g.extent.xmax - g.extent.xmin) / g.cell_size.x)
    h = int((g.extent.ymax - g.extent.ymin) / g.cell_size.y)
    dx = int((g.extent.xmin - point[0]) / g.cell_size.x)
    dy = int((point[1] - g.extent.ymax) / g.cell_size.y)
    data = np.meshgrid(
        np.array(range(dx, w + dx)) * g.cell_size.x,
        np.array(range(dy, h + dy)) * g.cell_size.y,
    )
    gx = grid(data[0], g.transform, g.crs)
    gy = grid(data[1], g.transform, g.crs)
    return (gx**2 + gy**2) ** (1 / 2)


def _aggregate(func: Callable, *grids: Grid) -> Grid:
    grids_adjusted = standardize(*grids)
    data = func(np.array([grid.data for grid in grids_adjusted]), axis=0)
    return grids_adjusted[0].update(data)


def mean(*grids: Grid) -> Grid:
    return _aggregate(np.mean, *grids)


def std(*grids: Grid) -> Grid:
    return _aggregate(np.std, *grids)


def minimum(*grids: Grid) -> Grid:
    return _aggregate(np.min, *grids)


def maximum(*grids: Grid) -> Grid:
    return _aggregate(np.max, *grids)


def pca(n_components: int = 1, *grids: Grid) -> Tuple[Grid, ...]:
    grids_adjusted = [con(g.is_nan(), float(g.mean), g)
                      for g in standardize(*grids)]
    arrays = (
        PCA(n_components=n_components)
        .fit_transform(
            np.array(
                [g.scale(StandardScaler()).data.ravel()
                 for g in grids_adjusted]
            ).transpose((1, 0))
        )
        .transpose((1, 0))
    )
    g = grids_adjusted[0]
    return tuple(g.update(a.reshape((g.height, g.width))) for a in arrays)
