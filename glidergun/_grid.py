import dataclasses
import hashlib
import logging
import warnings
from base64 import b64encode
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from types import FunctionType
from typing import Any, Union, cast, overload

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.warp
from numpy import arctan, arctan2, cos, gradient, ndarray, pi, sin, sqrt
from rasterio import DatasetReader, features
from rasterio.crs import CRS
from rasterio.errors import NotGeoreferencedWarning
from rasterio.io import MemoryFile
from rasterio.transform import Affine
from rasterio.warp import Resampling, calculate_default_transform, reproject
from rasterio.windows import Window, from_bounds
from scipy.ndimage import distance_transform_edt, gaussian_filter, sobel
from shapely.geometry import Point, Polygon
from shapely.geometry.base import BaseGeometry
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from glidergun._focal import Focal
from glidergun._interpolation import Interpolation
from glidergun._literals import ColorMap, DataType, ExtentResolution, ResamplingMethod
from glidergun._types import CellSize, Chart, Extent, GridCore, PointValue, Scaler
from glidergun._utils import create_directory_for, format_type, get_crs, get_driver, get_geojson, get_nodata_value
from glidergun._zonal import Zonal

logger = logging.getLogger(__name__)

Operand = Union["Grid", float, int]


@dataclass(frozen=True)
class Grid(GridCore, Interpolation, Focal, Zonal):
    """Core grid class representing a 2D raster with geospatial metadata.

    Attributes:
        data (ndarray): 2D array of raster values.
        transform (Affine): Affine transformation mapping pixel to geographic coordinates.
        crs (CRS): Coordinate reference system of the grid.
    """

    display: ColorMap | Any = "gray"

    def __post_init__(self):
        self.data.flags.writeable = False

        if self.width * self.height == 0:
            raise ValueError("Empty raster.")

    def __repr__(self):
        d = 3 if self.dtype.startswith("float") else 0
        return (
            f"image: {self.width}x{self.height} {self.dtype} | "
            + f"range: {self.min:.{d}f}~{self.max:.{d}f} | "
            + f"mean: {self.mean:.{d}f} | "
            + f"std: {self.std:.{d}f} | "
            + f"crs: {self.crs} | "
            + f"cell: {self.cell_size} | "
            + f"extent: {self.extent}"
        )

    def _thumbnail(self, figsize: tuple[float, float] | None = None):
        """Render a small PNG preview of the grid for notebook display."""
        with BytesIO() as buffer:
            figure = plt.figure(figsize=figsize, frameon=False)
            axes = figure.add_axes((0, 0, 1, 1))
            axes.axis("off")
            obj = _to_uint8_range(self)
            plt.imshow(obj.data, cmap=self.display)
            plt.savefig(buffer, bbox_inches="tight", pad_inches=0)
            plt.close(figure)
            return buffer.getvalue()

    @cached_property
    def img(self) -> str:
        image = b64encode(self._thumbnail()).decode()
        return f"data:image/png;base64, {image}"

    @cached_property
    def width(self) -> int:
        return self.data.shape[1]

    @cached_property
    def height(self) -> int:
        return self.data.shape[0]

    @cached_property
    def dtype(self) -> DataType:
        return cast(DataType, str(self.data.dtype))

    @cached_property
    def nodata(self):
        return get_nodata_value(self.dtype)

    @cached_property
    def has_nan(self):
        return bool(self.is_nan().data.any())

    @cached_property
    def xmin(self) -> float:
        return self.extent.xmin

    @cached_property
    def ymin(self) -> float:
        return self.extent.ymin

    @cached_property
    def xmax(self) -> float:
        return self.extent.xmax

    @cached_property
    def ymax(self) -> float:
        return self.extent.ymax

    @cached_property
    def extent(self) -> Extent:
        return _extent(self.width, self.height, self.transform)

    @cached_property
    def mean(self):
        return float(np.nanmean(self.data))

    @cached_property
    def std(self):
        return float(np.nanstd(self.data))

    @cached_property
    def min(self):
        return float(np.nanmin(self.data))

    @cached_property
    def max(self):
        return float(np.nanmax(self.data))

    @cached_property
    def sum(self):
        return float(np.nansum(self.data))

    @cached_property
    def cell_size(self) -> CellSize:
        return CellSize(self.transform.a, -self.transform.e)

    @cached_property
    def bins(self) -> dict[float, int]:
        unique, counts = np.unique(self.data, return_counts=True)
        return dict(sorted(zip(map(float, unique), map(int, counts), strict=False)))

    @cached_property
    def md5(self) -> str:
        return hashlib.md5(self.data.copy(order="C")).hexdigest()

    def __add__(self, n: Operand):
        return self._apply(self, n, np.add)

    __radd__ = __add__

    def __sub__(self, n: Operand):
        return self._apply(self, n, np.subtract)

    def __rsub__(self, n: Operand):
        return self._apply(n, self, np.subtract)

    def __mul__(self, n: Operand):
        return self._apply(self, n, np.multiply)

    __rmul__ = __mul__

    def __pow__(self, n: Operand):
        return self._apply(self, n, np.power)

    def __rpow__(self, n: Operand):
        return self._apply(n, self, np.power)

    def __truediv__(self, n: Operand):
        return self._apply(self, n, np.true_divide)

    def __rtruediv__(self, n: Operand):
        return self._apply(n, self, np.true_divide)

    def __floordiv__(self, n: Operand):
        return self._apply(self, n, np.floor_divide)

    def __rfloordiv__(self, n: Operand):
        return self._apply(n, self, np.floor_divide)

    def __mod__(self, n: Operand):
        return self._apply(self, n, np.mod)

    def __rmod__(self, n: Operand):
        return self._apply(n, self, np.mod)

    def __lt__(self, n: Operand):
        return self._apply(self, n, np.less)

    def __gt__(self, n: Operand):
        return self._apply(self, n, np.greater)

    __rlt__ = __gt__

    __rgt__ = __lt__

    def __le__(self, n: Operand):
        return self._apply(self, n, np.less_equal)

    def __ge__(self, n: Operand):
        return self._apply(self, n, np.greater_equal)

    __rle__ = __ge__

    __rge__ = __le__

    def __eq__(self, n: object):
        if not isinstance(n, (Grid | float | int)):
            return NotImplemented
        return self._apply(self, n, np.equal)

    __req__ = __eq__

    def __ne__(self, n: object):
        if not isinstance(n, (Grid | float | int)):
            return NotImplemented
        return self._apply(self, n, np.not_equal)

    __rne__ = __ne__

    def __and__(self, n: Operand):
        return self._apply(self, n, np.bitwise_and)

    __rand__ = __and__

    def __or__(self, n: Operand):
        return self._apply(self, n, np.bitwise_or)

    __ror__ = __or__

    def __xor__(self, n: Operand):
        return self._apply(self, n, np.bitwise_xor)

    __rxor__ = __xor__

    def __rshift__(self, n: Operand):
        return self._apply(self, n, np.right_shift)

    def __lshift__(self, n: Operand):
        return self._apply(self, n, np.left_shift)

    __rrshift__ = __lshift__

    __rlshift__ = __rshift__

    def __neg__(self):
        return self.local(-1 * self.data)

    def __pos__(self):
        return self.local(1 * self.data)

    def __invert__(self):
        return con(self, False, True)

    def is_greater_than(self, n: Operand):
        return self > n

    def is_less_than(self, n: Operand):
        return self < n

    def is_greater_than_or_equal(self, n: Operand):
        return self >= n

    def is_less_than_or_equal(self, n: Operand):
        return self <= n

    def is_equal(self, n: Operand):
        return self == n

    def is_not_equal(self, n: Operand):
        return self != n

    def _data(self, n: Operand):
        if isinstance(n, Grid):
            return n.data
        return n

    def _apply(self, left: Operand, right: Operand, op: Callable):
        if not isinstance(left, Grid) or not isinstance(right, Grid):
            return self.local(op(self._data(left), self._data(right)))

        if left.cell_size == right.cell_size and left.extent == right.extent:
            return self.local(op(left.data, right.data))

        l_adjusted, r_adjusted = standardize(left, right)

        return self.local(op(l_adjusted.data, r_adjusted.data))

    def local(self, func: Callable[[ndarray], ndarray] | ndarray):
        """Apply a numpy function to the data and return a new Grid."""
        data = func if isinstance(func, ndarray) else func(self.data)
        return grid(data, self.transform, self.crs)

    def mosaic(self, *grids: "Grid"):
        """Union multiple grids onto this grid's canvas, respecting NaNs."""
        grids_adjusted = standardize(self, *grids, extent="union")
        result = grids_adjusted[0]
        for g in grids_adjusted[1:]:
            result = con(result.is_nan(), g, result)
        return result

    def standardize(self, *grids: "Grid"):
        """Standardize multiple grids to this grid's extent and cell size."""
        return standardize(self, *grids)

    def is_nan(self):
        """Return a boolean grid where True indicates NaN cells."""
        return self.local(np.isnan)

    def abs(self):
        """Absolute value per cell."""
        return self.local(np.abs)

    def sin(self):
        """Sine per cell (radians)."""
        return self.local(np.sin)

    def cos(self):
        """Cosine per cell (radians)."""
        return self.local(np.cos)

    def tan(self):
        """Tangent per cell (radians)."""
        return self.local(np.tan)

    def arcsin(self):
        """Arcsine per cell (returns radians)."""
        return self.local(np.arcsin)

    def arccos(self):
        """Arccos per cell (returns radians)."""
        return self.local(np.arccos)

    def arctan(self):
        """Arctangent per cell (returns radians)."""
        return self.local(np.arctan)

    def log(self, base: float | None = None):
        """Natural log per cell, or change-of-base if `base` provided."""
        if base is None:
            return self.local(np.log)
        return self.local(lambda a: np.log(a) / np.log(base))

    def round(self, decimals: int = 0):
        """Round per cell to `decimals`."""
        return self.local(lambda a: np.round(a, decimals))

    def georeference(self, extent: tuple[float, float, float, float] | list[float], crs: int | str | CRS | None = None):
        """Assign extent and CRS to the current data without resampling."""
        return grid(self.data, extent, get_crs(crs) if crs else self.crs)

    def _reproject(
        self,
        transform,
        crs,
        width,
        height,
        resampling: Resampling | ResamplingMethod,
    ) -> "Grid":
        source = self * 1 if self.dtype == "bool" else self
        destination = np.ones((round(height), round(width))) * np.nan
        reproject(
            source=source.data,
            destination=destination,
            src_transform=self.transform,
            src_crs=self.crs,
            src_nodata=self.nodata,
            dst_transform=transform,
            dst_crs=crs,
            dst_nodata=np.nan,
            resampling=(Resampling[resampling] if isinstance(resampling, str) else resampling),
        )
        result = grid(destination, transform, crs)
        if self.dtype == "bool":
            return result == 1
        return con(result == result.nodata, np.nan, result)

    def project(
        self,
        crs: int | str | CRS,
        resampling: Resampling | ResamplingMethod = "nearest",
    ) -> "Grid":
        """Reproject to a new CRS using the specified resampling method."""
        crs = get_crs(crs)
        if crs == self.crs:
            return self.type("float32")
        transform, width, height = calculate_default_transform(self.crs, crs, self.width, self.height, *self.extent)
        return self._reproject(
            transform,
            crs,
            width,
            height,
            Resampling[resampling] if isinstance(resampling, str) else resampling,
        )

    def _resample(
        self,
        extent: tuple[float, float, float, float] | list[float],
        cell_size: tuple[float, float],
        resampling: Resampling | ResamplingMethod,
    ) -> "Grid":
        xmin, ymin, xmax, ymax = _adjust_extent(extent, self.transform)
        xoff = (xmin - self.xmin) / self.transform.a
        yoff = (ymax - self.ymax) / self.transform.e
        scaling_x = cell_size[0] / self.cell_size.x
        scaling_y = cell_size[1] / self.cell_size.y
        transform = self.transform * Affine.translation(xoff, yoff) * Affine.scale(scaling_x, scaling_y)
        width = (xmax - xmin) / abs(self.transform.a) / scaling_x
        height = (ymax - ymin) / abs(self.transform.e) / scaling_y
        return self._reproject(transform, self.crs, width, height, resampling)

    def clip(self, extent: tuple[float, float, float, float] | list[float]):
        """Clip to the given extent (same CRS), preserving cell size."""
        return self._resample(extent, self.cell_size, Resampling.nearest)

    def clip_at(self, x: float, y: float, width: int = 8, height: int = 8):
        """Clip a window centered at `(x,y)` with given pixel width/height."""
        x_offset = self.cell_size.x * width / 2
        y_offset = self.cell_size.y * height / 2
        xmin = x - x_offset
        ymin = y - y_offset
        xmax = x + x_offset
        ymax = y + y_offset
        return self.clip((xmin, ymin, xmax, ymax))

    def resample(
        self,
        cell_size: tuple[float, float] | float,
        resampling: Resampling | ResamplingMethod = "nearest",
    ):
        """Resample to a new cell size (tuple or scalar), same extent/CRS."""
        if isinstance(cell_size, (int | float)):
            cell_size = (cell_size, cell_size)
        if self.cell_size == cell_size:
            return self.type("float32")
        return self._resample(self.extent, cell_size, resampling)

    def resample_by(self, times: float, resampling: Resampling | ResamplingMethod = "nearest"):
        """Resample by a scaling factor, adjusting cell size accordingly."""
        return self.resample(self.cell_size * times, resampling)

    def resize(
        self,
        width: int,
        height: int,
        resampling: Resampling | ResamplingMethod = "nearest",
    ):
        """Resize to an exact cell count, adjusting cell size accordingly."""
        cell_size_x = self.cell_size.x * self.width / width
        cell_size_y = self.cell_size.y * self.height / height
        return self.resample((cell_size_x, cell_size_y), resampling)

    def buffer(self, value: float | int, count: int):
        """Grow/shrink regions of `value` by `count` cells using focal logic."""
        if count < 0:
            g = (self != value).buffer(1, -count)
            return con(g == 0, value, self.set_nan(self == value))
        g = self
        for _ in range(count):
            g = con(g.focal_count(value, 1, True) > 0, value, g)
        return g

    def heat_map(self, sigma: float = 5.0):
        density = self.con(self.is_nan(), 0.0).gaussian_filter(sigma).stretch(0.0, 1.0)
        return density.cap_min(0.1, True).color("turbo")

    def gaussian_filter(self, sigma: float = 1.0, **kwargs):
        """Apply Gaussian filter to the grid."""
        return self.local(lambda a: gaussian_filter(a, sigma=sigma, **kwargs))

    def sobel(self, axis: int = -1, **kwargs):
        """Apply Sobel filter to the grid."""
        return self.local(lambda a: sobel(a, axis=axis, **kwargs))

    def distance(self):
        """Euclidean distance from cells with positive values."""
        g = self.con(self.is_nan(), 0.0)
        x, y = g.cell_size
        cell_size = min(x, y)
        if x != y:
            g = g.resample(cell_size, Resampling.nearest)
        result = g.local(lambda a: distance_transform_edt(a == 0)) * cell_size  # type: ignore
        if x != y:
            return result.resample((x, y), Resampling.nearest)
        return result

    def interp_idw(
        self,
        points: Sequence[tuple[float, float, float]] | None = None,
        cell_size: tuple[float, float] | float | None = None,
        radius: float | None = None,
        max_workers: int = 1,
    ):
        """Inverse-distance weighting interpolation to a grid."""
        if points is None:
            points = self.to_points()
        return idw(
            points,
            self.extent,
            self.crs,
            cell_size or self.cell_size,
            radius,
            max_workers,
        )

    def randomize(self, normal_distribution: bool = False):
        """Fill with random values (uniform or normal) matching shape."""
        f = np.random.randn if normal_distribution else np.random.rand
        return self.local(f(self.height, self.width))

    def aspect(self, radians: bool = False):
        """Compute aspect (downslope direction) in degrees by default."""
        y, x = gradient(self.data)
        g = self.local(arctan2(-y, x))
        if radians:
            return g
        return (g.local(np.degrees) + 360) % 360

    def slope(self, radians: bool = False):
        """Compute slope from gradients; degrees by default."""
        y, x = gradient(self.data)
        g = self.local(arctan(sqrt(x * x + y * y)))
        if radians:
            return g
        return g.local(np.degrees)

    def hillshade(self, azimuth: float = 315, altitude: float = 45):
        """Simulate illumination given sun azimuth/altitude (degrees)."""
        azimuth = np.deg2rad(azimuth)
        altitude = np.deg2rad(altitude)
        aspect = self.aspect(True).data
        slope = (pi / 2.0 - self.slope(True)).data
        shaded = sin(altitude) * sin(slope) + cos(altitude) * cos(slope) * cos(azimuth - aspect)
        return self.local(255 * (shaded + 1) / 2)

    def con(
        self,
        predicate: Operand | Callable[["Grid"], "Grid"],
        replacement: Operand,
        fallback: Operand | None = None,
    ):
        """Conditional evaluation: if predicate then replacement else fallback."""
        if isinstance(predicate, FunctionType):
            g = predicate(self)
        elif isinstance(predicate, Grid):
            g = predicate
        else:
            g = self == predicate
        return con(self.standardize(g)[1], replacement, self if fallback is None else fallback)

    def set_nan(
        self,
        predicate: Operand | Callable[["Grid"], "Grid"],
        fallback: Operand | None = None,
    ):
        """Set cells matching predicate to NaN, optionally using fallback."""
        return self.con(predicate, np.nan, fallback)

    def then(self, true_value: Operand, false_value: Operand):
        """Alias for conditional: replace True/False cells with given values."""
        return con(self, true_value, false_value)

    def process_tiles(
        self,
        func: Callable[["Grid"], "Grid"],
        tile_size: int = 256,
        buffer: int = 0,
        max_workers: int = 1,
    ) -> "Grid":
        """Apply a function to tiles and mosaic results.

        Splits the grid into tiles of approximately `tile_size` cells per
        side (scaled by cell size), optionally expands each tile by `buffer`
        cells to reduce edge effects, applies `func` to each tile, and
        mosaics the results back into a single grid. Uses threads when
        `max_workers > 1`.
        """
        cell_x, cell_y = self.cell_size
        tiles = list(self.extent.tiles(cell_x * tile_size, cell_y * tile_size))

        if len(tiles) <= 4:
            return func(self)

        x = buffer * cell_x
        y = buffer * cell_y

        def f(tile: Extent):
            xmin, ymin, xmax, ymax = tile
            g = func(self.clip((xmin - x, ymin - y, xmax + x, ymax + y)))
            if buffer == 0:
                return g
            return g.clip((xmin, ymin, xmax, ymax))

        result: Grid | None = None

        if max_workers > 1:
            with ThreadPoolExecutor(max_workers or 1) as executor:
                for r in executor.map(f, tiles):
                    result = result.mosaic(r) if result else r
        else:
            for i, tile in enumerate(tiles):
                logger.info(f"Processing tile {i + 1} of {len(tiles)}...")
                result = result.mosaic(f(tile)) if result else f(tile)

        assert result
        return result

    def value_at(self, x: float, y: float) -> float:
        """Return the cell value at map coordinate `(x,y)` or NaN if out-of-bounds."""
        c = int((x - self.xmin) / self.cell_size.x)
        r = int((self.ymax - y) / self.cell_size.y)
        if c < 0 or c >= self.width or r < 0 or r >= self.height:
            return float(np.nan)
        return float(self.data[r, c])

    @cached_property
    def data_extent(self) -> Extent:
        if not self.has_nan:
            return self.extent
        rows = np.any(~np.isnan(self.data), axis=1)
        cols = np.any(~np.isnan(self.data), axis=0)
        row_min, row_max = np.where(rows)[0][[0, -1]]
        col_min, col_max = np.where(cols)[0][[0, -1]]
        data = self.data[row_min : row_max + 1, col_min : col_max + 1]
        transform = self.transform * rasterio.Affine.translation(col_min, row_min)
        return _extent(data.shape[1], data.shape[0], transform)

    def _xs(self):
        offset = self.cell_size.x / 2
        return np.linspace(self.xmin + offset, self.xmax - offset, self.width)

    def _ys(self):
        offset = self.cell_size.y / 2
        return np.linspace(self.ymax - offset, self.ymin + offset, self.height)

    def _coords(self):
        x, y = np.meshgrid(self._xs(), self._ys())
        return np.asarray(np.column_stack([x.ravel(), y.ravel()]), "float32")

    def _coords_with_values(self, include_nan: bool = False):
        x, y = np.meshgrid(self._xs(), self._ys())
        array = np.column_stack([x.ravel(), y.ravel(), self.data.ravel()])
        if include_nan:
            return array
        return array[~np.isnan(array[:, 2])]

    def to_points(self, include_nan: bool = False) -> list[PointValue]:
        """Convert cells to `(x,y,value)` points; optionally include NaNs."""
        return [PointValue(float(x), float(y), float(v)) for x, y, v in self._coords_with_values(include_nan)]

    def to_polygons(self, include_nan: bool = False, smooth_factor: float = 0.0) -> list[tuple[Polygon, float]]:
        """Polygonize regions of constant values; optionally include NaNs and smooth."""
        g = self * 1
        mask = None if include_nan else np.isfinite(g.data)
        buffer = min(g.cell_size) * smooth_factor
        results = []
        for shape, value in features.shapes(g.data, mask=mask, transform=g.transform):
            polygon = Polygon(shape["coordinates"][0], shape["coordinates"][1:])
            if buffer > 0:
                polygon = polygon.buffer(-buffer).buffer(buffer)
            results.append((polygon, float(value)))
        return results

    def to_geojson(self, polygonize: bool = False, include_nan: bool = False, smooth_factor: float = 0.0):
        """Export points or polygons as a GeoJSON FeatureCollection."""
        if polygonize:
            features = self.to_polygons(include_nan=include_nan, smooth_factor=smooth_factor)
        else:
            features = [(Point(x, y), value) for x, y, value in self.to_points(include_nan=include_nan)]
        return get_geojson((shape, {"value": None if np.isnan(value) else value}) for shape, value in features)

    def rasterize(
        self,
        items: Iterable[tuple[BaseGeometry, float] | tuple[float, float, float]],
        all_touched: bool = False,
    ):
        """Rasterize geometries or (x,y,value) tuples onto this grid's canvas."""

        def get_geometries():
            for item in items:
                if isinstance(item[0], BaseGeometry):
                    yield (item[0], float(item[1]))
                else:
                    yield (Point(item[:2]), float(item[2]))  # type: ignore

        array = features.rasterize(
            shapes=get_geometries(),
            out_shape=self.data.shape,
            fill=np.nan,  # type: ignore
            transform=self.transform,
            all_touched=all_touched,
            default_value=np.nan,  # type: ignore
        )
        return self.local(array)

    def to_stack(self):
        """Convert single-band grid to RGB stack using current colormap."""
        from glidergun._stack import stack

        grid1 = self.percent_clip(0.1, 99.9)
        grid2 = grid1 - grid1.min
        grid3 = grid2 / grid2.max
        arrays = plt.get_cmap(self.display)(grid3.data).transpose(2, 0, 1)[:3]
        mask = self.is_nan()
        r, g, b = [self.local(a * 253 + 1).set_nan(mask) for a in arrays]
        return stack([r, g, b])

    def percentile(self, percent: float) -> float:
        """Return the `percent` percentile (NaN-aware)."""
        return np.nanpercentile(self.data, percent)  # type: ignore

    def percent_clip(self, min_percent: float, max_percent: float):
        """Clip values to a percentile range, returning a new grid."""
        min_value = self.percentile(min_percent)
        max_value = self.percentile(max_percent)

        if min_value == max_value:
            return self.type("float32")

        return self.cap_range(min_value, max_value)

    def reclass(self, *mapping: tuple[float, float, float]):
        """Reclassify ranges to specific values based on `(min,max,value)` tuples."""
        conditions = [(self.data >= min) & (self.data < max) for min, max, _ in mapping]
        values = [value for _, _, value in mapping]
        return self.local(np.select(conditions, values, np.nan))

    def slice(self, count: int, percent_clip: float = 0.1):
        """Divide range into `count` equal slices and assign slice indices."""
        min = self.percentile(percent_clip)
        max = self.percentile(100 - percent_clip)
        interval = (max - min) / count
        mapping = [
            (
                min + (i - 1) * interval if i > 1 else float("-inf"),
                min + i * interval if i < count else float("inf"),
                float(i),
            )
            for i in range(1, count + 1)
        ]
        return self.reclass(*mapping)

    def stretch(self, min_value: float, max_value: float):
        """Linearly stretch values to `[min_value, max_value]`."""
        expected_range = max_value - min_value
        actual_range = self.max - self.min

        if actual_range == 0:
            n = (min_value + max_value) / 2
            return self * 0 + n

        return (self - self.min) * expected_range / actual_range + min_value

    def cap_range(self, min: Operand, max: Operand, set_nan: bool = False):
        """Cap values to `[min,max]`; optionally set outside to NaN."""
        return self.cap_min(min, set_nan).cap_max(max, set_nan)

    def cap_min(self, value: Operand, set_nan: bool = False):
        """Cap values below `value`; optionally set capped cells to NaN."""
        return con(self < value, np.nan if set_nan else value, self)

    def cap_max(self, value: Operand, set_nan: bool = False):
        """Cap values above `value`; optionally set capped cells to NaN."""
        return con(self > value, np.nan if set_nan else value, self)

    def kmeans_cluster(self, n_clusters: int, nodata: float = 0.0, **kwargs):
        """Cluster cell values via KMeans and return cluster-center grid."""
        g = self.is_nan().then(nodata, self)
        kmeans = KMeans(n_clusters=n_clusters, **kwargs).fit(g.data.reshape(-1, 1))
        data = kmeans.cluster_centers_[kmeans.labels_].reshape(self.data.shape)
        result = self.local(data)
        return result.set_nan(self.is_nan())

    def scale(self, scaler: Scaler, **fit_params):
        """Apply a scaler's `fit_transform` across the grid."""
        self = cast("Grid", self)
        return self.local(lambda a: scaler.fit_transform(a, **fit_params))

    def hist(self, **kwargs) -> Chart:
        """Build a histogram chart of value counts (NaN-aware)."""
        figure, axes = plt.subplots()
        kwargs.setdefault("color", "gray")
        axes.bar(list(self.bins.keys()), list(self.bins.values()), **kwargs)
        return Chart(figure, axes)

    def color(self, cmap: ColorMap | Any):
        """Return a Grid with a different display colormap (for previews/maps)."""
        return dataclasses.replace(self, display=cmap)

    def type(self, dtype: DataType, nan_to_num: float | None = None):
        """Convert dtype; optionally replace NaNs before casting."""
        if self.dtype == dtype:
            return self
        if nan_to_num is None:
            return self.local(lambda a: np.asanyarray(a, dtype=dtype))
        return self.local(lambda a: np.nan_to_num(a, nan=nan_to_num).astype(dtype))

    def to_bytes(self, dtype: DataType | None = None, driver: str = "") -> bytes:
        """Serialize grid to bytes (COG by default)."""
        with MemoryFile() as memory_file:
            self.save(memory_file, dtype, driver)
            return memory_file.read()

    def save(self, file: str | MemoryFile, dtype: DataType | None = None, driver: str = ""):
        """Saves the grid to disk or a `MemoryFile`.

        If the file extension is `.jpg`, `.kml`, `.kmz`, or `.png`, the grid
        is converted to an RGB stack using the current colormap and saved as
        an image. Otherwise, the grid is saved as a single-band raster
        using the specified `dtype` (or current dtype if not provided) and
        driver (or inferred from file extension if not provided).

        Args:
            file: File path or `MemoryFile` to save to.
            dtype: Data type for saving (optional).
            driver: Rasterio driver name (optional).
        """
        g = self * 1 if self.dtype == "bool" else self

        if isinstance(file, str) and (
            file.lower().endswith(".jpg")
            or file.lower().endswith(".kml")
            or file.lower().endswith(".kmz")
            or file.lower().endswith(".png")
        ):
            g.to_stack().save(file)
            return

        if dtype is None:
            dtype = g.dtype

        nodata = get_nodata_value(dtype)

        if nodata is not None:
            g = con(g.is_nan(), nodata, g)

        if isinstance(file, str):
            create_directory_for(file)
            with rasterio.open(
                file,
                "w",
                driver=driver or get_driver(file),
                count=1,
                dtype=dtype,
                nodata=nodata,
                **_metadata(self),
            ) as dataset:
                dataset.write(g.data, 1)
        elif isinstance(file, MemoryFile):
            with file.open(
                driver=driver or "COG",
                count=1,
                dtype=dtype,
                nodata=nodata,
                **_metadata(self),
            ) as dataset:
                dataset.write(g.data, 1)


@overload
def grid(
    data: str,
    extent: tuple[float, float, float, float] | list[float] | None = None,
    crs: int | str | CRS | None = None,
    cell_size: tuple[float, float] | float | None = None,
    index: int = 1,
) -> Grid:
    """Creates a new grid from a file path or url.

    Args:
        data: File path.
        extent: Map extent used to clip the raster.
        crs: CRS or an EPSG code (e.g. 4326).
        index (int, optional): Band index.  Defaults to 1.

    Example:
        >>> grid("n55_e008_1arc_v3.bil")
        image: 1801x3601 float32 | range: -28.000~101.000 | mean: 11.566 | std: 14.645 | crs: EPSG:4326 | cell: ...

    Returns:
        Grid: A new grid.
    """
    ...


@overload
def grid(  # type: ignore
    data: DatasetReader,
    extent: tuple[float, float, float, float] | list[float] | None = None,
    crs: int | str | CRS | None = None,
    cell_size: tuple[float, float] | float | None = None,
    index: int = 1,
) -> Grid:
    """Creates a new grid from a data reader.

    Args:
        data: Data reader.
        extent: Map extent used to clip the raster.
        crs: CRS or an EPSG code (e.g. 4326).
        index (int, optional): Band index.  Defaults to 1.

    Example:
        >>> with rasterio.open("n55_e008_1arc_v3.bil") as dataset:
        ...     grid(dataset)
        ...
        image: 1801x3601 float32 | range: -28.000~101.000 | mean: 11.566 | std: 14.645 | crs: EPSG:4326 | cell: ...

    Returns:
        Grid: A new grid.
    """
    ...


@overload
def grid(
    data: bytes,
    extent: tuple[float, float, float, float] | list[float] | None = None,
    crs: int | str | CRS | None = None,
    cell_size: tuple[float, float] | float | None = None,
    index: int = 1,
) -> Grid:
    """Creates a new grid from bytes data.

    Args:
        data: Bytes.
        extent: Map extent used to clip the raster.
        crs: CRS or an EPSG code (e.g. 4326).
        index (int, optional): Band index.  Defaults to 1.

    Example:
        >>> grid(data)
        image: 1801x3601 float32 | range: -28.000~101.000 | mean: 11.566 | std: 14.645 | crs: EPSG:4326 | cell: ...

    Returns:
        Grid: A new grid.
    """
    ...


@overload
def grid(  # type: ignore
    data: MemoryFile,
    extent: tuple[float, float, float, float] | list[float] | None = None,
    crs: int | str | CRS | None = None,
    cell_size: tuple[float, float] | float | None = None,
    index: int = 1,
) -> Grid:
    """Creates a new grid from a memory file.

    Args:
        data: Memory file.
        extent: Map extent used to clip the raster.
        crs: CRS or an EPSG code (e.g. 4326).
        index (int, optional): Band index.  Defaults to 1.

    Example:
        >>> grid(memory_file)
        image: 1801x3601 float32 | range: -28.000~101.000 | mean: 11.566 | std: 14.645 | crs: EPSG:4326 | cell: ...

    Returns:
        Grid: A new grid.
    """
    ...


@overload
def grid(
    data: ndarray,
    extent: tuple[float, float, float, float] | list[float] | Affine | None = None,
    crs: int | str | CRS = 4326,
) -> Grid:
    """Creates a new grid from an array.

    Args:
        data: Array.
        extent: Map extent.  Defaults to (0.0, 0.0, 1.0, 1.0).
        crs: CRS or an EPSG code.  Defaults to 4326.

    Example:
        >>> grid(np.arange(12).reshape(3, 4))
        image: 4x3 int32 | range: 0~11 | mean: 6 | std: 3 | crs: EPSG:4326 | cell: 0.25, 0.3333333333333333

    Returns:
        Grid: A new grid.
    """
    ...


@overload
def grid(
    data: tuple[int, int],
    extent: tuple[float, float, float, float] | list[float] | None = None,
    crs: int | str | CRS = 4326,
) -> Grid:
    """Creates a new grid from a tuple (width, height).

    Args:
        data: Tuple (width, height).
        extent: Map extent.  Defaults to (0.0, 0.0, 1.0, 1.0).
        crs: CRS or an EPSG code.  Defaults to 4326.

    Example:
        >>> grid((40, 30))
        image: 40x30 int32 | range: 0~1199 | mean: 600 | std: 346 | crs: EPSG:4326 | cell: 0.025, 0.03333333333333333

    Returns:
        Grid: A new grid.
    """
    ...


@overload
def grid(
    data: int | float,
    extent: tuple[float, float, float, float] | list[float],
    crs: int | str | CRS,
    cell_size: tuple[float, float] | float,
) -> Grid:
    """Creates a new grid from a constant value.

    Args:
        data: Constant value.
        extent: Map extent.
        crs: CRS or an EPSG code.
        cell_size: Cell size.

    Example:
        >>> grid(123.456, (-120, 40, -100, 60), 4326, 1.0)
        image: 20x20 float32 | range: 123.456~123.456 | mean: 123.456 | std: 0.000 | crs: EPSG:4326 | cell: 1.0, 1.0

    Returns:
        Grid: A new grid.
    """
    ...


@overload
def grid(
    data: Iterable[tuple[BaseGeometry, float] | tuple[float, float, float]],
    extent: tuple[float, float, float, float] | list[float],
    crs: int | str | CRS,
    cell_size: tuple[float, float] | float,
) -> Grid:
    """Creates a new grid from an iterable of tuples (geometry, value) or (x, y, value).

    Args:
        data: Tuples (geometry, value) or (x, y, value).
        extent: Map extent.
        crs: CRS or an EPSG code.
        cell_size: Cell size.

    Example:
        >>> grid([(-119, 55, 1.23), (-104, 52, 4.56), (-112, 47, 7.89)], (-120, 40, -100, 60), 4326, 1.0)
        image: 20x20 float32 | range: 1.230~7.890 | mean: 4.560 | std: 2.719 | crs: EPSG:4326 | cell: 1.0, 1.0

    Returns:
        Grid: A new grid.
    """
    ...


def grid(
    data: str
    | DatasetReader
    | bytes
    | MemoryFile
    | ndarray
    | tuple[int, int]
    | int
    | float
    | Iterable[tuple[float, float, float] | tuple[BaseGeometry, float]],
    extent: tuple[float, float, float, float] | list[float] | Affine | None = None,
    crs: int | str | CRS | None = None,
    cell_size: tuple[float, float] | float | None = None,
    index: int = 1,
) -> Grid:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=NotGeoreferencedWarning)
        match data:
            case DatasetReader():
                return from_dataset(data, extent, crs, cell_size, index)  # type: ignore
            case str():
                with rasterio.open(data) as dataset:
                    return from_dataset(dataset, extent, crs, cell_size, index)  # type: ignore
            case bytes():
                with MemoryFile(data) as memory_file, memory_file.open() as dataset:
                    return from_dataset(dataset, extent, crs, cell_size, index)  # type: ignore
            case MemoryFile():
                with data.open() as dataset:
                    return from_dataset(dataset, extent, crs, cell_size, index)  # type: ignore
            case ndarray():
                return from_ndarray(data, extent or (0, 0, 1, 1), crs or 4326)
            case w, h if isinstance(w, int) and isinstance(h, int):
                data = np.arange(w * h).reshape(h, w)
                return from_ndarray(data, extent or (0, 0, 1, 1), crs or 4326)
            case _:
                assert extent, "Extent is required."
                assert cell_size, "Cell size is required."
                if isinstance(data, (int | float)):
                    return from_constant(data, extent, crs or 4326, cell_size)  # type: ignore
                return from_shapes(data, extent, crs or 4326, cell_size)  # type: ignore


def from_shapes(
    shapes: Iterable[tuple[float, float, float] | tuple[BaseGeometry, float]],
    extent: tuple[float, float, float, float] | list[float],
    crs: CRS,
    cell_size: tuple[float, float] | float,
):
    g = grid(np.nan, extent, crs, cell_size)
    return g.rasterize(shapes)


def from_constant(
    data: int | float,
    extent: tuple[float, float, float, float] | list[float],
    crs: CRS,
    cell_size: tuple[float, float] | float,
):
    cell_size = CellSize(cell_size, cell_size) if isinstance(cell_size, (int | float)) else CellSize(*cell_size)
    xmin, ymin, xmax, ymax = extent
    width = int((xmax - xmin) / cell_size.x + 0.5)
    height = int((ymax - ymin) / cell_size.y + 0.5)
    xmin = xmax - cell_size.x * width
    ymax = ymin + cell_size.y * height
    transform = Affine(cell_size.x, 0, xmin, 0, -cell_size.y, ymax, 0, 0, 1)
    return Grid(np.ones((height, width)) * data, transform, crs)


def from_ndarray(
    data: ndarray,
    extent: tuple[float, float, float, float] | list[float] | Affine,
    crs: CRS,
):
    if isinstance(extent, Affine):
        transform = extent
    else:
        transform = rasterio.transform.from_bounds(  # type: ignore
            *extent, data.shape[1], data.shape[0]
        )
    return Grid(format_type(data), transform, crs)


def from_dataset(
    dataset: DatasetReader,
    extent: tuple[float, float, float, float] | list[float] | None,
    crs: int | str | CRS | None,
    cell_size: tuple[float, float] | float | None,
    index: int,
) -> Grid:
    if extent:
        if crs is not None:
            xmin, ymin, xmax, ymax = extent
            crs = get_crs(crs)
            [xmin, xmax], [ymin, ymax], *_ = rasterio.warp.transform(crs, dataset.crs, [xmin, xmax], [ymin, ymax])
            extent = xmin, ymin, xmax, ymax
        extent = _adjust_extent(extent, dataset.transform)
        w = int(dataset.profile.data["width"])
        h = int(dataset.profile.data["height"])
        e1 = Extent(*extent)
        e2 = _extent(w, h, dataset.transform)
        e = e1.intersect(e2)
        left = (e.xmin - e2.xmin) / (e2.xmax - e2.xmin) * w
        right = (e.xmax - e2.xmin) / (e2.xmax - e2.xmin) * w
        top = (e2.ymax - e.ymax) / (e2.ymax - e2.ymin) * h
        bottom = (e2.ymax - e.ymin) / (e2.ymax - e2.ymin) * h
        width = right - left
        height = bottom - top
        if width > 0 and height > 0:
            window = Window(left, top, width, height)  # type: ignore
            data = dataset.read(index, window=window)
            g = grid(data, dataset.window_transform(window), dataset.crs)
        else:
            raise ValueError("The specified extent does not overlap the dataset.")
    else:
        data = dataset.read(index)
        g = grid(data, dataset.transform, dataset.crs)

    if dataset.nodata is not None and g.dtype != "uint8":
        g = g.set_nan(dataset.nodata)
    if crs is not None and g.crs != get_crs(crs):
        g = g.project(crs)
    if cell_size is not None and g.cell_size != cell_size:
        g = g.resample(cell_size)

    return g


def _adjust_extent(extent: tuple[float, float, float, float] | list[float], transform: Affine):
    xmin, ymin, xmax, ymax = extent
    window = from_bounds(xmin, ymin, xmax, ymax, transform)
    window = window.round_offsets().round_lengths()
    transform2: Any = transform * Affine.translation(window.col_off, window.row_off)
    xmin2 = transform2.c
    ymax2 = transform2.f
    xmax2 = xmin2 + window.width * transform2.a
    ymin2 = ymax2 + window.height * transform2.e
    return xmin2, ymin2, xmax2, ymax2


def _extent(width, height, transform) -> Extent:
    xmin, ymax, *_ = transform * (0, 0)
    xmax, ymin, *_ = transform * (width, height)
    return Extent(xmin, ymin, xmax, ymax)


def _metadata(grid: Grid):
    return {
        "height": grid.height,
        "width": grid.width,
        "crs": grid.crs,
        "transform": grid.transform,
    }


def _to_uint8_range(grid: Grid):
    if grid.dtype == "bool" or grid.min >= 0 and grid.max <= 255:
        return grid.type("float32")
    return grid.percent_clip(0.1, 99.9).stretch(0, 255)


def con(grid: Grid, true_value: Operand, false_value: Operand):
    """Evaluates a boolean grid .

    Args:
        grid: Boolean grid.
        true_value: Values applied for cells evaluating to true.
        false_value: Values applied for cells evaluating to false.

    Returns:
        Grid: A new grid.
    """
    return grid.local(lambda data: np.where(data, grid._data(true_value), grid._data(false_value)))


def standardize(
    *grids: Grid,
    extent: Extent | ExtentResolution = "intersect",
    cell_size: tuple[float, float] | float | None = None,
) -> tuple[Grid, ...]:
    """Standardizes input grids to have the same map extent and cell size.

    Args:
        extent: Rule for extent.  Defaults to "intersect".
        cell_size: Cell size.  Defaults to None.

    Raises:
        ValueError: If grids are in different coordinate systems.

    Returns:
        tuple[Grid, ...]: Standardized grids.
    """
    if len(grids) == 1:
        return tuple(grids)

    crs_set = {grid.crs for grid in grids}

    if len(crs_set) > 1:
        raise ValueError("Input grids must have the same CRS.")

    if isinstance(cell_size, (int | float)):
        cell_size_standardized = (cell_size, cell_size)
    elif cell_size is None:
        cell_size_standardized = grids[0].cell_size
    else:
        cell_size_standardized = cell_size

    if isinstance(extent, Extent):
        extent_standardized = extent
    else:
        extent_standardized = grids[0].extent
        for g in grids:
            if extent == "intersect":
                extent_standardized = extent_standardized & g.extent
            elif extent == "union":
                extent_standardized = extent_standardized | g.extent

    results = []

    for g in grids:
        if g.cell_size != cell_size_standardized:
            g = g.resample(cell_size_standardized)
        if g.extent != extent_standardized:
            g = g.clip(extent_standardized)
        results.append(g)

    return tuple(results)


def _aggregate(func: Callable, *grids: Grid) -> Grid:
    grids_adjusted = standardize(*grids)
    data = func(np.array([grid.data for grid in grids_adjusted]), axis=0)
    return grids_adjusted[0].local(data)


def mean(*grids: Grid) -> Grid:
    """Calculates the mean value across input grids."""
    return _aggregate(np.mean, *grids)


def std(*grids: Grid) -> Grid:
    """Calculates the standard deviation across input grids."""
    return _aggregate(np.std, *grids)


def minimum(*grids: Grid) -> Grid:
    """Calculates the minimum value across input grids."""
    return _aggregate(np.min, *grids)


def maximum(*grids: Grid) -> Grid:
    """Calculates the maximum value across input grids."""
    return _aggregate(np.max, *grids)


def idw(
    points: Sequence[tuple[float, float, float]],
    extent: tuple[float, float, float, float] | list[float],
    crs: int | str | CRS,
    cell_size: tuple[float, float] | float,
    radius: float | None = None,
    max_workers: int = 1,
):
    """Calculates the inverse distance weighting (IDW) interpolation for a set of points."""
    g = grid(np.nan, extent, crs, cell_size)

    if len(points) == 0:
        return g

    def f(g: Grid) -> Grid:
        if radius:
            e = Extent(
                g.extent.xmin - radius,
                g.extent.ymin - radius,
                g.extent.xmax + radius,
                g.extent.ymax + radius,
            )
            points_adjusted = [p for p in points if e.intersects((p[0], p[1], p[0], p[1]))]
            if not points_adjusted:
                return g
        else:
            points_adjusted = points

        x, y, v = zip(*points_adjusted, strict=False)
        values = np.array(v)
        coords = g._coords()
        xi, yi = coords[:, 0], coords[:, 1]
        distances = np.asarray((xi[:, None] - x) ** 2 + (yi[:, None] - y) ** 2, "float32")
        distances[distances == 0] = 1e-10
        weights = 1 / distances

        if radius:
            mask = distances > radius**2
            distances[mask] = np.inf
            weights[mask] = 0

        result = np.dot(weights / weights.sum(axis=1, keepdims=True), values)
        return g.local(result.reshape(g.height, g.width))

    return g.process_tiles(f, 256, 0, max_workers)


def pca(n_components: int = 1, *grids: Grid) -> tuple[Grid, ...]:
    """Performs principal component analysis (PCA) on input grids."""
    grids_adjusted = [con(g.is_nan(), float(g.mean), g) for g in standardize(*grids)]
    arrays = (
        PCA(n_components=n_components)
        .fit_transform(np.array([g.scale(StandardScaler()).data.ravel() for g in grids_adjusted]).transpose((1, 0)))
        .transpose((1, 0))
    )
    g = grids_adjusted[0]
    return tuple(g.local(a.reshape((g.height, g.width))) for a in arrays)
