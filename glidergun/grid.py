import dataclasses
import hashlib
import logging
import warnings
from base64 import b64encode
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import cached_property
from types import FunctionType
from typing import TYPE_CHECKING, Any, Literal, Union, cast, overload

import numpy as np
import rasterio
from cv2 import Canny
from numpy import arctan, arctan2, cos, gradient, ndarray, pi, sin, sqrt
from rasterio import DatasetReader, features
from rasterio.crs import CRS
from rasterio.errors import NotGeoreferencedWarning
from rasterio.io import MemoryFile
from rasterio.transform import Affine
from rasterio.warp import Resampling, calculate_default_transform, reproject
from rasterio.windows import Window, from_bounds
from scipy.ndimage import gaussian_filter, sobel
from shapely.geometry import Point, Polygon
from shapely.geometry.base import BaseGeometry

from glidergun.focal import Focal
from glidergun.geojson import FeatureCollection
from glidergun.interp import Interpolation
from glidergun.io import write_to_file
from glidergun.literals import Basemap, BasemapUrl, ColorMap, DataType, ExtentResolution, ResamplingMethod
from glidergun.plot import create_histogram, create_thumbnail
from glidergun.types import BBox, CellSize, Chart, Extent, GridCore, PointValue, Scaler
from glidergun.utils import (
    add_to_map,
    format_type,
    get_crs,
    get_crs_name,
    get_nodata_value,
    is_image_format,
    is_tile_url,
)
from glidergun.zonal import Zonal

if TYPE_CHECKING:
    from glidergun.stac import Stack

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
            + f"crs: {self.crs} {get_crs_name(self.crs)} | "
            + f"cell: {self.cell_size} | "
            + f"extent: {self.extent}"
        )

    @cached_property
    def img(self) -> str:
        resample_factor = max(self.width / 1000, self.height / 1000, 1)
        g = self.resample_by(resample_factor) if resample_factor > 1 else self
        obj = _to_uint8_range(g)
        bytes_data = create_thumbnail(obj.data, self.display)
        image = b64encode(bytes_data).decode()
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
        return _extent(self.width, self.height, self.transform, self.crs)

    @cached_property
    def extent_4326(self) -> Extent:
        return self.extent.project(4326)

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
        return CellSize(abs(self.transform.a), abs(self.transform.e))

    @cached_property
    def bins(self) -> dict[float, int]:
        unique, counts = np.unique(self.data, return_counts=True)
        return dict(sorted(zip(map(float, unique), map(int, counts), strict=True)))

    @cached_property
    def sha256(self) -> str:
        return hashlib.sha256(self.data.tobytes(order="C")).hexdigest()

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
        return self._apply(self, n, np.bitwise_and, convert_to_float=False)

    __rand__ = __and__

    def __or__(self, n: Operand):
        return self._apply(self, n, np.bitwise_or, convert_to_float=False)

    __ror__ = __or__

    def __xor__(self, n: Operand):
        return self._apply(self, n, np.bitwise_xor, convert_to_float=False)

    __rxor__ = __xor__

    def __rshift__(self, n: Operand):
        return self._apply(self, n, np.right_shift, convert_to_float=False)

    def __lshift__(self, n: Operand):
        return self._apply(self, n, np.left_shift, convert_to_float=False)

    __rrshift__ = __lshift__

    __rlshift__ = __rshift__

    def __neg__(self):
        return self.local(-1.0 * self.data)

    def __pos__(self):
        return self.local(1.0 * self.data)

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

    def _data(self, n: Operand, convert_to_float: bool):
        value = n.data if isinstance(n, Grid) else n
        return value * 1.0 if convert_to_float else value

    def _apply(self, left: Operand, right: Operand, op: Callable, convert_to_float: bool = True):
        if (
            isinstance(left, Grid)
            and isinstance(right, Grid)
            and (left.cell_size != right.cell_size or left.extent != right.extent)
        ):
            l_adjusted, r_adjusted = standardize(left, right)
            return l_adjusted.local(
                op(self._data(l_adjusted, convert_to_float), self._data(r_adjusted, convert_to_float))
            )

        return self.local(op(self._data(left, convert_to_float), self._data(right, convert_to_float)))

    def local(self, func: Callable[[ndarray], ndarray] | ndarray):
        """Apply a numpy function to the data and return a new Grid."""
        data = func if isinstance(func, ndarray) else func(self.data)
        return from_ndarray(data, self.transform, crs=self.crs)

    def mosaic(self, *grids: "Grid", blend: bool = False):
        """Create a mosaic of multiple grids blending overlaps."""
        grids_adjusted = standardize(self, *grids, extent="union")
        result = grids_adjusted[0]
        for g in grids_adjusted[1:]:
            if not blend:
                result = result.coalesce(g)
                continue
            i = ~result.is_nan() & ~g.is_nan()
            if not i.max:
                result = result.coalesce(g)
                continue
            g = result.calibrate(g)
            dx, dy = result.set_nan(i).distance(), g.set_nan(i).distance()
            d = dx + dy
            result = con(i, result * (dy / d) + g * (dx / d), result.coalesce(g))
        return result

    def calibrate(self, other: "Grid"):
        """Calibrate another grid to this grid's mean and standard deviation."""
        g1, g2 = standardize(self, other)
        xor = g1.is_nan() | g2.is_nan()
        g1, g2 = g1.set_nan(xor), g2.set_nan(xor)
        return other * (g1.std / g2.std) + g1.mean - g2.mean * (g1.std / g2.std)

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

    def georeference(self, extent: BBox, crs: int | str | CRS | None = None):
        """Assign extent and CRS to the current data without resampling."""
        return from_ndarray(self.data, extent, crs=get_crs(crs) if crs else self.crs)

    def georeference_to(self, reference: "Grid | Stack"):
        """Automatically georeference to a reference grid or stack using scan + LoFTR."""
        try:
            from glidergun.loftr import georeference_to_reference
        except ImportError as ex:
            raise ImportError(
                "This method requires the torch option.  Please install it with `pip install glidergun[torch]`."
            ) from ex

        return georeference_to_reference(self, reference)

    def _reproject(self, transform, crs, width, height, resampling: Resampling | ResamplingMethod) -> "Grid":
        is_bool = self.dtype == "bool"
        source = self.data.astype("uint8") if is_bool else self.data
        destination = np.empty((round(height), round(width)), dtype="float32")
        destination.fill(np.nan)
        method = Resampling[resampling] if isinstance(resampling, str) else resampling
        reproject(
            source=source,
            destination=destination,
            src_transform=self.transform,
            src_crs=self.crs,
            src_nodata=None if is_bool else self.nodata,
            dst_transform=transform,
            dst_crs=crs,
            dst_nodata=np.nan,
            resampling=method,
        )
        result = from_ndarray(destination, transform, crs=crs)
        if is_bool:
            return result == 1
        return result

    def project(self, crs: int | str | CRS, resampling: Resampling | ResamplingMethod = "nearest") -> "Grid":
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
        self, extent: BBox, cell_size: tuple[float, float], resampling: Resampling | ResamplingMethod
    ) -> "Grid":
        xmin, ymin, xmax, ymax = extent
        xoff = (xmin - self.xmin) / self.transform.a
        yoff = (ymax - self.ymax) / self.transform.e
        scaling_x = cell_size[0] / self.cell_size.x
        scaling_y = cell_size[1] / self.cell_size.y
        transform = self.transform * Affine.translation(xoff, yoff) * Affine.scale(scaling_x, scaling_y)
        width = (xmax - xmin) / abs(self.transform.a) / scaling_x
        height = (ymax - ymin) / abs(self.transform.e) / scaling_y
        return self._reproject(transform, self.crs, width, height, resampling)

    def clip(self, extent: BBox, preserve: bool = True):
        """Clip to the given extent (same CRS), preserving cell size."""
        if preserve:
            extent = _adjust_extent(extent, self.transform)
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

    def resample_by(self, factor: float, resampling: Resampling | ResamplingMethod = "nearest"):
        """Resample by a scaling factor, adjusting cell size accordingly."""
        return self.resample(self.cell_size * factor, resampling)

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

    def canny(self, threshold1: float = 100.0, threshold2: float = 200.0, **kwargs):
        """Detect edges using the Canny algorithm with given thresholds."""
        return self.local(lambda a: Canny(a, threshold1, threshold2, **kwargs))

    def gaussian_filter(self, sigma: float = 1.0, **kwargs):
        """Apply Gaussian filter to the grid."""
        return self.local(lambda a: gaussian_filter(a, sigma=sigma, **kwargs))

    def sobel(self, axis: int = -1, **kwargs):
        """Apply Sobel filter to the grid."""
        return self.local(lambda a: sobel(a, axis=axis, **kwargs))

    def distance(self):
        """Euclidean distance from cells with positive values."""
        from scipy.ndimage import distance_transform_edt

        g = self.coalesce(0.0)
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
        return con(g, replacement, self if fallback is None else fallback)

    def set_nan(
        self,
        predicate: Operand | Callable[["Grid"], "Grid"],
        fallback: Operand | None = None,
    ):
        """Set cells matching predicate to NaN, optionally using fallback."""
        return self.con(predicate, np.nan, fallback)

    def coalesce(self, value: Operand):
        """Replace NaN cells with the given value."""
        return self.con(self.is_nan(), value, self)

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

        if not result:
            raise ValueError("Resulting mosaic is empty or invalid.")
        return result

    @overload
    def overlay(self, other: "Grid", alpha: float = 1.0) -> "Grid": ...

    @overload
    def overlay(self, other: "Stack", alpha: float = 1.0) -> "Stack": ...

    def overlay(self, other: "Grid | Stack", alpha: float = 1.0):
        """Overlays another grid or stack on top of the current grid with alpha blending."""
        if isinstance(other, Grid):
            g1, g2 = standardize(self, other, extent="first")
            return con(g2.is_nan(), g1, g1 * (1 - alpha) + g2 * alpha)

        return self.to_stack().overlay(other, alpha)

    def draw(self, shape: BaseGeometry, value: float | None):
        """Draws a shape on the grid with the given value."""
        if value is None or value is np.nan:
            nodata_value: float = get_nodata_value("float32")  # type: ignore
            return self.draw(shape, nodata_value).set_nan(nodata_value)
        return self.overlay(grid([(shape, value)], self.extent, cell_size=self.cell_size))

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
        return _extent(data.shape[1], data.shape[0], transform, self.crs)

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

    def to_polygons(self, include_nan: bool = False) -> list[tuple[Polygon, float]]:
        """Polygonize regions of constant values; optionally include NaNs and smooth."""
        g = self * 1
        mask = None if include_nan else np.isfinite(g.data)
        results = []
        for shape, value in features.shapes(g.data, mask=mask, transform=g.transform):
            polygon = Polygon(shape["coordinates"][0], shape["coordinates"][1:])
            results.append((polygon, float(value)))
        return results

    @overload
    def to_geojson(self, polygonize: Literal[True], include_nan: bool = False) -> FeatureCollection[Polygon]: ...

    @overload
    def to_geojson(self, polygonize: Literal[False], include_nan: bool = False) -> FeatureCollection[Point]: ...

    def to_geojson(self, polygonize: bool, include_nan: bool = False):  # type: ignore
        """Export points or polygons as a GeoJSON FeatureCollection."""
        if polygonize:
            features = self.to_polygons(include_nan=include_nan)
        else:
            features = [(Point(x, y), value) for x, y, value in self.to_points(include_nan=include_nan)]
        return FeatureCollection((shape, {"value": None if np.isnan(value) else value}) for shape, value in features)

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
        import matplotlib.pyplot as plt

        from glidergun.stack import stack

        grid1 = self.percent_clip(0.1, 99.9)
        grid2 = grid1 - grid1.min
        grid3 = grid2 / grid2.max
        arrays = plt.get_cmap(self.display)(grid3.data).transpose(2, 0, 1)[:3]
        mask = self.is_nan()
        r, g, b = [self.local(a * 253 + 1).set_nan(mask) for a in arrays]
        return stack([r, g, b])

    def percentile(self, percent: float) -> float:
        """Return the `percent` percentile (NaN-aware)."""
        return np.nanpercentile(self.data, percent)

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
        from sklearn.cluster import KMeans

        g = self.coalesce(nodata)
        kmeans = KMeans(n_clusters=n_clusters, **kwargs).fit(g.data.reshape(-1, 1))
        data = kmeans.cluster_centers_[kmeans.labels_].reshape(self.data.shape)
        result = self.local(data)
        return result.set_nan(self.is_nan())

    def scale(self, scaler: Scaler | None = None, **fit_params):
        """Apply a scaler's `fit_transform` across the grid."""
        from sklearn.preprocessing import StandardScaler

        if scaler is None:
            scaler = StandardScaler()
        return self.local(lambda a: scaler.fit_transform(a, **fit_params))

    def hist(self, **kwargs) -> Chart:
        """Build a histogram chart of value counts (NaN-aware)."""
        return create_histogram((self.bins, "gray"))

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

    def sam(
        self,
        *prompt: str,
        checkpoint_path: str | None = None,
        hf_token: str | None = None,
        confidence_threshold: float = 0.5,
        tile_size: int = 1024,
    ):
        """Run Segment Anything Model 3 (SAM 3) over the stack with text prompts.

        Args:
            prompt: One or more text prompts used for segmentation.
            checkpoint_path: Optional path to a pre-trained SAM 3 checkpoint.
            hf_token: Optional Hugging Face token.
            confidence_threshold: Minimum confidence to accept a predicted mask.

        Returns:
            SamResult: Collection of masks and an overview visualization stack.
        """
        try:
            from glidergun.sam import sam
        except ImportError as ex:
            raise ImportError(
                "This method requires the torch option.  Please install it with `pip install glidergun[torch]`."
            ) from ex

        return sam(
            self.to_stack(),
            *prompt,
            checkpoint_path=checkpoint_path,
            hf_token=hf_token,
            confidence_threshold=confidence_threshold,
            tile_size=tile_size,
        )

    @overload
    def save(self, file: str, dtype: DataType | None = None) -> None:
        """Saves the grid to a file.

        Most commonly, `.tif` extension is used to save as a GeoTIFF (COG).
        If the file extension is `.jpg`, `.kml`, `.kmz`, `.mbtiles`, `.png`
        or `.webp`, the grid is converted to an RGB stack using the current
        colormap and saved as an image.  Otherwise, the grid is saved as a
        single-band raster.

        Args:
            file (str): File path (e.g. "output.tif").
            dtype (DataType | None): Data type for saving (optional).
        """
        ...

    @overload
    def save(self, file: MemoryFile, dtype: DataType | None, driver: str) -> None:
        """Saves the grid to a memory file.

        Args:
            file (MemoryFile): Memory file to save the grid to.
            dtype (DataType | None): Data type for saving (optional).
            driver (str): Rasterio driver name (optional).
        """
        ...

    def save(self, file: str | MemoryFile, dtype: DataType | None = None, driver: str = ""):
        g = self * 1 if self.dtype == "bool" else self

        if is_image_format(driver, file):
            g.to_stack().save(file)
            return

        if dtype is None:
            dtype = g.dtype

        nodata = get_nodata_value(dtype)

        if nodata is not None:
            g = g.coalesce(nodata)

        write_to_file(file, driver, dtype, nodata, g)

    def add_to_map(self, name: str | None = None):
        """Add as a layer to the active map in ArcGIS or QGIS."""
        add_to_map(self, name)


@overload
def grid(
    data: str,
    extent: BBox | Literal["#"] | None = None,
    *,
    resample_by: float | None = None,
    resampling: Resampling | ResamplingMethod | None = None,
    index: int = 1,
) -> Grid:
    """Creates a new grid from a file path or url.

    Args:
        data: File path.
        extent: Map extent used to clip the raster.

    Example:
        >>> grid("n55_e008_1arc_v3.bil")
        image: 1801x3601 float32 | range: -28.000~101.000 | mean: 11.566 | std: 14.645 | crs: EPSG:4326 | cell: ...

    Returns:
        Grid: A new grid.
    """
    ...


@overload
def grid(  # type: ignore
    data: Literal["alos-dem", "cop-dem-glo-30", "cop-dem-glo-90", "nasadem"],
    extent: BBox | Literal["#"],
    *,
    resample_by: float | None = None,
    resampling: Resampling | ResamplingMethod | None = None,
) -> Grid:
    """Creates a new grid from the specified dataset based on the given extent.

    Args:
        data: "alos-dem", "cop-dem-glo-30", "cop-dem-glo-90", or "nasadem".
        extent: Map extent used to clip the raster.

    Example:
        >>> grid("cop-dem-glo-30", (138.00, 34.20, 140.90, 36.80))
        image: 10440x9360 float32 | range: -25.906~3768.410 | mean: 335.900 | std: 516.297 | crs: EPSG:4326 | cell: ...

    Returns:
        Grid: A new grid.
    """
    ...


@overload
def grid(  # type: ignore
    data: DatasetReader,
    extent: BBox | None = None,
    *,
    resample_by: float | None = None,
    resampling: Resampling | ResamplingMethod | None = None,
    index: int = 1,
) -> Grid:
    """Creates a new grid from a data reader.

    Args:
        data: Data reader.
        extent: Map extent used to clip the raster.

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
    extent: BBox | None = None,
    *,
    index: int = 1,
) -> Grid:
    """Creates a new grid from bytes data.

    Args:
        data: Bytes.
        extent: Map extent used to clip the raster.

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
    extent: BBox | None = None,
    *,
    index: int = 1,
) -> Grid:
    """Creates a new grid from a memory file.

    Args:
        data: Memory file.
        extent: Map extent used to clip the raster.

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
    extent: BBox | Affine | None = None,
) -> Grid:
    """Creates a new grid from an array.

    Args:
        data: Array.
        extent: Map extent.

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
    extent: BBox | None = None,
) -> Grid:
    """Creates a new grid from a tuple (width, height).

    Args:
        data: Tuple (width, height).
        extent: Map extent.

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
    extent: Extent,
    *,
    cell_size: tuple[float, float] | float,
) -> Grid:
    """Creates a new grid from a constant value.

    Args:
        data: Constant value.
        extent: Map extent.
        cell_size: Cell size.

    Example:
        >>> grid(123.456, Extent(-120, 40, -100, 60, 4326), cell_size=1.0)
        image: 20x20 float32 | range: 123.456~123.456 | mean: 123.456 | std: 0.000 | crs: EPSG:4326 | cell: 1.0, 1.0

    Returns:
        Grid: A new grid.
    """
    ...


@overload
def grid(
    data: Iterable[tuple[BaseGeometry, float] | tuple[float, float, float]],
    extent: Extent,
    *,
    cell_size: tuple[float, float] | float,
) -> Grid:
    """Creates a new grid from an iterable of tuples (geometry, value) or (x, y, value).

    Args:
        data: Tuples (geometry, value) or (x, y, value).
        extent: Map extent.
        cell_size: Cell size.

    Example:
        >>> grid([(-119, 55, 1.23), (-104, 52, 4.56), (-112, 47, 7.89)], Extent(-120, 40, -100, 60, 4326), cell_size=1)
        image: 20x20 float32 | range: 1.230~7.890 | mean: 4.560 | std: 2.719 | crs: EPSG:4326 | cell: 1.0, 1.0

    Returns:
        Grid: A new grid.
    """
    ...


@overload
def grid(
    data: Basemap | str,
    extent: BBox | Literal["#"] = (-179.999, -85.0511, 179.999, 85.0511),
    *,
    max_tiles: int = 10,
    cache_dir: str | None = "~/.glidergun/cache",
    preserve_alpha: bool = False,
) -> Grid:
    """Creates a new grid from a tile service url template with {x}{y}{z} or {q} placeholders.

    Args:
        data (str): Tile service url template with {x}{y}{z} or {q} placeholders.
        extent (BBox): Map extent used to clip the raster.
        max_tiles (int, optional): Maximum number of tiles to load.  Defaults to 10.
        cache_dir (str | None, optional): Directory to cache downloaded tiles.  Defaults to "~/.glidergun/cache".
        preserve_alpha (bool, optional): Whether to preserve the alpha channel.  Defaults to False.

    Example:
        >>> grid("microsoft", (-123.164, 49.272, -123.162, 49.273), max_tiles=10)
        image: 373x285 uint8 | range: 13~243 | mean: 94 | std: 45 | crs: EPSG:3857

    Returns:
        Grid: A new grid in Web Mercator.
    """
    ...


def grid(  # type: ignore
    data: str
    | DatasetReader
    | bytes
    | MemoryFile
    | ndarray
    | tuple[int, int]
    | int
    | float
    | Iterable[tuple[BaseGeometry, float] | tuple[float, float, float]],
    extent: BBox | Literal["#"] | Affine | None = None,
    *,
    cell_size: tuple[float, float] | float | None = None,
    resample_by: float | None = None,
    resampling: Resampling | ResamplingMethod | None = None,
    max_tiles: int = 10,
    cache_dir: str | None = "~/.glidergun/cache",
    preserve_alpha: bool = False,
    index: int = 1,
) -> Grid:
    if extent == "#":
        try:
            extent = Extent.from_active_map()
        except RuntimeError as ex:
            raise ValueError(
                "The symbol '#' is only suppoted within ArcGIS Pro or QGIS.  Please provide an explicit extent."
            ) from ex

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=NotGeoreferencedWarning)

        if isinstance(data, ndarray):
            w, h = data.shape[1], data.shape[0]
            crs = extent.crs if isinstance(extent, Extent) else 4326
            return from_ndarray(data, extent or (0, 0, w / 1000, h / 1000), crs)
        if isinstance(extent, Affine):
            raise ValueError("When extent is an Affine transform, data must be an ndarray.")

        match data:
            case DatasetReader():
                return from_dataset(data, extent, resample_by=resample_by, resampling=resampling, index=index)
            case str():
                if data == "alos-dem" or data == "cop-dem-glo-30" or data == "cop-dem-glo-90" or data == "nasadem":
                    if not extent:
                        raise ValueError("Extent is required for dem collections.")
                    extent = extent.project(4326) if isinstance(extent, Extent) else Extent(*extent)
                    if extent.width * extent.height > 1000:
                        raise ValueError("Extent is too large for dem collections.")
                    return from_dem(
                        data, extent, limit=1000, resample_by=resample_by, resampling=resampling or "average"
                    )
                if data in BasemapUrl or is_tile_url(data):
                    from glidergun.stack import stack

                    s = stack(data, extent, max_tiles=max_tiles, cache_dir=cache_dir, preserve_alpha=preserve_alpha)
                    g = s.mean()
                    return g if preserve_alpha else g.type("uint8")
                return from_path(data, extent, resample_by=resample_by, resampling=resampling, index=index)
            case bytes():
                with MemoryFile(data) as memory_file, memory_file.open() as dataset:
                    return from_dataset(dataset, extent, resample_by=resample_by, resampling=resampling, index=index)
            case MemoryFile():
                with data.open() as dataset:
                    return from_dataset(dataset, extent, resample_by=resample_by, resampling=resampling, index=index)
            case w, h if isinstance(w, int) and isinstance(h, int):
                data = np.arange(w * h).reshape(h, w)
                if not extent:
                    extent = Extent(0, 0, w / 1000, h / 1000)
                if not isinstance(extent, Extent):
                    extent = Extent(*extent)
                return from_ndarray(data, extent, extent.crs)
            case _:
                if not extent:
                    raise ValueError("Extent is required.")
                if not cell_size:
                    raise ValueError("Cell size is required.")
                if isinstance(data, (int | float)):
                    return from_constant(data, extent, cell_size)  # type: ignore
                return from_shapes(data, extent, cell_size)  # type: ignore


def from_path(
    path: str,
    extent: BBox | None = None,
    *,
    resample_by: float | None = None,
    resampling: Resampling | ResamplingMethod | None = None,
    index: int = 1,
):
    with rasterio.open(path) as dataset:
        return from_dataset(dataset, extent, resample_by=resample_by, resampling=resampling, index=index)


def from_dataset(
    dataset: DatasetReader,
    extent: BBox | None,
    *,
    resample_by: float | None = None,
    resampling: Resampling | ResamplingMethod | None = None,
    index: int = 1,
) -> Grid:
    dataset_crs = dataset.crs or CRS.from_epsg(3857)

    if not resampling:
        resampling = "nearest"

    if isinstance(resampling, str):
        resampling = Resampling[resampling]

    if extent:
        if isinstance(extent, Extent):
            extent = extent.project(dataset_crs)
        extent = _adjust_extent(extent, dataset.transform)
        w = int(dataset.profile.data["width"])
        h = int(dataset.profile.data["height"])
        e1 = Extent(*extent, dataset.crs)
        e2 = _extent(w, h, dataset.transform, dataset.crs)
        e = e1.intersect(e2)
        left = (e.xmin - e2.xmin) / e2.width * w
        right = (e.xmax - e2.xmin) / e2.width * w
        top = (e2.ymax - e.ymax) / e2.height * h
        bottom = (e2.ymax - e.ymin) / e2.height * h
        width = right - left
        height = bottom - top
        if width > 0 and height > 0:
            window = Window(left, top, width, height)  # type: ignore
            if resample_by is None:
                data = dataset.read(index, window=window)
                transform = dataset.window_transform(window)
            else:
                factor = 1 / resample_by
                new_width, new_height = int(width * factor), int(height * factor)
                data = dataset.read(index, window=window, out_shape=(new_height, new_width), resampling=resampling)
                t = dataset.window_transform(window)
                transform = t * t.scale(width / new_width, height / new_height)  # type: ignore
            g = from_ndarray(data, transform, crs=dataset_crs)
        else:
            raise ValueError("The specified extent does not overlap the dataset.")
    else:
        if resample_by is None:
            data = dataset.read(index)
            transform = dataset.transform
        else:
            factor = 1 / resample_by
            new_width, new_height = int(dataset.width * factor), int(dataset.height * factor)
            data = dataset.read(index, out_shape=(new_height, new_width), resampling=resampling)
            t = dataset.transform
            transform = t * t.scale(dataset.width / new_width, dataset.height / new_height)  # type: ignore
        g = from_ndarray(data, transform, crs=dataset_crs)

    if dataset.nodata is not None and g.dtype != "uint8":
        g = g.set_nan(dataset.nodata)

    return g


def from_dem(
    collection: Literal["alos-dem", "cop-dem-glo-30", "cop-dem-glo-90", "nasadem"],
    extent: BBox,
    limit: int,
    resample_by: float | None = None,
    resampling: Resampling | ResamplingMethod = "average",
):
    from glidergun.stac import planetary_computer_url, search_mosaic

    collections = {
        "alos-dem": "data",
        "cop-dem-glo-30": "data",
        "cop-dem-glo-90": "data",
        "nasadem": "elevation",
    }

    return search_mosaic(
        planetary_computer_url,
        collection,
        collections[collection],
        extent,
        limit,
        resample_by=resample_by,
        resampling=resampling,
    )


def from_ndarray(data: ndarray, extent: BBox | Affine, crs: int | str | CRS):
    if isinstance(extent, Affine):
        transform = extent
    else:
        transform = rasterio.transform.from_bounds(*extent, data.shape[1], data.shape[0])  # type: ignore
    return Grid(format_type(data), transform, get_crs(crs))


def from_constant(data: int | float, extent: Extent, cell_size: tuple[float, float] | float):
    cell_size = CellSize(cell_size, cell_size) if isinstance(cell_size, (int | float)) else CellSize(*cell_size)
    xmin, ymin, xmax, ymax = extent
    width = int((xmax - xmin) / cell_size.x + 0.5)
    height = int((ymax - ymin) / cell_size.y + 0.5)
    xmin = xmax - cell_size.x * width
    ymax = ymin + cell_size.y * height
    transform = Affine(cell_size.x, 0, xmin, 0, -cell_size.y, ymax, 0, 0, 1)
    return from_ndarray(np.ones((height, width)) * data, transform, extent.crs)


def from_shapes(
    shapes: Iterable[tuple[float, float, float] | tuple[BaseGeometry, float]],
    extent: Extent,
    cell_size: tuple[float, float] | float,
):
    g = grid(np.nan, extent, cell_size=cell_size)
    return g.rasterize(shapes)


def _adjust_extent(extent: BBox, transform: Affine):
    xmin, ymin, xmax, ymax = extent
    window = from_bounds(xmin, ymin, xmax, ymax, transform)
    window = window.round_offsets().round_lengths()
    transform2: Any = transform * Affine.translation(window.col_off, window.row_off)
    xmin2 = transform2.c
    ymax2 = transform2.f
    xmax2 = xmin2 + window.width * transform2.a
    ymin2 = ymax2 + window.height * transform2.e
    return xmin2, ymin2, xmax2, ymax2


def _extent(width, height, transform, crs) -> Extent:
    x0, y0, *_ = transform * (0, 0)
    x1, y1, *_ = transform * (width, height)
    return Extent(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1), crs)


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
    if isinstance(true_value, Grid):
        grid, true_value = standardize(grid, true_value)
    if isinstance(false_value, Grid):
        grid, false_value = standardize(grid, false_value)

    return grid.local(
        lambda data: np.where(
            data, grid._data(true_value, convert_to_float=False), grid._data(false_value, convert_to_float=False)
        )
    )


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
        cell_size_x_min = min(grid.cell_size.x for grid in grids)
        cell_size_y_min = min(grid.cell_size.y for grid in grids)
        cell_size_standardized = (cell_size_x_min, cell_size_y_min)
    else:
        cell_size_standardized = cell_size

    if isinstance(extent, Extent):
        extent_standardized = extent
    else:
        extent_standardized = grids[0].extent
        for g in grids[1:]:
            if extent == "intersect":
                extent_standardized = extent_standardized & g.extent
            elif extent == "union":
                extent_standardized = extent_standardized | g.extent

    results = []

    for g in grids:
        if g.cell_size != cell_size_standardized:
            g = g.resample(cell_size_standardized)
        if g.extent != extent_standardized:
            g = g.clip(extent_standardized, preserve=False)
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
    extent: Extent,
    crs: int | str | CRS,
    cell_size: tuple[float, float] | float,
    radius: float | None = None,
    max_workers: int = 1,
):
    """Calculates the inverse distance weighting (IDW) interpolation for a set of points."""
    g = grid(np.nan, extent, cell_size=cell_size)

    if len(points) == 0:
        return g

    def f(g: Grid) -> Grid:
        if radius:
            e = Extent(
                g.extent.xmin - radius,
                g.extent.ymin - radius,
                g.extent.xmax + radius,
                g.extent.ymax + radius,
                g.crs,
            )
            points_adjusted = [p for p in points if e.intersects((p[0], p[1], p[0], p[1]))]
            if not points_adjusted:
                return g
        else:
            points_adjusted = points

        x, y, v = zip(*points_adjusted, strict=True)
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
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    arrays = [g.data for g in standardize(*grids)]
    h, w = arrays[0].shape
    x = np.stack([a.reshape(-1) for a in arrays], axis=1)
    valid = ~np.isnan(x).any(axis=1)
    x_valid = x[valid]
    x_valid = StandardScaler().fit_transform(x_valid)
    z = PCA(n_components=n_components).fit_transform(x_valid)
    outputs: list[ndarray] = []
    for i in range(n_components):
        out = np.full(x.shape[0], np.nan, dtype=z.dtype)
        out[valid] = z[:, i]
        outputs.append(out.reshape(h, w))
    return tuple(from_ndarray(a, grids[0].transform, crs=grids[0].crs) for a in outputs)
