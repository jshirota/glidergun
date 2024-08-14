import dataclasses
import hashlib
from dataclasses import dataclass
from functools import cached_property
from typing import (Any, Callable, Dict, Iterable, Optional, Tuple, Union,
                    cast, overload)

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import scipy as sp
from numpy import arctan, arctan2, cos, gradient, ndarray, pi, sin, sqrt
from rasterio import features
from rasterio.crs import CRS
from rasterio.drivers import driver_from_extension
from rasterio.io import MemoryFile
from rasterio.transform import Affine, from_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject
from rasterio.windows import Window
from shapely import Polygon
from sklearn.preprocessing import QuantileTransformer

from glidergun._focal import Focal
from glidergun._literals import (BaseMap, ColorMap, DataType, ExtentResolution,
                                 ResamplingMethod)
from glidergun._prediction import Prediction
from glidergun._types import CellSize, Extent, GridCore, Point, Scaler
from glidergun._utils import (create_parent_directory, format_type, get_crs,
                              get_nodata_value)
from glidergun._zonal import Zonal

Operand = Union["Grid", float, int]


@dataclass(frozen=True)
class Grid(GridCore, Prediction, Focal, Zonal):
    _cmap: Union[ColorMap, Any] = "gray"

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
            + f"cell: {self.cell_size.x}, {self.cell_size.y}"
        )

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
        return self.is_nan().data.any()

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
        return np.nanmean(self.data)

    @cached_property
    def std(self):
        return np.nanmean(self.data)

    @cached_property
    def min(self):
        return np.nanmin(self.data)

    @cached_property
    def max(self):
        return np.nanmax(self.data)

    @cached_property
    def cell_size(self) -> CellSize:
        return CellSize(self.transform.a, -self.transform.e)

    @cached_property
    def bins(self) -> Dict[float, int]:
        unique, counts = zip(np.unique(self.data, return_counts=True))
        return dict(sorted(zip(map(float, unique[0]), map(int, counts[0]))))

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
        if not isinstance(n, (Grid, float, int)):
            return NotImplemented
        return self._apply(self, n, np.equal)

    __req__ = __eq__

    def __ne__(self, n: object):
        if not isinstance(n, (Grid, float, int)):
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
        return self.update(-1 * self.data)

    def __pos__(self):
        return self.update(1 * self.data)

    def __invert__(self):
        return con(self, False, True)

    def update(self, data: ndarray):
        return grid(data, self.transform, self.crs)

    def _data(self, n: Operand):
        if isinstance(n, Grid):
            return n.data
        return n

    def _apply(self, left: Operand, right: Operand, op: Callable):
        if not isinstance(left, Grid) or not isinstance(right, Grid):
            return self.update(op(self._data(left), self._data(right)))

        if left.cell_size == right.cell_size and left.extent == right.extent:
            return self.update(op(left.data, right.data))

        l_adjusted, r_adjusted = standardize(left, right)

        return self.update(op(l_adjusted.data, r_adjusted.data))

    def local(self, func: Callable[[ndarray], Any]):
        return self.update(func(self.data))

    def con(self, trueValue: Operand, falseValue: Operand):
        return self.local(
            lambda data: np.where(data, self._data(
                trueValue), self._data(falseValue))
        )

    def mosaic(self, *grids: "Grid"):
        grids_adjusted = standardize(self, *grids, extent="union")
        result = grids_adjusted[0]
        for g in grids_adjusted[1:]:
            result = con(result.is_nan(), g, result)
        return result

    def standardize(self, *grids: "Grid"):
        return standardize(self, *grids)

    def is_nan(self):
        return self.local(np.isnan)

    def abs(self):
        return self.local(np.abs)

    def sin(self):
        return self.local(np.sin)

    def cos(self):
        return self.local(np.cos)

    def tan(self):
        return self.local(np.tan)

    def arcsin(self):
        return self.local(np.arcsin)

    def arccos(self):
        return self.local(np.arccos)

    def arctan(self):
        return self.local(np.arctan)

    def log(self, base: Optional[float] = None):
        if base is None:
            return self.local(np.log)
        return self.local(lambda a: np.log(a) / np.log(base))

    def round(self, decimals: int = 0):
        return self.local(lambda a: np.round(a, decimals))

    def gaussian_filter(self, sigma: float, **kwargs):
        return self.local(lambda a: sp.ndimage.gaussian_filter(a, sigma, **kwargs))

    def gaussian_filter1d(self, sigma: float, **kwargs):
        return self.local(lambda a: sp.ndimage.gaussian_filter1d(a, sigma, **kwargs))

    def gaussian_gradient_magnitude(self, sigma: float, **kwargs):
        return self.local(
            lambda a: sp.ndimage.gaussian_gradient_magnitude(
                a, sigma, **kwargs)
        )

    def gaussian_laplace(self, sigma: float, **kwargs):
        return self.local(lambda a: sp.ndimage.gaussian_laplace(a, sigma, **kwargs))

    def prewitt(self, **kwargs):
        return self.local(lambda a: sp.ndimage.prewitt(a, **kwargs))

    def sobel(self, **kwargs):
        return self.local(lambda a: sp.ndimage.sobel(a, **kwargs))

    def uniform_filter(self, **kwargs):
        return self.local(lambda a: sp.ndimage.uniform_filter(a, **kwargs))

    def uniform_filter1d(self, size: float, **kwargs):
        return self.local(lambda a: sp.ndimage.uniform_filter1d(a, size, **kwargs))

    def georeference(self, xmin: float, ymin: float, xmax: float, ymax: float, crs: Union[int, CRS] = 4326):
        return grid(self.data, (xmin, ymin, xmax, ymax), crs)

    def _reproject(
        self,
        transform,
        crs,
        width,
        height,
        resampling: Union[Resampling, ResamplingMethod],
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
            dst_nodata=self.nodata,
            resampling=(
                Resampling[resampling] if isinstance(
                    resampling, str) else resampling
            ),
        )
        result = grid(destination, transform, crs)
        if self.dtype == "bool":
            return result == 1
        return con(result == result.nodata, np.nan, result)

    def project(
        self,
        crs: Union[int, CRS],
        resampling: Union[Resampling, ResamplingMethod] = "nearest",
    ) -> "Grid":
        if get_crs(crs).wkt == self.crs.wkt:
            return self
        transform, width, height = calculate_default_transform(
            self.crs, crs, self.width, self.height, *self.extent
        )
        return self._reproject(
            transform,
            crs,
            width,
            height,
            Resampling[resampling] if isinstance(
                resampling, str) else resampling,
        )

    def _resample(
        self,
        extent: Tuple[float, float, float, float],
        cell_size: Tuple[float, float],
        resampling: Union[Resampling, ResamplingMethod],
    ) -> "Grid":
        (xmin, ymin, xmax, ymax) = extent
        xoff = (xmin - self.xmin) / self.transform.a
        yoff = (ymax - self.ymax) / self.transform.e
        scaling_x = cell_size[0] / self.cell_size.x
        scaling_y = cell_size[1] / self.cell_size.y
        transform = (
            self.transform
            * Affine.translation(xoff, yoff)
            * Affine.scale(scaling_x, scaling_y)
        )
        width = (xmax - xmin) / abs(self.transform.a) / scaling_x
        height = (ymax - ymin) / abs(self.transform.e) / scaling_y
        return self._reproject(transform, self.crs, width, height, resampling)

    def clip(self, xmin: float, ymin: float, xmax: float, ymax: float):
        return self._resample((xmin, ymin, xmax, ymax), self.cell_size, Resampling.nearest)

    def resample(
        self,
        cell_size: Union[Tuple[float, float], float],
        resampling: Union[Resampling, ResamplingMethod] = "nearest",
    ):
        if isinstance(cell_size, (int, float)):
            cell_size = (cell_size, cell_size)
        if self.cell_size == cell_size:
            return self
        return self._resample(self.extent, cell_size, resampling)

    def buffer(self, value: Union[float, int], count: int):
        if count < 0:
            g = (self != value).buffer(1, -count)
            return con(g == 0, value, self.set_nan(self == value))
        g = self
        for _ in range(count):
            g = con(g.focal_count(value, 1, True) > 0, value, g)
        return g

    def distance(self, *points: Tuple[float, float]):
        from glidergun._functions import distance
        if not points:
            points = tuple((p.x, p.y) for p in self.to_points() if p.value)
        return distance(self.extent, self.crs, self.cell_size, *points)

    def randomize(self):
        return self.update(np.random.rand(self.height, self.width))

    def aspect(self):
        x, y = gradient(self.data)
        return self.update(arctan2(-x, y))

    def slope(self):
        x, y = gradient(self.data)
        return self.update(pi / 2.0 - arctan(sqrt(x * x + y * y)))

    def hillshade(self, azimuth: float = 315, altitude: float = 45):
        azimuth = np.deg2rad(azimuth)
        altitude = np.deg2rad(altitude)
        aspect = self.aspect().data
        slope = self.slope().data
        shaded = sin(altitude) * sin(slope) + cos(altitude) * cos(slope) * cos(
            azimuth - aspect
        )
        return self.update((255 * (shaded + 1) / 2))

    def reclass(self, *mappings: Tuple[float, float, float]):
        conditions = [
            (self.data >= min) & (self.data < max) for min, max, _ in mappings
        ]
        values = [value for _, _, value in mappings]
        return self.update(np.select(conditions, values, np.nan))

    def percentile(self, percent: float) -> float:
        return np.nanpercentile(self.data, percent)  # type: ignore

    def slice(self, count: int, percent_clip: float = 0.1):
        min = self.percentile(percent_clip)
        max = self.percentile(100 - percent_clip)
        interval = (max - min) / count
        mappings = [
            (
                min + (i - 1) * interval if i > 1 else float("-inf"),
                min + i * interval if i < count else float("inf"),
                float(i),
            )
            for i in range(1, count + 1)
        ]
        return self.reclass(*mappings)

    def replace(
        self, value: Operand, replacement: Operand, fallback: Optional[Operand] = None
    ):
        return con(
            value if isinstance(value, Grid) else self == value,
            replacement,
            self if fallback is None else fallback,
        )

    def set_nan(self, value: Operand, fallback: Optional[Operand] = None):
        return self.replace(value, np.nan, fallback)

    def value(self, x: float, y: float) -> float:
        xoff = (x - self.xmin) / self.transform.a
        yoff = (y - self.ymax) / self.transform.e
        if xoff < 0 or xoff >= self.width or yoff < 0 or yoff >= self.height:
            return float(np.nan)
        return float(self.data[int(yoff), int(xoff)])

    @cached_property
    def data_extent(self) -> Extent:
        if not self.has_nan:
            return self.extent
        xmin, ymin, xmax, ymax = None, None, None, None
        for x, y, _ in self.to_points():
            if not xmin or x < xmin:
                xmin = x
            if not ymin or y < ymin:
                ymin = y
            if not xmax or x > xmax:
                xmax = x
            if not ymax or y > ymax:
                ymax = y
        if xmin is None or ymin is None or xmax is None or ymax is None:
            raise ValueError("None of the cells has a value.")
        return Extent(
            xmin - self.cell_size.x / 2,
            ymin - self.cell_size.y / 2,
            xmax + self.cell_size.x / 2,
            ymax + self.cell_size.y / 2,
        )

    def to_points(self) -> Iterable[Point]:
        for y, row in enumerate(self.data):
            for x, value in enumerate(row):
                if np.isfinite(value):
                    yield Point(
                        self.xmin + (x + 0.5) * self.cell_size.x,
                        self.ymax - (y + 0.5) * self.cell_size.y,
                        float(value),
                    )

    def to_polygons(self) -> Iterable[Tuple[Polygon, float]]:
        g = self * 1
        for shape, value in features.shapes(
            g.data, mask=np.isfinite(g.data), transform=g.transform
        ):
            if np.isfinite(value):
                coordinates = shape["coordinates"]
                yield Polygon(coordinates[0], coordinates[1:]), float(value)

    def from_polygons(
        self, polygons: Iterable[Tuple[Polygon, float]], all_touched: bool = False
    ):
        array = features.rasterize(
            shapes=polygons,
            out_shape=self.data.shape,
            fill=np.nan,  # type: ignore
            transform=self.transform,
            all_touched=all_touched,
            default_value=np.nan,  # type: ignore
        )
        return self.update(array)

    def to_stack(self, cmap: Union[ColorMap, Any]):
        from glidergun._stack import stack
        grid1 = self - self.min
        grid2 = grid1 / grid1.max
        arrays = plt.get_cmap(cmap)(grid2.data).transpose(2, 0, 1)[:3]
        mask = self.is_nan()
        r, g, b = [self.update(a * 253 + 1).set_nan(mask) for a in arrays]
        return stack(r, g, b)

    def scale(self, scaler: Optional[Scaler] = None, **fit_params):
        if not scaler:
            scaler = QuantileTransformer(n_quantiles=10)
        return self.local(lambda a: scaler.fit_transform(a, **fit_params))

    def stretch(self, min_value: float, max_value: float):
        expected_range = max_value - min_value
        actual_range = self.max - self.min

        if actual_range == 0:
            n = (min_value + max_value) / 2
            return self * 0 + n

        return (self - self.min) * expected_range / actual_range + min_value

    def cap_range(self, min: Operand, max: Operand, set_nan: bool = False):
        return self.cap_min(min, set_nan).cap_max(max, set_nan)

    def cap_min(self, value: Operand, set_nan: bool = False):
        return con(self < value, np.nan if set_nan else value, self)

    def cap_max(self, value: Operand, set_nan: bool = False):
        return con(self > value, np.nan if set_nan else value, self)

    def percent_clip(self, min_percent: float, max_percent: float):
        min_value = self.percentile(min_percent)
        max_value = self.percentile(max_percent)

        if min_value == max_value:
            return self

        return self.cap_range(min_value, max_value)

    def to_uint8_range(self):
        if self.dtype == "bool" or self.min > 0 and self.max < 255:
            return self
        return self.percent_clip(0.1, 99.9).stretch(1, 254)

    def hist(self, **kwargs):
        return plt.bar(list(self.bins.keys()), list(self.bins.values()), **kwargs)

    def plot(self, cmap: Union[ColorMap, Any]):
        return dataclasses.replace(self, _cmap=cmap)

    def map(
        self,
        cmap: Union[ColorMap, Any] = "gray",
        opacity: float = 1.0,
        basemap: Union[BaseMap, Any, None] = None,
        width: int = 800,
        height: int = 600,
        attribution: Optional[str] = None,
        grayscale: bool = True,
        **kwargs,
    ):
        from glidergun._ipython import _map
        return _map(
            self,
            cmap,
            opacity,
            basemap,
            width,
            height,
            attribution,
            grayscale,
            **kwargs,
        )

    def type(self, dtype: DataType):
        if self.dtype == dtype:
            return self
        return self.local(lambda data: np.asanyarray(data, dtype=dtype))

    @overload
    def save(
        self, file: str, dtype: Optional[DataType] = None, driver: str = ""
    ) -> None: ...

    @overload
    def save(  # type: ignore
        self, file: MemoryFile, dtype: Optional[DataType] = None, driver: str = ""
    ) -> None: ...

    def save(self, file, dtype: Optional[DataType] = None, driver: str = ""):
        g = self * 1 if self.dtype == "bool" else self

        if isinstance(file, str) and (
            file.lower().endswith(".jpg")
            or file.lower().endswith(".kml")
            or file.lower().endswith(".kmz")
            or file.lower().endswith(".png")
        ):
            g = g.to_uint8_range()
            dtype = "uint8"
        else:
            g = g
            if dtype is None:
                dtype = g.dtype

        nodata = get_nodata_value(dtype)

        if nodata is not None:
            g = con(g.is_nan(), nodata, g)

        if isinstance(file, str):
            create_parent_directory(file)
            with rasterio.open(
                file,
                "w",
                driver=driver if driver else driver_from_extension(file),
                count=1,
                dtype=dtype,
                nodata=nodata,
                **_metadata(self),
            ) as dataset:
                dataset.write(g.data, 1)
        elif isinstance(file, MemoryFile):
            with file.open(
                driver=driver if driver else "GTiff",
                count=1,
                dtype=dtype,
                nodata=nodata,
                **_metadata(self),
            ) as dataset:
                dataset.write(g.data, 1)

    def save_plot(self, file):
        self.to_stack(self._cmap).save(file)


@overload
def grid(
    data: str,
) -> Grid: ...


@overload
def grid(
    data: str,
    extent: Tuple[float, float, float, float],
) -> Grid: ...


@overload
def grid(
    data: str,
    extent: Tuple[float, float, float, float],
    crs: Union[int, CRS, None] = None,
    index: int = 1,
) -> Grid: ...


@overload
def grid(  # type: ignore
    data: MemoryFile,
) -> Grid: ...


@overload
def grid(  # type: ignore
    data: MemoryFile,
    extent: Tuple[float, float, float, float],
) -> Grid: ...


@overload
def grid(  # type: ignore
    data: MemoryFile,
    extent: Tuple[float, float, float, float],
    crs: Union[int, CRS, None] = None,
    index: int = 1,
) -> Grid: ...


@overload
def grid(
    data: ndarray,
    extent: Tuple[float, float, float, float],
    crs: Union[int, CRS],
) -> Grid: ...


@ overload
def grid(
    data: ndarray,
    extent: Affine,
    crs: CRS,
) -> Grid: ...


def grid(
    data: Union[str, MemoryFile, ndarray],
    extent: Union[Tuple[float, float, float, float], Affine, None] = None,
    crs: Union[int, CRS, None] = None,
    index: int = 1,
) -> Grid:
    """Creates new grid.

    Args:
        data (Union[str, MemoryFile, ndarray]): File path, memory file or ndarray.
        extent (Union[Tuple[float, float, float, float], Affine, None], optional):
            Map extent used to clip the raster.  If the input data is an ndarray,
            this is used to define the map extent.  Defaults to None.
        crs (Union[int, CRS, None], optional): CRS or an EPSG code (e.g. 4326).
            Defaults to None.
        index (int, optional): Band index.  Defaults to 1.

    Raises:
        ValueError: Indicating ivalid file path, etc.

    Returns:
        Grid: A new grid.
    """
    if isinstance(data, str):
        with rasterio.open(data) as dataset:
            g = _read(dataset, index, extent)
    elif isinstance(data, MemoryFile):
        with data.open() as dataset:
            g = _read(dataset, index, extent)
    elif isinstance(data, ndarray):
        if isinstance(extent, Affine):
            transform = extent
        elif extent:
            transform = from_bounds(*extent, data.shape[1], data.shape[0])
        g = Grid(format_type(data), transform, get_crs(crs))  # type: ignore
    if g:
        return g
    raise ValueError()


def _read(dataset, index, extent) -> Optional[Grid]:
    if extent:
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
            g = grid(data, from_bounds(*e, width, height), dataset.crs)
        else:
            return None
    else:
        data = dataset.read(index)
        g = grid(data, dataset.transform, dataset.crs)
    return g if dataset.nodata is None else g.set_nan(dataset.nodata)


def _extent(width, height, transform) -> Extent:
    xmin = transform.c
    ymax = transform.f
    ymin = ymax + height * transform.e
    xmax = xmin + width * transform.a
    return Extent(xmin, ymin, xmax, ymax)


def _metadata(grid: Grid):
    return {
        "height": grid.height,
        "width": grid.width,
        "crs": grid.crs,
        "transform": grid.transform,
    }


def con(grid: Grid, trueValue: Operand, falseValue: Operand):
    """Evaluates a boolean grid .

    Args:
        grid (Grid): Boolean grid.
        trueValue (Operand): Values applied for cells evaluating to true.
        falseValue (Operand): Values applied for cells evaluating to false.

    Returns:
        Grid: A new grid.
    """
    return grid.con(trueValue, falseValue)


def standardize(
    *grids: Grid,
    extent: Union[Extent, ExtentResolution] = "intersect",
    cell_size: Union[Tuple[float, float], float, None] = None,
) -> Tuple[Grid, ...]:
    """Standardizes input grids to have the same map extent and cell size.

    Args:
        extent (Union[Extent, ExtentResolution], optional): Rule for extent.
            Defaults to "intersect".
        cell_size (Union[Tuple[float, float], float, None], optional): Cell size.
            Defaults to None.

    Raises:
        ValueError: If grids are in different coordinate systems.

    Returns:
        Tuple[Grid, ...]: Standardized grids.
    """
    if len(grids) == 1:
        return tuple(grids)

    crs_set = set(grid.crs for grid in grids)

    if len(crs_set) > 1:
        raise ValueError("Input grids must have the same CRS.")

    if isinstance(cell_size, (int, float)):
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
            g = g.clip(*extent_standardized)  # type: ignore
        results.append(g)

    return tuple(results)
