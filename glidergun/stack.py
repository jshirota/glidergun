import contextlib
import dataclasses
import inspect
import io
import logging
import warnings
from base64 import b64encode
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Union, overload

import numpy as np
import PIL.Image as Image
import rasterio
from rasterio import DatasetReader
from rasterio.crs import CRS
from rasterio.errors import NotGeoreferencedWarning
from rasterio.io import MemoryFile
from rasterio.transform import Affine
from rasterio.warp import Resampling
from requests.exceptions import HTTPError
from shapely.geometry.base import BaseGeometry

from glidergun.grid import (
    Extent,
    Grid,
    _to_uint8_range,
    con,
    from_dataset,
    grid,
    maximum,
    mean,
    minimum,
    pca,
    standardize,
    std,
)
from glidergun.io import http_get, write_to_file
from glidergun.literals import Basemap, BasemapUrl, DataType, ExtentResolution, ResamplingMethod
from glidergun.plot import create_histogram, create_thumbnail
from glidergun.quadkey import get_rows
from glidergun.types import BBox, CellSize, Chart, Scaler
from glidergun.utils import add_to_active_map, get_crs_name, get_driver, get_nodata_value, is_image_format, is_tile_url

logger = logging.getLogger(__name__)

Operand = Union["Stack", Grid, float, int]


@dataclass(frozen=True)
class Stack:
    """A multi-band raster container.

    Wraps a tuple of `Grid` objects representing bands of an image and
    provides band-wise arithmetic, transformations, visualization, and I/O.

    Attributes:
        grids: Tuple of `Grid` bands in the stack.
        display: 1-based band indices for RGB visualization order.
    """

    grids: tuple[Grid, ...]
    display: tuple[int, int, int] = (1, 2, 3)

    def __repr__(self):
        return (
            f"image: {self.width}x{self.height} {self.dtype} | "
            + f"crs: {self.crs} {get_crs_name(self.crs)} | "
            + f"count: {len(self.grids)} | "
            + f"rgb: {self.display} | "
            + f"cell: {self.cell_size} | "
            + f"extent: {self.extent}"
        )

    @cached_property
    def img(self) -> str:
        resample_factor = max(self.width / 1000, self.height / 1000, 1)
        s = self.resample_by(resample_factor) if resample_factor > 1 else self
        if len(s.grids) < 3:
            logger.warning("Stack has fewer than 3 bands.  Displaying the mean.")
            g = _to_uint8_range(s.mean())
            obj = stack((g, g, g))
        else:
            obj = s.each(_to_uint8_range)
        rgb = [obj.grids[i - 1].data for i in (self.display if self.display else (1, 2, 3))]
        alpha = np.where(np.isfinite(rgb[0] + rgb[1] + rgb[2]), 255, 0)
        array = np.dstack([*[np.asanyarray(np.nan_to_num(a, nan=0), "uint8") for a in rgb], alpha])
        bytes_data = create_thumbnail(array)
        image = b64encode(bytes_data).decode()
        return f"data:image/png;base64, {image}"

    @property
    def first(self) -> Grid:
        """Returns the first grid in the stack."""
        return self.grids[0]

    @property
    def transform(self) -> Affine:
        return self.first.transform

    @property
    def crs(self) -> CRS:
        return self.first.crs

    @cached_property
    def width(self) -> int:
        return self.first.width

    @cached_property
    def height(self) -> int:
        return self.first.height

    @cached_property
    def dtype(self) -> DataType:
        return self.first.dtype

    @property
    def xmin(self) -> float:
        return self.first.xmin

    @property
    def ymin(self) -> float:
        return self.first.ymin

    @property
    def xmax(self) -> float:
        return self.first.xmax

    @property
    def ymax(self) -> float:
        return self.first.ymax

    @property
    def extent(self) -> Extent:
        return self.first.extent

    @property
    def extent_4326(self) -> Extent:
        return self.first.extent_4326

    @property
    def cell_size(self) -> CellSize:
        return self.first.cell_size

    @property
    def sha256s(self) -> tuple[str, ...]:
        return tuple(g.sha256 for g in self.grids)

    def __add__(self, n: Operand):
        return self._apply(n, lambda g, n: g.__add__(n))

    __radd__ = __add__

    def __sub__(self, n: Operand):
        return self._apply(n, lambda g, n: g.__sub__(n))

    def __rsub__(self, n: Operand):
        return self._apply(n, lambda g, n: g.__rsub__(n))

    def __mul__(self, n: Operand):
        return self._apply(n, lambda g, n: g.__mul__(n))

    __rmul__ = __mul__

    def __pow__(self, n: Operand):
        return self._apply(n, lambda g, n: g.__pow__(n))

    def __rpow__(self, n: Operand):
        return self._apply(n, lambda g, n: g.__rpow__(n))

    def __truediv__(self, n: Operand):
        return self._apply(n, lambda g, n: g.__truediv__(n))

    def __rtruediv__(self, n: Operand):
        return self._apply(n, lambda g, n: g.__rtruediv__(n))

    def __floordiv__(self, n: Operand):
        return self._apply(n, lambda g, n: g.__floordiv__(n))

    def __rfloordiv__(self, n: Operand):
        return self._apply(n, lambda g, n: g.__rfloordiv__(n))

    def __mod__(self, n: Operand):
        return self._apply(n, lambda g, n: g.__mod__(n))

    def __rmod__(self, n: Operand):
        return self._apply(n, lambda g, n: g.__rmod__(n))

    def __lt__(self, n: Operand):
        return self._apply(n, lambda g, n: g.__lt__(n))

    def __gt__(self, n: Operand):
        return self._apply(n, lambda g, n: g.__gt__(n))

    __rlt__ = __gt__

    __rgt__ = __lt__

    def __le__(self, n: Operand):
        return self._apply(n, lambda g, n: g.__le__(n))

    def __ge__(self, n: Operand):
        return self._apply(n, lambda g, n: g.__ge__(n))

    __rle__ = __ge__

    __rge__ = __le__

    def __eq__(self, n: object):
        if not isinstance(n, (Grid | float | int)):
            return NotImplemented
        return self._apply(n, lambda g, n: g.__eq__(n))

    __req__ = __eq__

    def __ne__(self, n: object):
        if not isinstance(n, (Grid | float | int)):
            return NotImplemented
        return self._apply(n, lambda g, n: g.__ne__(n))

    __rne__ = __ne__

    def __and__(self, n: Operand):
        return self._apply(n, lambda g, n: g.__and__(n))

    __rand__ = __and__

    def __or__(self, n: Operand):
        return self._apply(n, lambda g, n: g.__or__(n))

    __ror__ = __or__

    def __xor__(self, n: Operand):
        return self._apply(n, lambda g, n: g.__xor__(n))

    __rxor__ = __xor__

    def __rshift__(self, n: Operand):
        return self._apply(n, lambda g, n: g.__rshift__(n))

    def __lshift__(self, n: Operand):
        return self._apply(n, lambda g, n: g.__lshift__(n))

    __rrshift__ = __lshift__

    __rlshift__ = __rshift__

    def __neg__(self):
        return self.each(lambda g: g.__neg__())

    def __pos__(self):
        return self.each(lambda g: g.__pos__())

    def __invert__(self):
        return self.each(lambda g: g.__invert__())

    def _apply(self, n: Operand, op: Callable):
        if isinstance(n, Stack):
            return self.zip_with(n, lambda g1, g2: op(g1, g2))
        return self.each(lambda g: op(g, n))

    def scale(self, scaler: Scaler | None = None, **fit_params):
        """Scales the stack using the given scaler."""
        return self.each(lambda g: g.scale(scaler, **fit_params))

    def percent_clip(self, min_percent: float, max_percent: float):
        """Clips the stack values to the given percentile range."""
        return self.each(lambda g: g.percent_clip(min_percent, max_percent))

    def stretch(self, min_value: float, max_value: float):
        """Linearly stretch values to `[min_value, max_value]`."""
        return self.each(lambda g: g.stretch(min_value, max_value))

    def hist(self, **kwargs) -> Chart:
        """Build a histogram chart of value counts (NaN-aware)."""
        colors = {0: "red", 1: "green", 2: "blue"}
        return create_histogram(*((g.bins, colors.get(i, "gray")) for i, g in enumerate(self.grids)))

    def color(self, rgb: tuple[int, int, int]):
        """Sets the RGB display bands."""
        valid = set(range(1, len(self.grids) + 1))
        if set(rgb) - valid:
            raise ValueError("Invalid bands specified.")
        return dataclasses.replace(self, display=rgb)

    @overload
    def each(self, func: Callable[[Grid], Grid]) -> "Stack":
        """Applies a function (of Grid) and returns a new `Stack`."""
        ...

    @overload
    def each(self, func: Callable[[int, Grid], Grid]) -> "Stack":
        """Applies a function (of band number and Grid) and returns a new `Stack`."""
        ...

    def each(self, func: Callable[[Grid], Grid] | Callable[[int, Grid], Grid]):
        arg_count = len(inspect.signature(func).parameters)
        if arg_count == 1:
            return stack([func(g) for g in self.grids])  # type: ignore
        return stack([func(i, g) for i, g in enumerate(self.grids, 1)])  # type: ignore

    def local(self, func: Callable[[np.ndarray], np.ndarray] | np.ndarray):
        """Apply a numpy function to the data for each grid."""
        return self.each(lambda g: g.local(func))

    def georeference(self, extent: BBox, crs: int | str | CRS | None = None):
        """Assign extent and CRS to the current data without resampling."""
        return self.each(lambda g: g.georeference(extent, crs))

    def georeference_to(self, reference: "Grid | Stack"):
        """Automatically georeference to a reference grid or stack using scan + LoFTR."""
        try:
            from glidergun.loftr import georeference_to_reference
        except ImportError as ex:
            raise ImportError(
                "This method requires the torch option.  Please install it with `pip install glidergun[torch]`."
            ) from ex

        return georeference_to_reference(self, reference)

    def mosaic(self, *stacks: "Stack", blend: bool = False):
        return self.each(lambda i, g: g.mosaic(*(s.grids[i - 1] for s in stacks), blend=blend))

    def clip(self, extent: BBox):
        """Clips the stack to the given extent."""
        return self.each(lambda g: g.clip(extent))

    def clip_at(self, x: float, y: float, width: int = 8, height: int = 8):
        """Clips the stack to a window centered at (x, y) with the given width and height."""
        return self.each(lambda g: g.clip_at(x, y, width, height))

    def extract_bands(self, *bands: int):
        """Extracts selected 1-based bands into a new `Stack`."""
        return stack([self.grids[i - 1] for i in bands])

    def mean(self):
        """Calculates the mean across all bands."""
        return mean(*self.grids)

    def std(self):
        """Calculates the standard deviation across all bands."""
        return std(*self.grids)

    def min(self):
        """Calculates the minimum value across all bands."""
        return minimum(*self.grids)

    def max(self):
        """Calculates the maximum value across all bands."""
        return maximum(*self.grids)

    @overload
    def pca(self, n_components: int) -> "Stack": ...

    @overload
    def pca(self) -> Grid: ...

    def pca(self, n_components: int | None = None):
        """Performs Principal Component Analysis (PCA) on the stack bands."""
        if n_components:
            return stack(pca(n_components, *self.grids))
        return stack(pca(1, *self.grids)).first

    def project(self, crs: int | str | CRS, resampling: Resampling | ResamplingMethod = "nearest"):
        """Projects the stack to a new coordinate reference system (CRS)."""
        return self.each(lambda g: g.project(crs, resampling))

    def resample(self, cell_size: tuple[float, float] | float, resampling: Resampling | ResamplingMethod = "nearest"):
        """Resamples the stack to a new cell size."""
        return self.each(lambda g: g.resample(cell_size, resampling))

    def resample_by(self, factor: float, resampling: Resampling | ResamplingMethod = "nearest"):
        """Resample by a scaling factor, adjusting cell size accordingly."""
        return self.each(lambda g: g.resample_by(factor, resampling))

    def canny(self, threshold1: float = 100.0, threshold2: float = 200.0, **kwargs):
        """Detect edges using the Canny algorithm with given thresholds."""
        return self.each(lambda g: g.canny(threshold1, threshold2, **kwargs))

    def gaussian_filter(self, sigma: float = 1.0, **kwargs):
        """Apply Gaussian filter to the stack."""
        return self.each(lambda g: g.gaussian_filter(sigma=sigma, **kwargs))

    def sobel(self, axis: int = -1, **kwargs):
        """Apply Sobel filter to the stack."""
        return self.each(lambda g: g.sobel(axis=axis, **kwargs))

    def zip_with(
        self, other_stack: "Stack", func: Callable[[Grid, Grid], Grid], extent: ExtentResolution | Extent = "intersect"
    ):
        """Combines two stacks by applying a function to corresponding bands."""
        grids = []
        for grid1, grid2 in zip(self.grids, other_stack.grids, strict=True):
            grid1, grid2 = standardize(grid1, grid2, extent=extent)
            grids.append(func(grid1, grid2))
        return stack(grids)

    def overlay(self, other: "Grid | Stack", alpha: float = 1.0):
        """Overlays another grid or stack on top of the current stack with alpha blending."""
        return self.zip_with(
            other if isinstance(other, Stack) else other.to_stack(),
            lambda g1, g2: con(g2.is_nan(), g1, g1 * (1 - alpha) + g2 * alpha),
            extent="first",
        )

    def draw(self, shape: BaseGeometry, *values: float | None):
        """Draws a shape on the stack with the given values."""
        if not values:
            values = (None,) * len(self.grids)
        elif len(values) == 1:
            values = values * len(self.grids)
        elif len(values) != len(self.grids):
            raise ValueError("Number of values must match number of bands.")
        return self.each(lambda i, g: g.draw(shape, values[i - 1]))

    def value_at(self, x: float, y: float):
        """Returns the values at the given coordinates for all bands."""
        return tuple(g.value_at(x, y) for g in self.grids)

    def type(self, dtype: DataType, nan_to_num: float | None = None):
        """Converts the stack to the given data type, optionally replacing NaNs."""
        return self.each(lambda g: g.type(dtype, nan_to_num))

    def to_array(self):
        return np.stack([g.data for g in self.grids], axis=-1)

    def to_bytes(self, dtype: DataType | None = None, driver: str = "") -> bytes:
        """Serializes the stack to bytes (COG by default)."""
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
            self,
            *prompt,
            checkpoint_path=checkpoint_path,
            hf_token=hf_token,
            confidence_threshold=confidence_threshold,
            tile_size=tile_size,
        )

    @overload
    def save(self, file: str, dtype: DataType | None = None) -> None:
        """Saves the stack to a file.

        Most commonly, `.tif` extension is used to save as a GeoTIFF (COG).
        If the file extension is `.jpg`, `.kml`, `.kmz`, `.mbtiles`, `.png`
        or `.webp`, the stack is converted to an RGB image with nodata handling
        if supported by the format.  Otherwise, all bands are saved as-is.

        Args:
            file (str): File path (e.g. "output.tif").
            dtype (DataType | None): Data type for saving (optional).
        """
        ...

    @overload
    def save(self, file: MemoryFile, dtype: DataType | None, driver: str) -> None:
        """Saves the stack to a memory file.

        Args:
            file (MemoryFile): Memory file to save the stack to.
            dtype (DataType | None): Data type for saving (optional).
            driver (str): Rasterio driver name (optional).
        """
        ...

    def save(self, file: str | MemoryFile, dtype: DataType | None = None, driver: str = ""):
        driver = driver or get_driver(file)

        if is_image_format(driver, file):
            grids = self.extract_bands(*self.display).each(lambda g: _to_uint8_range(g)).grids
            if driver.lower() in ("mbtiles", "png", "webp"):
                alpha = con(grids[0].is_nan() or grids[1].is_nan() or grids[2].is_nan(), 0, 255)
                grids = tuple(g.coalesce(0) for g in grids) + (alpha,)
            else:
                grids = tuple(g.coalesce(0) for g in grids)
            dtype = "uint8"
        else:
            grids = self.grids
            if dtype is None:
                dtype = grids[0].dtype

        nodata = get_nodata_value(dtype)

        if nodata is not None:
            grids = tuple(g.coalesce(nodata) for g in grids)

        write_to_file(file, driver, dtype, nodata, *grids)

    def add_to_active_map(self):
        """Add as a layer to the active map in ArcGIS or QGIS."""
        add_to_active_map(self)


@overload
def stack(
    data: str,
    extent: BBox | None = None,
    *,
    resample_by: float | None = None,
    resampling: Resampling | ResamplingMethod = "nearest",
) -> Stack:
    """Creates a new stack from a file path or url.

    Args:
        data (str): File path or url.
        extent (BBox | None): Map extent used to clip the raster.

    Example:
        >>> stack("aerial_photo.tif")
        image: 373x286 float32 | crs: EPSG:4326 | count: 3 | rgb: (1, 2, 3)

    Returns:
        Stack: A new stack.
    """
    ...


@overload
def stack(
    data: Basemap | str,
    extent: BBox | None = (-179.999, -85.0511, 179.999, 85.0511),
    *,
    max_tiles: int = 10,
    cache_dir: str | None = "~/.glidergun/cache",
    preserve_alpha: bool = False,
) -> Stack:
    """Creates a new stack from a tile service url template with {x}{y}{z} or {q} placeholders.

    Args:
        data (str): Tile service url template with {x}{y}{z} or {q} placeholders.
        extent (BBox): Map extent used to clip the raster.
        max_tiles (int, optional): Maximum number of tiles to load.  Defaults to 10.
        cache_dir (str | None, optional): Directory to cache downloaded tiles.  Defaults to "~/.glidergun/cache".
        preserve_alpha (bool, optional): Whether to preserve the alpha channel.  Defaults to False.

    Example:
        >>> stack("microsoft", (-123.164, 49.272, -123.162, 49.273), max_tiles=10)
        image: 373x285 uint8 | crs: EPSG:3857 | count: 3 | rgb: (1, 2, 3)

    Returns:
        Stack: A new stack in Web Mercator.
    """
    ...


@overload
def stack(  # type: ignore
    data: DatasetReader,
    extent: BBox | None = None,
    *,
    resample_by: float | None = None,
    resampling: Resampling | ResamplingMethod = "nearest",
) -> Stack:  # type: ignore
    """Creates a new stack from a data reader.

    Args:
        data: Data reader.
        extent (BBox | None): Map extent used to clip the raster.

    Example:
        >>> with rasterio.open("aerial_photo.tif") as dataset:
        ...     stack(dataset)
        ...
        image: 373x286 float32 | crs: EPSG:4326 | count: 3 | rgb: (1, 2, 3)

    Returns:
        Stack: A new stack.
    """
    ...


@overload
def stack(
    data: bytes,
    extent: BBox | None = None,
    *,
    resample_by: float | None = None,
    resampling: Resampling | ResamplingMethod = "nearest",
) -> Stack:
    """Creates a new stack from bytes data.

    Args:
        data: Bytes.
        extent (BBox | None): Map extent used to clip the raster.

    Example:
        >>> stack(data)
        image: 373x286 float32 | crs: EPSG:4326 | count: 3 | rgb: (1, 2, 3)

    Returns:
        Stack: A new stack.
    """
    ...


@overload
def stack(  # type: ignore
    data: MemoryFile,
    extent: BBox | None = None,
    *,
    resample_by: float | None = None,
    resampling: Resampling | ResamplingMethod = "nearest",
) -> Stack:  # type: ignore
    """Creates a new stack from a memory file.

    Args:
        data: Memory file.
        extent (BBox | None): Map extent used to clip the raster.

    Example:
        >>> stack(memory_file)
        image: 373x286 float32 | crs: EPSG:4326 | count: 3 | rgb: (1, 2, 3)

    Returns:
        Stack: A new stack.
    """
    ...


@overload
def stack(
    data: Sequence[Grid],
    extent: BBox | None = None,
) -> Stack:
    """Creates a new stack from grids.

    Args:
        data: Grids.
        extent (BBox | None): Map extent used to clip the raster.

    Example:
        >>> stack([r_grid, g_grid, b_grid])
        image: 373x286 float32 | crs: EPSG:4326 | count: 3 | rgb: (1, 2, 3)

    Returns:
        Stack: A new stack.
    """
    ...


@overload
def stack(
    data: Sequence[str],
    extent: BBox | None = None,
) -> Stack:
    """Creates a new stack from file paths or urls.

    Args:
        data: File paths or urls.
        extent (BBox | None): Map extent used to clip the raster.

    Example:
        >>> stack(["r.tif", "g.tif", "b.tif"])
        image: 373x286 float32 | crs: EPSG:4326 | count: 3 | rgb: (1, 2, 3)

    Returns:
        Stack: A new stack.
    """
    ...


def stack(
    data: str | DatasetReader | bytes | MemoryFile | Sequence[Grid] | Sequence[str],
    extent: BBox | None = None,
    *,
    resample_by: float | None = None,
    resampling: Resampling | ResamplingMethod = "nearest",
    max_tiles: int | None = None,
    cache_dir: str | None = "~/.glidergun/cache",
    preserve_alpha: bool = False,
) -> Stack:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=NotGeoreferencedWarning)

        if isinstance(data, str):
            if data in BasemapUrl:
                data = BasemapUrl[data]
            if is_tile_url(data):
                return from_tile_service(
                    data,
                    extent or (-179.999, -85.0511, 179.999, 85.0511),
                    max_tiles or 10,
                    24,
                    cache_dir,
                    preserve_alpha,
                )

        if isinstance(data, str | DatasetReader | bytes | MemoryFile):
            data = [data]

        bands: list[Grid] = []

        for g in data:
            if isinstance(g, str):
                with rasterio.open(g) as dataset:
                    bands.extend(read_grids(dataset, extent, resample_by, resampling))
            elif isinstance(g, DatasetReader):
                bands.extend(read_grids(g, extent, resample_by, resampling))
            elif isinstance(g, bytes):
                with MemoryFile(g) as memory_file, memory_file.open() as dataset:
                    bands.extend(read_grids(dataset, extent, resample_by, resampling))
            elif isinstance(g, MemoryFile):
                with g.open() as dataset:
                    bands.extend(read_grids(dataset, extent, resample_by, resampling))
            elif isinstance(g, Grid):
                bands.append(g.clip(extent) if extent else g)

        def assert_unique(property: str):
            values = {getattr(g, property) for g in bands}
            if len(values) > 1:
                raise ValueError(f"All grids must have the same {property}.")

        assert_unique("width")
        assert_unique("height")
        assert_unique("dtype")
        assert_unique("crs")
        assert_unique("extent")

        return Stack(tuple(bands))


def read_grids(
    dataset, extent: BBox | None, resample_by: float | None, resampling: Resampling | ResamplingMethod
) -> Iterator[Grid]:
    if dataset.subdatasets:
        for index, _ in enumerate(dataset.subdatasets):
            with rasterio.open(dataset.subdatasets[index]) as subdataset:
                yield from read_grids(subdataset, extent, resample_by, resampling)
    elif dataset.indexes:
        for index in dataset.indexes:
            with contextlib.suppress(Exception):
                yield from_dataset(dataset, extent, index, resample_by, resampling)


def from_tile_service(
    url_template: str,
    extent: BBox,
    max_tiles: int,
    max_zoom: int,
    cache_dir: str | None,
    preserve_alpha: bool,
) -> Stack:
    if isinstance(extent, Extent):
        extent = extent.project(4326)

    rows = get_rows(extent, max_tiles, max_zoom)

    if not rows:
        raise ValueError("No tiles found for the specified extent and max_tiles.")

    r_rows, g_rows, b_rows, a_rows = [], [], [], []
    e: Extent | None = None
    in_crs = CRS.from_epsg(4326)
    out_crs = CRS.from_epsg(3857)
    failed_tile = np.full((256, 256), np.nan)

    for i, row in enumerate(rows):
        r_columns, g_columns, b_columns, a_columns = [], [], [], []
        for j, (x, y, z, q, xmin, ymin, xmax, ymax) in enumerate(row):
            logger.info(f"Processing tile {i * len(row) + j + 1} of {sum(len(r) for r in rows)}...")
            _e = Extent(xmin, ymin, xmax, ymax, out_crs)
            e = e.union(_e) if e else _e
            url = url_template.format(x=x, y=y, z=z, q=q)
            try:
                data = http_get(url, cache_dir=cache_dir)
                with Image.open(io.BytesIO(data)) as image:
                    rgb_img = image.convert("RGBA")
                    rgb_array = np.array(rgb_img, dtype=np.uint8)
                    r_columns.append(rgb_array[:, :, 0])
                    g_columns.append(rgb_array[:, :, 1])
                    b_columns.append(rgb_array[:, :, 2])
                    if preserve_alpha:
                        a_columns.append(rgb_array[:, :, 3])
            except HTTPError as ex:
                logging.warning(f"Failed to download tile at {url}: {ex}")
                r_columns.append(failed_tile)
                g_columns.append(failed_tile)
                b_columns.append(failed_tile)
                if preserve_alpha:
                    a_columns.append(failed_tile)
            except Exception:
                if max_zoom < 18:
                    raise
                return from_tile_service(url_template, extent, max_tiles, max_zoom - 1, cache_dir, preserve_alpha)
        r_rows.append(r_columns)
        g_rows.append(g_columns)
        b_rows.append(b_columns)
        if preserve_alpha:
            a_rows.append(a_columns)

    s = stack([grid(np.block(r), e) for r in (r_rows, g_rows, b_rows)])

    if preserve_alpha:
        alpha = grid(np.block(a_rows), e) < 1
        s = s.each(lambda g: g.set_nan(alpha))

    s = s.clip(Extent(*extent, crs=in_crs).project(out_crs))

    return s if preserve_alpha else s.type("uint8")
