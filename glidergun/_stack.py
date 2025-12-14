import contextlib
import dataclasses
import logging
import warnings
from base64 import b64encode
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from typing import Union, overload

import numpy as np
import rasterio
import requests
from matplotlib import pyplot as plt
from rasterio import DatasetReader
from rasterio.crs import CRS
from rasterio.errors import NotGeoreferencedWarning
from rasterio.io import MemoryFile
from rasterio.warp import Resampling

from glidergun._grid import Extent, Grid, _metadata, _to_uint8_range, con, from_dataset, pca, standardize
from glidergun._literals import DataType, ResamplingMethod
from glidergun._quadkey import get_tiles
from glidergun._sam import Sam
from glidergun._types import CellSize, Chart, Scaler
from glidergun._utils import create_directory_for, get_crs, get_driver, get_nodata_value

logger = logging.getLogger(__name__)


Operand = Union["Stack", Grid, float, int]


@dataclass(frozen=True)
class Stack(Sam):
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
            + f"crs: {self.crs} | "
            + f"count: {len(self.grids)} | "
            + f"rgb: {self.display} | "
            + f"cell: {self.cell_size} | "
            + f"extent: {self.extent}"
        )

    def _thumbnail(self, figsize: tuple[float, float] | None = None):
        with BytesIO() as buffer:
            figure = plt.figure(figsize=figsize, frameon=False)
            axes = figure.add_axes((0, 0, 1, 1))
            axes.axis("off")
            obj = self.each(_to_uint8_range)
            rgb = [obj.grids[i - 1].data for i in (self.display if self.display else (1, 2, 3))]
            alpha = np.where(np.isfinite(rgb[0] + rgb[1] + rgb[2]), 255, 0)
            plt.imshow(np.dstack([*[np.asanyarray(np.nan_to_num(a, nan=0), "uint8") for a in rgb], alpha]))
            plt.savefig(buffer, bbox_inches="tight", pad_inches=0)
            plt.close(figure)
            return buffer.getvalue()

    @cached_property
    def img(self) -> str:
        image = b64encode(self._thumbnail()).decode()
        return f"data:image/png;base64, {image}"

    @property
    def crs(self) -> CRS:
        return self.grids[0].crs

    @cached_property
    def width(self) -> int:
        return self.grids[0].width

    @cached_property
    def height(self) -> int:
        return self.grids[0].height

    @cached_property
    def dtype(self) -> DataType:
        return self.grids[0].dtype

    @property
    def xmin(self) -> float:
        return self.grids[0].xmin

    @property
    def ymin(self) -> float:
        return self.grids[0].ymin

    @property
    def xmax(self) -> float:
        return self.grids[0].xmax

    @property
    def ymax(self) -> float:
        return self.grids[0].ymax

    @property
    def extent(self) -> Extent:
        return self.grids[0].extent

    @property
    def cell_size(self) -> CellSize:
        return self.grids[0].cell_size

    @property
    def md5s(self) -> tuple[str, ...]:
        return tuple(g.md5 for g in self.grids)

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

    def scale(self, scaler: Scaler, **fit_params):
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
        figure, axes = plt.subplots()
        colors = {0: "red", 1: "green", 2: "blue"}
        for i, g in reversed(list(enumerate(self.grids))):
            axes.bar(list(g.bins.keys()), list(g.bins.values()), color=colors.get(i, "gray"), **kwargs)
        return Chart(figure, axes)

    def color(self, rgb: tuple[int, int, int]):
        """Sets the RGB display bands."""
        valid = set(range(1, len(self.grids) + 1))
        if set(rgb) - valid:
            raise ValueError("Invalid bands specified.")
        return dataclasses.replace(self, display=rgb)

    def each(self, func: Callable[[Grid], Grid]):
        """Applies a function to each band and returns a new `Stack`."""
        return stack(list(map(func, self.grids)))

    def georeference(self, extent: tuple[float, float, float, float] | list[float], crs: int | str | CRS | None = None):
        """Assign extent and CRS to the current data without resampling."""
        return self.each(lambda g: g.georeference(extent, crs))

    def clip(self, extent: tuple[float, float, float, float] | list[float]):
        """Clips the stack to the given extent."""
        return self.each(lambda g: g.clip(extent))

    def clip_at(self, x: float, y: float, width: int = 8, height: int = 8):
        """Clips the stack to a window centered at (x, y) with the given width and height."""
        return self.each(lambda g: g.clip_at(x, y, width, height))

    def extract_bands(self, *bands: int):
        """Extracts selected 1-based bands into a new `Stack`."""
        return stack([self.grids[i - 1] for i in bands])

    def pca(self, n_components: int = 3):
        """Performs Principal Component Analysis (PCA) on the stack bands."""
        return stack(pca(n_components, *self.grids))

    def project(self, crs: int | str | CRS, resampling: Resampling | ResamplingMethod = "nearest"):
        """Projects the stack to a new coordinate reference system (CRS)."""
        return self.each(lambda g: g.project(crs, resampling))

    def resample(self, cell_size: tuple[float, float] | float, resampling: Resampling | ResamplingMethod = "nearest"):
        """Resamples the stack to a new cell size."""
        return self.each(lambda g: g.resample(cell_size, resampling))

    def resample_by(self, times: float, resampling: Resampling | ResamplingMethod = "nearest"):
        """Resample by a scaling factor, adjusting cell size accordingly."""
        return self.each(lambda g: g.resample_by(times, resampling))

    def zip_with(self, other_stack: "Stack", func: Callable[[Grid, Grid], Grid]):
        """Combines two stacks by applying a function to corresponding bands."""
        grids = []
        for grid1, grid2 in zip(self.grids, other_stack.grids, strict=False):
            grid1, grid2 = standardize(grid1, grid2)
            grids.append(func(grid1, grid2))
        return stack(*grids)

    def value_at(self, x: float, y: float):
        """Returns the values at the given coordinates for all bands."""
        return tuple(g.value_at(x, y) for g in self.grids)

    def type(self, dtype: DataType, nan_to_num: float | None = None):
        """Converts the stack to the given data type, optionally replacing NaNs."""
        return self.each(lambda g: g.type(dtype, nan_to_num))

    def to_bytes(self, dtype: DataType | None = None, driver: str = "") -> bytes:
        """Serializes the stack to bytes (COG by default)."""
        with MemoryFile() as memory_file:
            self.save(memory_file, dtype, driver)
            return memory_file.read()

    def save(self, file: str | MemoryFile, dtype: DataType | None = None, driver: str = ""):
        """Saves the stack to disk or a `MemoryFile`.

        If the file extension is `.jpg`, `.kml`, `.kmz`, or `.png`,
        the RGB display bands are converted to `uint8` with nodata handling.
        Otherwise, all bands are saved as-is or converted to the specified dtype.

        Args:
            file: File path or `MemoryFile` to save to.
            dtype: Data type for saving (optional).
            driver: Rasterio driver name (optional).
        """
        if isinstance(file, str) and (
            file.lower().endswith(".jpg")
            or file.lower().endswith(".kml")
            or file.lower().endswith(".kmz")
            or file.lower().endswith(".png")
        ):
            grids = self.extract_bands(*self.display).each(_to_uint8_range).grids
            dtype = "uint8"
        else:
            grids = self.grids
            if dtype is None:
                dtype = grids[0].dtype

        nodata = get_nodata_value(dtype)

        if nodata is not None:
            grids = tuple(con(g.is_nan(), float(nodata), g) for g in grids)

        if isinstance(file, str):
            create_directory_for(file)
            with rasterio.open(
                file,
                "w",
                driver=driver or get_driver(file),
                count=len(grids),
                dtype=dtype,
                nodata=nodata,
                **_metadata(self.grids[0]),
            ) as dataset:
                for index, grid in enumerate(grids):
                    dataset.write(grid.data, index + 1)
        elif isinstance(file, MemoryFile):
            with file.open(
                driver=driver or "COG",
                count=len(grids),
                dtype=dtype,
                nodata=nodata,
                **_metadata(self.grids[0]),
            ) as dataset:
                for index, grid in enumerate(grids):
                    dataset.write(grid.data, index + 1)


@overload
def stack(
    data: str,
    extent: tuple[float, float, float, float] | list[float] | None = None,
    crs: int | str | CRS | None = None,
) -> Stack:
    """Creates a new stack from a file path or url.

    Args:
        data (str): File path or url.
        extent (tuple[float, float, float, float] | list[float]): Map extent used to clip the raster.
        crs (int | str | CRS | None): Coordinate reference system of the extent.


    Example:
        >>> stack("aerial_photo.tif")
        image: 373x286 float32 | crs: EPSG:4326 | count: 3 | rgb: (1, 2, 3)

    Returns:
        Stack: A new stack.
    """
    ...


@overload
def stack(
    data: str,
    extent: tuple[float, float, float, float] | list[float],
    max_tiles: int = 10,
    request: Callable[[str], requests.Response] = requests.get,
) -> Stack:
    """Creates a new stack from a tile service url template with {x}{y}{z} or {q} placeholders.

    Args:
        data (str): Tile service url template with {x}{y}{z} or {q} placeholders.
        extent (tuple[float, float, float, float] | list[float]): Map extent used to clip the raster.
        max_tiles (int, optional): Maximum number of tiles to load.  Defaults to 10.

    Example:
        >>> stack(
        ... "https://t.ssl.ak.tiles.virtualearth.net/tiles/a{q}.jpeg?g=15437",
        ... extent=(-123.164, 49.272, -123.162, 49.273),
        ... max_tiles=50
        ... )
        image: 1491x1143 uint8 | crs: 4326 | count: 3 | rgb: (1, 2, 3)

    Returns:
        Stack: A new stack.
    """
    ...


@overload
def stack(data: DatasetReader) -> Stack:  # type: ignore
    """Creates a new stack from a data reader.

    Args:
        data: Data reader.

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
def stack(data: bytes) -> Stack:
    """Creates a new stack from bytes data.

    Args:
        data: Bytes.

    Example:
        >>> stack(data)
        image: 373x286 float32 | crs: EPSG:4326 | count: 3 | rgb: (1, 2, 3)

    Returns:
        Stack: A new stack.
    """
    ...


@overload
def stack(data: MemoryFile) -> Stack:  # type: ignore
    """Creates a new stack from a memory file.

    Args:
        data: Memory file.

    Example:
        >>> stack(memory_file)
        image: 373x286 float32 | crs: EPSG:4326 | count: 3 | rgb: (1, 2, 3)

    Returns:
        Stack: A new stack.
    """
    ...


@overload
def stack(data: Sequence[Grid]) -> Stack:
    """Creates a new stack from grids.

    Args:
        data: Grids.

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
    extent: tuple[float, float, float, float] | list[float] | None = None,
    crs: int | str | CRS | None = None,
) -> Stack:
    """Creates a new stack from file paths or urls.

    Args:
        data: File paths or urls.
        extent (tuple[float, float, float, float] | list[float]): Map extent used to clip the raster.
        crs (int | str | CRS | None): Coordinate reference system of the extent.

    Example:
        >>> stack(["r.tif", "g.tif", "b.tif"])
        image: 373x286 float32 | crs: EPSG:4326 | count: 3 | rgb: (1, 2, 3)

    Returns:
        Stack: A new stack.
    """
    ...


def stack(  # type: ignore
    data: str | DatasetReader | bytes | MemoryFile | Sequence[Grid] | Sequence[str],
    extent: tuple[float, float, float, float] | list[float] | None = None,
    crs: int | str | CRS | None = None,
    max_tiles: int | None = None,
    request: Callable[[str], requests.Response] = requests.get,
) -> Stack:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=NotGeoreferencedWarning)

        if isinstance(data, str) and data.startswith("https://") and "{" in data and "}" in data and extent:
            return from_tile_service(data, extent, max_tiles or 10, request=request)

        if isinstance(data, (str, DatasetReader, bytes, MemoryFile)):
            data = [data]

        bands: list[Grid] = []

        for g in data:
            if isinstance(g, str):
                with rasterio.open(g) as dataset:
                    bands.extend(read_grids(dataset, extent, crs))
            elif isinstance(g, DatasetReader):
                bands.extend(read_grids(g, extent, crs))
            elif isinstance(g, bytes):
                with MemoryFile(g) as memory_file, memory_file.open() as dataset:
                    bands.extend(read_grids(dataset, extent, crs))
            elif isinstance(g, MemoryFile):
                with g.open() as dataset:
                    bands.extend(read_grids(dataset, extent, crs))
            elif isinstance(g, Grid):
                bands.append(g)

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
    dataset, extent: tuple[float, float, float, float] | list[float] | None, crs: int | str | CRS | None
) -> Iterator[Grid]:
    crs = get_crs(crs) if crs else None
    if dataset.subdatasets:
        for index, _ in enumerate(dataset.subdatasets):
            with rasterio.open(dataset.subdatasets[index]) as subdataset:
                yield from read_grids(subdataset, extent, crs)
    elif dataset.indexes:
        for index in dataset.indexes:
            with contextlib.suppress(Exception):
                yield from_dataset(dataset, extent, crs, None, index)


def from_tile_service(
    url_template: str,
    extent: tuple[float, float, float, float] | list[float],
    max_tiles: int,
    max_zoom: int = 24,
    request: Callable[[str], requests.Response] = requests.get,
):
    tiles = get_tiles(extent, max_tiles, max_zoom)
    r_mosaic, g_mosaic, b_mosaic = None, None, None

    for i, (x, y, z, q, xmin, ymin, xmax, ymax) in enumerate(tiles):
        url = url_template.format(x=x, y=y, z=z, q=q)
        try:
            response = request(url)
            response.raise_for_status()
            with MemoryFile(response.content) as memory_file:
                s = stack(memory_file).type("float32")
            r, g, b = s.georeference((xmin, ymin, xmax, ymax), 4326).grids
            logger.info(f"Processing tile {i + 1} of {len(tiles)}...")
            r_mosaic = r.mosaic(r_mosaic) if r_mosaic else r
            g_mosaic = g.mosaic(g_mosaic) if g_mosaic else g
            b_mosaic = b.mosaic(b_mosaic) if b_mosaic else b
        except Exception:
            if max_zoom < 18:
                raise
            return from_tile_service(url_template, extent, max_tiles, max_zoom - 1, request)

    if r_mosaic and g_mosaic and b_mosaic:
        return stack([r_mosaic, g_mosaic, b_mosaic]).clip(extent).type("uint8", 0)

    raise ValueError("No data found for the specified extent.")
