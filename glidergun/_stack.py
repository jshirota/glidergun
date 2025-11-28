import contextlib
import dataclasses
import logging
import warnings
from base64 import b64encode
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import cached_property, lru_cache
from io import BytesIO
from typing import Any, Union, overload

import numpy as np
import rasterio
from matplotlib import pyplot as plt
from rasterio import DatasetReader
from rasterio.crs import CRS
from rasterio.errors import NotGeoreferencedWarning, RasterioIOError
from rasterio.io import MemoryFile
from rasterio.warp import Resampling
from shapely.ops import unary_union

from glidergun._grid import Extent, Grid, _metadata, con, from_dataset, grid, pca, standardize
from glidergun._literals import BaseMap, DataType
from glidergun._quadkey import get_tiles
from glidergun._types import FeatureCollection, Scaler
from glidergun._utils import create_directory, get_crs, get_driver, get_geojson, get_nodata_value

logger = logging.getLogger(__name__)


Operand = Union["Stack", Grid, float, int]


@dataclass(frozen=True, repr=False, slots=True)
class SamResult:
    masks: list["SamMask"]
    overview: "Stack"

    def to_geojson(self, crs: int | CRS | None = 4326, smooth_factor: float = 0.0) -> FeatureCollection:
        return get_geojson(
            (m.to_polygon(crs, smooth_factor=smooth_factor), {"prompt": m.prompt, "score": m.score}) for m in self.masks
        )


@dataclass(frozen=True, slots=True)
class SamMask:
    prompt: str
    score: float
    mask: Grid

    def to_polygon(self, crs: int | CRS | None = None, smooth_factor: float = 0.0):
        m = self.mask.set_nan(0)
        if crs:
            m = m.project(crs)
        return m.to_polygons(smooth_factor=smooth_factor)[0][0]


@dataclass(frozen=True)
class Stack:
    grids: tuple[Grid, ...]
    display: tuple[int, int, int] = (1, 2, 3)

    def __repr__(self):
        return (
            f"image: {self.width}x{self.height} {self.dtype} | "
            + f"crs: {self.crs} | "
            + f"count: {len(self.grids)} | "
            + f"rgb: {self.display}"
        )

    def _thumbnail(self, figsize: tuple[float, float] | None = None):
        with BytesIO() as buffer:
            figure = plt.figure(figsize=figsize, frameon=False)
            axes = figure.add_axes((0, 0, 1, 1))
            axes.axis("off")
            obj = self._to_uint8_range()
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
        return self.each(lambda g: g.scale(scaler, **fit_params))

    def percent_clip(self, min_percent: float, max_percent: float):
        return self.each(lambda g: g.percent_clip(min_percent, max_percent))

    def _to_uint8_range(self):
        return self.each(lambda g: g._to_uint8_range())

    def color(self, rgb: tuple[int, int, int]):
        valid = set(range(1, len(self.grids) + 1))
        if set(rgb) - valid:
            raise ValueError("Invalid bands specified.")
        return dataclasses.replace(self, display=rgb)

    def map(
        self,
        opacity: float = 1.0,
        basemap: BaseMap | Any | None = None,
        width: int = 800,
        height: int = 600,
        attribution: str | None = None,
        grayscale: bool = True,
        **kwargs,
    ):
        from glidergun._display import get_folium_map

        return get_folium_map(self, opacity, basemap, width, height, attribution, grayscale, **kwargs)

    def each(self, func: Callable[[Grid], Grid]):
        return stack(*map(func, self.grids))

    def georeference(self, xmin: float, ymin: float, xmax: float, ymax: float, crs: int | CRS = 4326):
        return self.each(lambda g: g.georeference(xmin, ymin, xmax, ymax, crs))

    def clip(self, xmin: float, ymin: float, xmax: float, ymax: float):
        return self.each(lambda g: g.clip(xmin, ymin, xmax, ymax))

    def clip_at(self, x: float, y: float, width: int = 8, height: int = 8):
        return self.each(lambda g: g.clip_at(x, y, width, height))

    def extract_bands(self, *bands: int):
        return stack(*(self.grids[i - 1] for i in bands))

    def pca(self, n_components: int = 3):
        return stack(*pca(n_components, *self.grids))

    def project(self, crs: int | CRS, resampling: Resampling = Resampling.nearest):
        return self.each(lambda g: g.project(crs, resampling))

    def resample(self, cell_size: tuple[float, float] | float, resampling: Resampling = Resampling.nearest):
        return self.each(lambda g: g.resample(cell_size, resampling))

    def sam3(self, *prompt: str, model=None, confidence_threshold: float = 0.5):
        try:
            import torch
            from sam3.model.sam3_image_processor import Sam3Processor
        except ImportError as ex:
            raise ImportError("Optional dependency missing.  Please install 'sam3'.") from ex

        if model is None:
            model = _build_sam3_model()

        processor = Sam3Processor(model, device=model.device, confidence_threshold=confidence_threshold)  # type: ignore
        rgb = np.stack([g.stretch(0, 255).type("uint8", 0).data for g in self.grids[:3]], axis=-1)
        tensor = torch.from_numpy(np.transpose(rgb, (2, 0, 1))).to(model.device)
        state = processor.set_image(tensor)

        masks = []
        big_mask = self.grids[0] * 0

        for p in prompt:
            output = processor.set_text_prompt(p, state)
            for m, s in zip(output["masks"].cpu().numpy(), output["scores"].cpu().numpy(), strict=True):
                g = grid(m[0], extent=self.extent, crs=self.crs)
                masks.append(SamMask(prompt=p, score=float(s), mask=g.clip(*g.set_nan(0).data_extent) == 1))
                big_mask += g

        overview = self.each(lambda g: con(big_mask > 0, g, g / 4)).type("uint8", 0)
        return SamResult(masks=masks, overview=overview)

    def sam3_geojson(
        self,
        *prompt: str,
        model=None,
        confidence_threshold: float = 0.5,
        tile_size: int = 400,
        smooth_factor: float = 0.0,
    ):
        g = self.grids[0]
        w, h = g.cell_size.x * tile_size, g.cell_size.y * tile_size
        d: dict[str, list[SamMask]] = {p: [] for p in prompt}
        tiles = list(self.extent.tiles(w, h))

        for i, tile in enumerate(tiles):
            logger.info(f"Processing tile {i + 1} of {len(tiles)}...")
            s = self.clip(*tile.buffer(w / 2, h / 2))
            r = s.sam3(*prompt, model=model, confidence_threshold=confidence_threshold)
            for m in r.masks:
                d[m.prompt].append(m)

        return get_geojson(
            (polygon, {"prompt": p})
            for p, masks in d.items()
            for polygon in unary_union([m.to_polygon(4326, smooth_factor) for m in masks]).geoms  # type: ignore
        )

    def zip_with(self, other_stack: "Stack", func: Callable[[Grid, Grid], Grid]):
        grids = []
        for grid1, grid2 in zip(self.grids, other_stack.grids, strict=False):
            grid1, grid2 = standardize(grid1, grid2)
            grids.append(func(grid1, grid2))
        return stack(*grids)

    def value_at(self, x: float, y: float):
        return tuple(grid.value_at(x, y) for grid in self.grids)

    def type(self, dtype: DataType, nan_to_num: float | None = None):
        return self.each(lambda g: g.type(dtype, nan_to_num))

    def to_bytes(self, dtype: DataType | None = None, driver: str = "") -> bytes:
        with MemoryFile() as memory_file:
            self.save(memory_file, dtype, driver)
            return memory_file.read()

    def save(self, file: str | MemoryFile, dtype: DataType | None = None, driver: str = ""):
        if isinstance(file, str) and (
            file.lower().endswith(".jpg")
            or file.lower().endswith(".kml")
            or file.lower().endswith(".kmz")
            or file.lower().endswith(".png")
        ):
            grids = self.extract_bands(*self.display)._to_uint8_range().grids
            dtype = "uint8"
        else:
            grids = self.grids
            if dtype is None:
                dtype = grids[0].dtype

        nodata = get_nodata_value(dtype)

        if nodata is not None:
            grids = tuple(con(g.is_nan(), float(nodata), g) for g in grids)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=NotGeoreferencedWarning)
            if isinstance(file, str):
                create_directory(file)
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
def stack(*data: Grid) -> Stack:
    """Creates a new stack from grids.

    Returns:
        Stack: Grids.
    """
    ...


@overload
def stack(*data: str) -> Stack:
    """Creates a new stack from file path(s) or url(s).

    Returns:
        Stack: Grids.
    """
    ...


@overload
def stack(*data: DatasetReader) -> Stack:  # type: ignore
    """Creates a new stack from data reader.

    Returns:
        Stack: Grids.
    """
    ...


@overload
def stack(*data: bytes) -> Stack:
    """Creates a new stack from bytes.

    Returns:
        Stack: Grids.
    """
    ...


@overload
def stack(*data: MemoryFile) -> Stack:  # type: ignore
    """Creates a new stack from memory file.

    Returns:
        Stack: Grids.
    """
    ...


@overload
def stack(*data: str, extent: tuple[float, float, float, float], max_tiles: int = 10) -> Stack:
    """Creates a new stack from a tiled service url template with {x}{y}{z} or {q} placeholders.

    Args:
        data (str): Tiled service url template with {x}{y}{z} or {q} placeholders.
        extent (tuple[float, float, float, float]): Map extent used to clip the raster.
        max_tiles (int, optional): Maximum number of tiles to load.  Defaults to 10.

    Example:
        >>> stack(
        ... "https://t.ssl.ak.tiles.virtualearth.net/tiles/a{q}.jpeg?g=15437&n=z&prx=1",
        ... extent=(-123.164, 49.272, -123.162, 49.273),
        ... max_tiles=50
        ... )
        image: 1491x1143 uint8 | crs: 4326 | count: 3 | rgb: (1, 2, 3)

    Returns:
        Stack: A new stack.
    """
    ...


def stack(
    *data: Grid | str | DatasetReader | bytes | MemoryFile,
    extent: tuple[float, float, float, float] | None = None,
    crs: int | CRS | None = None,
    max_tiles: int | None = None,
) -> Stack:
    bands: list[Grid] = []

    for g in data:
        if isinstance(g, str):
            if g.startswith("https://") and "{" in g and "}" in g and extent:
                return from_tile_service(g, extent, max_tiles or 10)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=NotGeoreferencedWarning)
                with rasterio.open(g) as dataset:
                    bands.extend(_read_grids(dataset, extent, crs))
        elif isinstance(g, DatasetReader):
            bands.extend(_read_grids(g, extent, crs))
        elif isinstance(g, bytes):
            with MemoryFile(g) as memory_file, memory_file.open() as dataset:
                bands.extend(_read_grids(dataset, extent, crs))
        elif isinstance(g, MemoryFile):
            with g.open() as dataset:
                bands.extend(_read_grids(dataset, extent, crs))
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


def _read_grids(dataset, extent: tuple[float, float, float, float] | None, crs: int | CRS | None) -> Iterator[Grid]:
    crs = get_crs(crs) if crs else None
    if dataset.subdatasets:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=NotGeoreferencedWarning)
            for index, _ in enumerate(dataset.subdatasets):
                with rasterio.open(dataset.subdatasets[index]) as subdataset:
                    yield from _read_grids(subdataset, extent, crs)
    elif dataset.indexes:
        for index in dataset.indexes:
            with contextlib.suppress(Exception):
                yield from_dataset(dataset, extent, crs, None, index)


def from_tile_service(url: str, extent: tuple[float, float, float, float], max_tiles: int, max_zoom: int = 24):
    tiles = get_tiles(*extent, max_tiles, max_zoom)
    r_mosaic, g_mosaic, b_mosaic = None, None, None

    for i, (x, y, z, q, xmin, ymin, xmax, ymax) in enumerate(tiles):
        logger.info(f"Processing {i + 1} of {len(tiles)} tiles...")
        url2 = url.replace("{x}", str(x)).replace("{y}", str(y)).replace("{z}", str(z)).replace("{q}", q)
        try:
            s = stack(url2)
        except RasterioIOError as ex:
            if max_zoom < 18:
                raise
            logger.warning(f"Failed to load tile from {url2}: {ex}")
            return from_tile_service(url, extent, max_tiles, max_zoom - 1)
        r, g, b = s.georeference(xmin, ymin, xmax, ymax, 4326).grids
        r_mosaic = r.mosaic(r_mosaic) if r_mosaic else r
        g_mosaic = g.mosaic(g_mosaic) if g_mosaic else g
        b_mosaic = b.mosaic(b_mosaic) if b_mosaic else b

    if r_mosaic and g_mosaic and b_mosaic:
        return stack(r_mosaic, g_mosaic, b_mosaic).clip(*extent).type("uint8", 0)

    raise ValueError("No data found for the specified extent.")


@lru_cache(maxsize=1)
def _build_sam3_model():
    from sam3.model_builder import build_sam3_image_model

    return build_sam3_image_model()
