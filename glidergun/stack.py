import dataclasses
import warnings
import rasterio
from dataclasses import dataclass
from typing import (
    Callable,
    List,
    Optional,
    Tuple,
    Union,
    overload,
)
from rasterio.crs import CRS
from rasterio.drivers import driver_from_extension
from rasterio.io import MemoryFile
from rasterio.warp import Resampling
from glidergun.core import (
    Grid,
    Extent,
    Scaler,
    _metadata,
    _nodata,
    _read,
    standardize,
    con,
)
from glidergun.literals import DataType

Operand = Union["Stack", Grid, float, int]


@dataclass(frozen=True)
class Stack:
    grids: Tuple[Grid, ...]
    _rgb: Tuple[int, int, int] = (1, 2, 3)

    def __repr__(self):
        g = self.grids[0]
        return (
            f"image: {g.width}x{g.height} {g.dtype} | "
            + f"crs: {g.crs} | "
            + f"cell: {g.cell_size} | "
            + f"count: {len(self.grids)}"
        )

    @property
    def width(self) -> int:
        return self.grids[0].width

    @property
    def height(self) -> int:
        return self.grids[0].height

    @property
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
    def cell_size(self) -> float:
        return self.grids[0].cell_size

    @property
    def extent(self) -> Extent:
        return self.grids[0].extent

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

    def __eq__(self, n: Operand):
        return self._apply(n, lambda g, n: g.__eq__(n))

    __req__ = __eq__

    def __ne__(self, n: Operand):
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

    def scale(self, scaler: Optional[Scaler] = None, **fit_params):
        return self.each(lambda g: g.scale(scaler, **fit_params))

    def plot(self, *rgb: int):
        return dataclasses.replace(self, _rgb=rgb)

    def map(
        self,
        rgb: Tuple[int, int, int] = (1, 2, 3),
        opacity: float = 1.0,
        folium_map=None,
        width: int = 800,
        height: int = 600,
        basemap: Optional[str] = None,
        attribution: Optional[str] = None,
        grayscale: bool = True,
        **kwargs,
    ):
        from glidergun.ipython import _map

        return _map(
            self,
            rgb,
            opacity,
            folium_map,
            width,
            height,
            basemap,
            attribution,
            grayscale,
            **kwargs,
        )

    def each(self, func: Callable[[Grid], Grid]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return stack(*map(func, self.grids))

    def clip(self, extent: Tuple[float, float, float, float]):
        return self.each(lambda g: g.clip(extent))

    def project(
        self, epsg: Union[int, CRS], resampling: Resampling = Resampling.nearest
    ):
        return self.each(lambda g: g.project(epsg, resampling))

    def resample(self, cell_size: float, resampling: Resampling = Resampling.nearest):
        return self.each(lambda g: g.resample(cell_size, resampling))

    def zip_with(self, other_stack: "Stack", func: Callable[[Grid, Grid], Grid]):
        grids = []
        for grid1, grid2 in zip(self.grids, other_stack.grids):
            grid1, grid2 = standardize(True, grid1, grid2)
            grids.append(func(grid1, grid2))
        return stack(*grids)

    def values(self, x: float, y: float):
        return tuple(grid.value(x, y) for grid in self.grids)

    def type(self, dtype: DataType):
        return self.each(lambda g: g.type(dtype))

    @overload
    def save(self, file: str, dtype: Optional[DataType] = None, driver: str = ""):
        ...

    @overload
    def save(
        self, file: MemoryFile, dtype: Optional[DataType] = None, driver: str = ""
    ):
        ...

    def save(self, file, dtype: Optional[DataType] = None, driver: str = ""):
        g = self.grids[0]

        if dtype is None:
            dtype = self.dtype

        nodata = _nodata(dtype)

        grids = (
            self.grids
            if nodata is None
            else self.each(lambda g: con(g.is_nan(), nodata, g)).grids
        )

        if isinstance(file, str):
            with rasterio.open(
                file,
                "w",
                driver=driver if driver else driver_from_extension(file),
                count=len(grids),
                dtype=dtype,
                nodata=nodata,
                **_metadata(g),
            ) as dataset:
                for index, grid in enumerate(grids):
                    dataset.write(grid.data, index + 1)
        elif isinstance(file, MemoryFile):
            with file.open(
                driver=driver if driver else "GTiff",
                count=len(grids),
                dtype=dtype,
                nodata=nodata,
                **_metadata(g),
            ) as dataset:
                for index, grid in enumerate(grids):
                    dataset.write(grid.data, index + 1)


@overload
def stack(*grids: str) -> Stack:
    ...


@overload
def stack(*grids: MemoryFile) -> Stack:
    ...


@overload
def stack(*grids: Grid) -> Stack:
    ...


def stack(*grids) -> Stack:
    bands: List[Grid] = []

    for grid in grids:
        if isinstance(grid, Grid):
            bands.append(grid)
        else:
            with rasterio.open(grid) if isinstance(
                grid, str
            ) else grid.open() as dataset:
                for index in dataset.indexes:
                    band = _read(dataset, index)
                    bands.append(band)

    return Stack(tuple(standardize(True, *bands)))
