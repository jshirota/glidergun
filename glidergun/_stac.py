from typing import Literal, Never, cast, overload

import planetary_computer as pc
import requests
from pystac.item import Item as PystacItem
from rasterio.crs import CRS
from shapely.geometry import box, shape

from glidergun._grid import Grid, standardize
from glidergun._mosaic import mosaic
from glidergun._stack import Stack, stack
from glidergun._types import Extent

planetary_computer_url = "https://planetarycomputer.microsoft.com/api/stac/v1"


class ItemBase(PystacItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.url: str = None  # type: ignore
        self.extent: Extent = None  # type: ignore
        self.crs: CRS = None  # type: ignore

    def get_url(self, asset: str) -> str:
        if asset not in self.assets:
            raise ValueError(f"Asset '{asset}' not found in item assets: {list(self.assets.keys())}")
        url = self.assets[asset].href
        if self.url == planetary_computer_url:
            return pc.sign(url)
        return url

    def download(self, asset: str | list[str]) -> Grid | Stack:
        if isinstance(asset, list):
            return stack(standardize(*(cast(Grid, self.download(a)) for a in asset)))
        s = stack(self.get_url(asset), self.extent, self.crs)
        if len(s.grids) == 1:
            return s.grids[0]
        return s


class Item[TGrid: str, TStack: str](ItemBase):
    @overload
    def download(self, asset: TGrid) -> Grid: ...

    @overload
    def download(self, asset: TStack) -> Stack: ...

    @overload
    def download(self, asset: list[TGrid]) -> Stack: ...

    def download(self, asset):
        return super().download(asset)


@overload
def search(
    collection: Literal["landsat-c2-l2"],
    extent: tuple[float, float, float, float] | list[float],
    query: dict | None = None,
    *,
    fully_contains_search_area: bool = True,
    cloud_cover_percent: float | None = None,
) -> list[
    Item[
        Literal[
            "qa",
            "red",
            "blue",
            "drad",
            "emis",
            "emsd",
            "trad",
            "urad",
            "atran",
            "cdist",
            "green",
            "nir08",
            "lwir11",
            "swir16",
            "swir22",
            "coastal",
            "qa_pixel",
            "qa_radsat",
            "qa_aerosol",
        ],
        Never,
    ]
]: ...


@overload
def search(
    collection: Literal["sentinel-2-l2a"],
    extent: tuple[float, float, float, float] | list[float],
    query: dict | None = None,
    *,
    fully_contains_search_area: bool = True,
    cloud_cover_percent: float | None = None,
) -> list[
    Item[
        Literal[
            "AOT", "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B11", "B12", "B8A", "SCL", "WVP"
        ],
        Literal["visual"],
    ]
]: ...


@overload
def search(
    collection: str,
    extent: tuple[float, float, float, float] | list[float],
    query: dict | None = None,
    *,
    url: str = planetary_computer_url,
    fully_contains_search_area: bool = True,
) -> list[ItemBase]: ...


def search(
    collection: str,
    extent: tuple[float, float, float, float] | list[float],
    query: dict | None = None,
    *,
    url: str = planetary_computer_url,
    fully_contains_search_area: bool = True,
    cloud_cover_percent: float | None = None,
):
    xmin, ymin, xmax, ymax = extent
    search_extent = Extent(xmin, ymin, xmax, ymax)
    search_polygon = box(xmin, ymin, xmax, ymax)

    response = requests.post(
        f"{url}/search",
        json={
            "bbox": list(extent),
            "collections": [collection],
            "query": {"eo:cloud_cover": {"lt": cloud_cover_percent}} | (query or {})
            if cloud_cover_percent is not None
            else query or {},
        },
    )

    response.raise_for_status()

    features = []
    from_crs = CRS.from_epsg(4326)

    for feature in response.json()["features"]:
        geometry = shape(feature["geometry"])
        if not fully_contains_search_area or search_polygon.within(geometry):
            if fully_contains_search_area:
                data_extent = search_extent
            else:
                data_extent = Extent(*search_polygon.intersection(geometry).bounds)
            to_crs = CRS.from_epsg(feature["properties"]["proj:epsg"])
            item = ItemBase.from_dict(feature)
            item.url = url
            item.extent = data_extent.project(from_crs, to_crs)
            item.crs = to_crs
            features.append(item)

    return features


def search_mosaic(
    url: str, collection: str, asset: str, extent: tuple[float, float, float, float] | list[float]
) -> Grid:
    items = search(collection, extent, url=url, fully_contains_search_area=False)
    if not items:
        raise ValueError(f"No items found for the given extent: {extent}")
    g = mosaic(*[i.get_url(asset) for i in items]).clip(extent)
    if not g:
        raise ValueError(f"Mosaic resulted in no data for the given extent: {extent}")
    return g
