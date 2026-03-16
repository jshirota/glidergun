from datetime import datetime, timezone
from typing import Generic, Literal, TypeVar, cast, overload
from urllib.parse import urlencode

try:
    from typing import Never  # type: ignore
except ImportError:  # Python < 3.11
    from typing_extensions import Never

import requests
from pystac.item import Item as PystacItem
from rasterio.crs import CRS
from shapely.geometry import box, shape

from glidergun.grid import Grid, standardize
from glidergun.mosaic import mosaic
from glidergun.stack import Stack, stack
from glidergun.types import Extent

planetary_computer_url = "https://planetarycomputer.microsoft.com/api/stac/v1"

DateRange = str | tuple[str | None, str | None] | tuple[datetime | None, datetime | None]


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
            import planetary_computer as pc

            return pc.sign(url)
        return url

    def download(self, asset: str | list[str], clip_to_extent: bool = True, preview: bool = True) -> Grid | Stack:
        is_preview = preview and self.url == planetary_computer_url
        if is_preview:
            params = {"collection": self.collection_id, "item": self.id, "assets": asset}
            url = "https://planetarycomputer.microsoft.com/api/data/v1/item/preview.tif?" + urlencode(
                params, doseq=True
            )
            s = stack(url, self.extent if clip_to_extent else None, self.crs)
        else:
            if isinstance(asset, list):
                return stack(standardize(*(cast(Grid, self.download(a, clip_to_extent, preview)) for a in asset)))
            s = stack(self.get_url(asset), self.extent if clip_to_extent else None, self.crs)
        if is_preview:
            s = stack([g.set_nan(s.grids[-1] == 0) for g in s.grids[:-1]])
        if len(s.grids) == 1:
            return s.first
        return s


TGrid = TypeVar("TGrid", bound=str)
TStack = TypeVar("TStack", bound=str)


class Item(ItemBase, Generic[TGrid, TStack]):
    @overload
    def download(self, asset: TGrid, clip_to_extent: bool = True, preview: bool = True) -> Grid: ...

    @overload
    def download(self, asset: TStack, clip_to_extent: bool = True, preview: bool = True) -> Stack: ...

    @overload
    def download(self, asset: list[TGrid], clip_to_extent: bool = True, preview: bool = True) -> Stack: ...

    def download(self, asset, clip_to_extent: bool = True, preview: bool = True):
        return super().download(asset, clip_to_extent=clip_to_extent, preview=preview)


@overload
def search(
    collection: Literal["landsat-c2-l2"],
    extent: tuple[float, float, float, float] | list[float],
    query: dict | None = None,
    *,
    datetime: DateRange | None = None,
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
    datetime: DateRange | None = None,
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
    datetime: DateRange | None = None,
    fully_contains_search_area: bool = True,
) -> list[ItemBase]: ...


def search(
    collection: str,
    extent: tuple[float, float, float, float] | list[float],
    query: dict | None = None,
    *,
    url: str = planetary_computer_url,
    datetime: DateRange | None = None,
    fully_contains_search_area: bool = True,
    cloud_cover_percent: float | None = None,
):
    xmin, ymin, xmax, ymax = extent
    search_extent = Extent(xmin, ymin, xmax, ymax)
    search_polygon = box(xmin, ymin, xmax, ymax)

    json = {
        "bbox": list(extent),
        "collections": [collection],
        "query": {"eo:cloud_cover": {"lt": cloud_cover_percent}} | (query or {})
        if cloud_cover_percent is not None
        else query or {},
    }

    if isinstance(datetime, tuple):
        datetime = "/".join(map(to_iso_string, datetime))

    if datetime and len(datetime) > 10:
        json["datetime"] = datetime

    response = requests.post(f"{url}/search", json=json)

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


def to_iso_string(t: str | datetime | None) -> str:
    if isinstance(t, str):
        return t
    if t is None:
        return ""
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


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
