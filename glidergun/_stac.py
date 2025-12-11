from datetime import datetime
from typing import Literal

import planetary_computer as pc
import requests
from rasterio.crs import CRS
from shapely.geometry import box, shape

from glidergun._grid import grid
from glidergun._stack import stack
from glidergun._types import Extent


class Feature(dict):
    def __init__(self, d: dict, extent: "Extent", crs: CRS, sign: bool):
        super().__init__(d)
        self.extent = extent
        self.crs = crs
        self.sign = sign

    @property
    def datetime(self) -> datetime:
        return datetime.fromisoformat(self["properties"]["datetime"])

    def get_url(self, name: str) -> str:
        assets = self["assets"]
        if name not in assets:
            raise ValueError(f"Asset '{name}' not found in item assets: {list(assets.keys())}")
        url = assets[name]["href"]
        return pc.sign(url) if self.sign else url

    def get_grid(self, name: str):
        return grid(self.get_url(name), self.extent, self.crs)

    def get_stack(self, *name: str):
        if len(name) == 1:
            return stack(self.get_url(name[0]), self.extent, self.crs)
        return stack([self.get_grid(n) for n in name])


def search(
    collection: Literal["sentinel-2-l2a", "landsat-c2-l2"] | str,
    extent: tuple[float, float, float, float],
    *,
    url: str = "https://planetarycomputer.microsoft.com/api/stac/v1/search",
    max_cloud_cover: float = 5.0,
    query: dict | None = None,
) -> list[Feature]:
    search_extent = Extent(*extent)
    search_polygon = box(*extent)
    sign = url == "https://planetarycomputer.microsoft.com/api/stac/v1/search"

    response = requests.post(
        url,
        json={
            "bbox": list(extent),
            "collections": [collection],
            "query": {"eo:cloud_cover": {"lt": max_cloud_cover}} | (query or {}),
        },
    )

    response.raise_for_status()

    features: list[Feature] = []

    for feature in response.json()["features"]:
        if search_polygon.within(shape(feature["geometry"])):
            crs = CRS.from_epsg(feature["properties"]["proj:epsg"])
            projected_extent = search_extent.project(from_crs=CRS.from_epsg(4326), to_crs=crs)
            features.append(Feature(feature, projected_extent, crs, sign))

    return features
