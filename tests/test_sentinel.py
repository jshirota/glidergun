import shutil

import planetary_computer as pc
from pystac_client import Client
from rasterio.crs import CRS

from glidergun import stack
from glidergun._types import Extent


def test_sentinel():
    client = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    search = client.search(
        collections=["sentinel-2-l2a"],
        bbox=[-75.72, 45.40, -75.67, 45.42],
        datetime="2025-09-01/2025-11-01",
        query={"eo:cloud_cover": {"lt": 5}},
        max_items=10,
    )

    collection = pc.sign_item_collection(search.get_all_items())
    i = collection.items[0]
    u = i.assets["visual"].href
    bbox = i.assets["visual"].to_dict()["proj:bbox"]
    e = Extent(*bbox).buffer(-50000)
    c = str(i.properties["proj:code"])
    s = stack(u, e, c)
    assert s.crs == CRS.from_string(c)
    assert s.extent == e
    assert len(s.grids) == 3
    assert s.dtype == "uint8"

    s.save("tests/output/temp/sentinel_test.tif")
    s2 = stack("tests/output/temp/sentinel_test.tif")
    assert s2.crs == CRS.from_string(c)
    assert s2.extent == e
    assert len(s2.grids) == 3
    assert s2.dtype == "uint8"

    shutil.rmtree("tests/output/temp")
