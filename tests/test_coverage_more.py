import numpy as np

from glidergun import grid, stack
from glidergun.display import animate, get_html
from glidergun.plot import create_animation, create_histogram, create_thumbnail
from glidergun.stac import ItemBase, search, to_iso_string
from glidergun.types import Chart


def test_plot_helpers():
    data = np.array([[0, 1], [1, 0]], dtype=np.float32)
    assert create_thumbnail(data)
    histogram = create_histogram(({0.0: 1, 1.0: 2}, "blue"))
    assert isinstance(histogram, Chart)
    assert histogram._repr_png_()

    animation = create_animation([data, data], cmap="gray", interval=10)
    assert animation._fig is not None


def test_display_helpers():
    g = grid(np.array([[1, 2], [3, 4]], dtype=np.float32), extent=(0, 0, 2, 2))
    html = get_html(g)
    assert "<div>" in html
    assert "image:" in html
    assert animate([g], interval=10).event_source is not None


def test_grid_geojson_overlay_draw_and_georeference():
    g = grid(np.array([[1, np.nan], [3, 4]], dtype=np.float32), extent=(0, 0, 2, 2))

    points = g.to_geojson(polygonize=False)
    assert len(points.features) == 3
    polygons = g.to_geojson(polygonize=True)
    assert polygons.features
    assert g.data_extent.is_valid

    overlaid = g.overlay(
        grid(np.array([[10, 10], [10, 10]], dtype=np.float32), extent=(0, 0, 2, 2)),
        0.5,
    )
    assert overlaid.width == g.width
    drawn = g.draw(points.features[0].geometry.buffer(0.1), 7)
    assert drawn.width == g.width

    georeferenced = g.georeference((0, 0, 4, 4), crs=3857)
    assert georeferenced.crs.to_epsg() == 3857
    assert georeferenced.width == 2


def test_stack_extended_methods():
    g1 = grid(np.array([[1, 2], [3, 4]], dtype=np.uint8), extent=(0, 0, 2, 2))
    g2 = grid(np.array([[5, 6], [7, 8]], dtype=np.uint8), extent=(0, 0, 2, 2))
    s = stack([g1, g2])

    assert s.georeference((0, 0, 2, 2)).width == 2
    assert s.clip((0, 0, 1, 1)).width == 1
    assert s.clip_at(0.5, 0.5, 1, 1).width == 1
    assert s.extract_bands(2).grids[0].data.tolist() == g2.data.tolist()
    assert s.pca(1).grids
    assert s.pca().width == 2
    assert s.project(3857).crs.to_epsg() == 3857
    assert s.resample_by(2).width == 1
    assert s.canny().width == 2
    assert s.gaussian_filter(1).width == 2
    assert s.sobel().width == 2
    assert s.zip_with(s, lambda a, b: a + b).grids
    assert s.overlay(g1).grids


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("HTTP error")

    def json(self):
        return self.payload


def test_stac_search_and_iso_helpers(monkeypatch):
    feature = {
        "type": "Feature",
        "stac_version": "1.0.0",
        "id": "item-1",
        "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]]},
        "bbox": [0, 0, 2, 2],
        "properties": {
            "datetime": "2020-01-01T00:00:00Z",
            "proj:epsg": 4326,
        },
        "assets": {},
        "links": [],
        "collection": "test",
    }

    request_payloads = []

    def fake_post(*args, **kwargs):
        request_payloads.append(kwargs["json"])
        return FakeResponse({"features": [feature]})

    monkeypatch.setattr("glidergun.stac.requests.post", fake_post)
    items = search("sentinel", (0, 0, 1, 1), url="https://example.test/stac")
    assert len(items) == 1
    assert isinstance(items[0], ItemBase)
    assert items[0].fully_contains_search_extent is True
    assert items[0].crs.to_epsg() == 4326
    assert "query" not in request_payloads[-1]

    feature["properties"] = {
        "datetime": "2020-01-01T00:00:00Z",
        "proj:code": "EPSG:32618",
    }
    items = search("sentinel", (0, 0, 1, 1), url="https://example.test/stac")
    assert items[0].crs.to_epsg() == 32618

    search(
        "sentinel",
        (0, 0, 1, 1),
        {"test:field": {"eq": "value"}},
        url="https://example.test/stac",
        max_cloud_cover_percent=5,
    )
    assert request_payloads[-1]["query"] == {
        "eo:cloud_cover": {"lt": 5},
        "test:field": {"eq": "value"},
    }

    assert to_iso_string("2020-01-01") == "2020-01-01"
    assert to_iso_string(None) == ""
    assert to_iso_string("2020-01-01T00:00:00") == "2020-01-01T00:00:00"
