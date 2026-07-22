import hashlib
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from shapely.geometry import Point

from glidergun import grid, search
from glidergun.geojson import FeatureCollection
from glidergun.gis import add_to_map
from glidergun.io import http_get


@pytest.fixture(scope="session")
def dem():
    return grid("./data/n55_e008_1arc_v3.bil").resample(0.01)


@pytest.fixture(scope="session")
def dem_color():
    return grid("./data/n55_e008_1arc_v3.bil").resample(0.01).color("terrain")


@pytest.fixture(scope="session")
def landsat():
    return search("landsat-c2-l2", [-122.52, 37.70, -122.35, 37.82])[0].get(["red", "green", "blue"], resample_by=10)


def save(obj, file_name):
    obj.save(file_name)
    with open(file_name, "rb") as f:
        hash = hashlib.md5(f.read()).hexdigest()
    with rasterio.open(file_name) as d:
        compress = d.profile.get("compress", None)
    shutil.rmtree("tests/output/temp")
    return hash, compress


def test_saving_dem_jpg(dem):
    hash, compress = save(dem, "tests/output/temp/dem.jpg")
    assert hash


def test_saving_dem_tif(dem):
    hash, compress = save(dem, "tests/output/temp/dem.tif")
    assert compress == "lzw"


def test_saving_dem_img(dem):
    hash, compress = save(dem, "tests/output/temp/dem.img")
    assert hash == "0834c56700cf1cc3b7155a8ef6e8b922"


def test_saving_dem_bil(dem):
    hash, compress = save(dem, "tests/output/temp/dem.bil")
    assert hash == "ce6230320c089d41ddbc8b3f17fd0c0d"


def test_saving_dem_color_jpg(dem_color):
    hash, compress = save(dem_color, "tests/output/temp/dem_color.jpg")
    assert hash


def test_saving_dem_color_tif(dem_color):
    hash, compress = save(dem_color, "tests/output/temp/dem_color.tif")
    assert compress == "lzw"


def test_saving_dem_color_img(dem_color):
    hash, compress = save(dem_color, "tests/output/temp/dem_color.img")
    assert hash == "0834c56700cf1cc3b7155a8ef6e8b922"


def test_saving_dem_color_bil(dem_color):
    hash, compress = save(dem_color, "tests/output/temp/dem_color.bil")
    assert hash == "ce6230320c089d41ddbc8b3f17fd0c0d"


def test_saving_landsat_jpg(landsat):
    hash, compress = save(landsat, "tests/output/temp/landsat.jpg")
    assert hash


def test_saving_landsat_tif(landsat):
    hash, compress = save(landsat, "tests/output/temp/landsat.tif")
    assert hash
    assert compress == "lzw"


def test_saving_landsat_img(landsat):
    hash, compress = save(landsat, "tests/output/temp/landsat.img")
    assert hash
    assert compress is None


def test_saving_landsat_bil(landsat):
    hash, compress = save(landsat, "tests/output/temp/landsat.bil")
    assert hash
    assert compress is None


def test_feature_collection_save_and_update(tmp_path):
    collection = FeatureCollection(((Point(1, 2), {"name": "a"}),))
    filtered = collection.filter(lambda feature: feature.properties["name"] == "a")
    assert len(filtered.features) == 1
    updated = filtered.update_geometry(lambda geometry: Point(geometry.x + 1, geometry.y))
    assert updated.features[0].geometry.x == 2

    bound = collection.bind(lambda feature: (feature,))
    assert len(bound.features) == 1

    text_file = tmp_path / "features.json"
    collection.save(str(text_file))
    assert text_file.exists()
    assert '"FeatureCollection"' in text_file.read_text()


def test_http_get_cache(tmp_path, monkeypatch):
    calls = {"count": 0}

    class Response:
        content = b"payload"

        def raise_for_status(self):
            return None

    def fake_get(url):
        calls["count"] += 1
        return Response()

    monkeypatch.setattr("glidergun.io.requests.get", fake_get)
    url = "https://example.com/data"
    assert http_get(url, str(tmp_path)) == b"payload"
    assert http_get(url, str(tmp_path)) == b"payload"
    assert calls["count"] == 1


def test_add_to_map_qgis(monkeypatch, tmp_path):
    g = grid(np.array([[1, 2], [3, 4]], dtype="float32"))

    class QgsProject:
        @staticmethod
        def instance():
            return SimpleNamespace(addMapLayer=lambda layer: setattr(g, "added", True))

    class QgsProcessingUtils:
        @staticmethod
        def tempFolder():  # noqa: N802 - matches QGIS API
            return str(tmp_path)

    class QgsRasterLayer:
        def __init__(self, *args):
            self.args = args

    qgis = SimpleNamespace(
        QgsProcessingUtils=QgsProcessingUtils,
        QgsProject=QgsProject,
        QgsRasterLayer=QgsRasterLayer,
        QgsVectorLayer=lambda *args: SimpleNamespace(args=args),
    )
    monkeypatch.setitem(sys.modules, "qgis.core", qgis)
    monkeypatch.setattr("glidergun.gis.safe_filename", lambda name: name)

    add_to_map(g, "test")
    assert (tmp_path / "test.tif").exists()


def test_add_to_map_without_gis(monkeypatch):
    monkeypatch.delitem(sys.modules, "arcpy", raising=False)
    monkeypatch.delitem(sys.modules, "qgis.core", raising=False)
    with pytest.raises(RuntimeError, match="ArcGIS Pro or QGIS"):
        add_to_map(grid(np.array([[1]], dtype="float32")), "test")
