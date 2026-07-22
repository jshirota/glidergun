import importlib

import numpy as np
import pytest
import rasterio
from affine import Affine as AffineTransform
from rasterio.crs import CRS
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from shapely.geometry import Point
from sklearn.preprocessing import StandardScaler

import glidergun.loftr as loftr
from glidergun import Grid, grid
from glidergun.loftr import Homography

grid_module = importlib.import_module("glidergun.grid")


def test_empty_grid_regression():
    with pytest.raises(ValueError, match="Empty raster"):
        Grid(np.array([[]]), rasterio.Affine.identity(), CRS.from_epsg(4326))
    with pytest.raises(ValueError, match="Empty raster"):
        grid(np.array([[]]))


def test_reverse_operators_and_comparison_methods():
    g = grid(np.array([[2, 4], [6, 8]], dtype=np.float32), extent=(0, 0, 2, 2))
    assert (2 ** g).data.tolist()[0] == [4.0, 16.0]
    assert (16 / g).data.tolist()[0] == [8.0, 4.0]
    assert (16 // g).data.tolist()[0] == [8.0, 4.0]
    assert (10 % g).data.tolist()[0] == [0.0, 2.0]
    assert (g == object()) is False
    assert (g != object()) is True
    assert g.is_greater_than(3).data[1, 0]
    assert g.is_less_than(5).data[0, 1]
    assert g.is_greater_than_or_equal(4).data[0, 1]
    assert g.is_less_than_or_equal(4).data[0, 1]
    assert g.is_equal(4).data[0, 1]
    assert g.is_not_equal(4).data[0, 0]


def test_grid_math_and_aspect():
    g = grid(np.array([[1, 4], [9, 16]], dtype=np.float32), extent=(0, 0, 2, 2))
    assert g.abs().min == 1
    assert g.sin().width == 2
    assert g.cos().width == 2
    assert g.tan().width == 2
    assert g.arcsin().width == 2
    assert g.arccos().width == 2
    assert g.arctan().width == 2
    assert g.log().width == 2
    assert g.log(10).width == 2
    assert g.round(0).width == 2
    assert g.aspect().width == 2
    assert g.aspect(True).width == 2


def test_mosaic_blending_calibration_and_no_overlap():
    g1 = grid(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), extent=(0, 0, 2, 2))
    g2 = grid(np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32), extent=(0, 0, 2, 2))
    assert g1.mosaic(g2, blend=True).width == 2
    calibrated = g1.calibrate(g2)
    assert calibrated.width == 2

    left = grid(np.array([[1.0, 2.0]], dtype=np.float32), extent=(0, 0, 2, 1))
    right = grid(np.array([[8.0, 9.0]], dtype=np.float32), extent=(10, 0, 12, 1))
    assert left.mosaic(right, blend=True).width > 0


def test_resizing_buffer_heat_distance_and_process_tiles():
    g = grid(np.array([[1, 1, 0], [1, 1, 0]], dtype=np.uint8), extent=(0, 0, 3, 2))
    assert g.resize(6, 4).width == 6
    assert g.buffer(1, 1).width == 3
    assert g.buffer(1, -1).width == 3
    assert g.heat_map(1).width == 3
    assert g.distance().width == 3
    assert g.canny(threshold1=20, threshold2=40).width == 3

    unequal = grid(np.array([[0, 1, 0], [0, 1, 0]], dtype=np.float32), extent=(0, 0, 6, 2))
    assert unequal.distance().width == 3

    large = grid(np.array([[i] * 40 for i in range(40)], dtype=np.float32), extent=(0, 0, 40, 40))
    threaded = large.process_tiles(lambda tile: tile, tile_size=5, buffer=1, max_workers=2)
    assert threaded.width == 40


def test_overlay_draw_points_and_percentile_stretch_branches():
    g = grid(np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32), extent=(0, 0, 2, 2))
    s = g.to_stack()
    assert isinstance(g.overlay(s), type(s))
    assert g.draw(Point(0.5, 0.5), None).width == 2
    assert len(g.to_points(include_nan=True)) == 4
    assert len(g._coords_with_values(include_nan=True)) == 4
    assert g.percent_clip(0, 100).width == 2
    assert grid(np.ones((2, 2), dtype=np.float32)).stretch(0, 1).min == 0.5
    assert grid(np.ones((2, 2), dtype=np.float32)).percent_clip(0, 100).width == 2


def test_scaling_clustering_and_save_hooks(monkeypatch, tmp_path):
    g = grid(np.array([[1, 2], [3, 4]], dtype=np.float32), extent=(0, 0, 2, 2))
    assert g.kmeans_cluster(2, random_state=0).width == 2
    assert g.scale().width == 2
    assert g.scale(StandardScaler()).width == 2
    assert g.type("uint16", nan_to_num=0).width == 2
    assert g.color("terrain").display == "terrain"
    assert g.to_array().shape == (1, 2, 2)

    memory = MemoryFile()
    g.save(memory, "float32", -9999, "GTiff")
    assert memory.read()
    memory.close()

    png = tmp_path / "grid.png"
    g.save(str(png))
    assert png.exists()

    monkeypatch.setattr(grid_module, "sam", lambda *args, **kwargs: "sam-result", raising=False)
    monkeypatch.setattr("glidergun.sam.sam", lambda *args, **kwargs: "sam-result")
    assert g.sam("object") == "sam-result"

    monkeypatch.setattr(grid_module, "add_to_map", lambda *args: None)
    g.add_to_map("test")


def test_homography_branch(monkeypatch):
    g = grid(np.array([[1, 2], [3, 4]], dtype=np.float32), extent=(0, 0, 2, 2))
    h = Homography(None, 0, 0, AffineTransform.identity(), CRS.from_epsg(4326), 2, 2)
    captured = {}

    def fake_reference(source, reference, **kwargs):
        captured["called"] = True
        return source

    monkeypatch.setattr(loftr, "georeference_to_reference", fake_reference)
    assert g.georeference_to(h) is g
    assert captured["called"]


def test_constructor_error_and_raster_branches(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="Extent is required"):
        grid([1, 2, 3])
    with pytest.raises(ValueError, match="Cell size is required"):
        grid([1, 2, 3], extent=(0, 0, 1, 1))
    with pytest.raises(ValueError, match="Affine transform"):
        grid("not-a-raster", extent=from_origin(0, 1, 1, 1), crs=4326)

    g = grid(np.array([[1, 2], [3, 4]], dtype=np.uint8), extent=(0, 0, 2, 2))
    g.save(str(tmp_path / "bytes.tif"))
    data = (tmp_path / "bytes.tif").read_bytes()
    assert grid(data).width == 2
    with rasterio.MemoryFile(data) as memory:
        assert grid(memory).width == 2

    with pytest.raises(ValueError, match="Extent is required"):
        grid("alos-dem")
    with pytest.raises(ValueError, match="too large"):
        grid("alos-dem", extent=(0, 0, 100, 100))

    monkeypatch.setattr(grid_module, "from_dem", lambda *args, **kwargs: g)
    assert grid("alos-dem", extent=(0, 0, 1, 1)) is g

    monkeypatch.setattr(
        "glidergun.types.Extent.from_active_map",
        staticmethod(lambda: grid_module.Extent(0, 0, 1, 1, 4326)),
    )
    assert grid(np.array([[1]], dtype=np.float32), extent="#").width == 1


def test_dataset_resample_and_from_ndarray():
    g = grid(np.array([[1, 2], [3, 4]], dtype=np.float32), extent=(0, 0, 2, 2))
    with rasterio.MemoryFile() as memory:
        with memory.open(
            driver="GTiff",
            height=2,
            width=2,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=from_origin(0, 2, 1, 1),
        ) as dataset:
            dataset.write(g.data, 1)
        with memory.open() as dataset:
            assert grid(dataset, resample_by=2).width == 1

    direct = grid_module.from_ndarray(
        np.array([[1, 2]], dtype=np.float32),
        from_origin(0, 1, 1, 1),
        4326,
    )
    assert direct.width == 2


def test_idw_empty_radius_and_aggregation():
    g = grid(np.array([[1, 2], [3, 4]], dtype=np.float32), extent=(0, 0, 2, 2))
    empty = g.interp_idw([])
    assert empty.width == 2
    radius = g.interp_idw([(0.5, 0.5, 10)], radius=1)
    assert radius.width == 2

    g2 = grid(np.array([[5, 6], [7, 8]], dtype=np.float32), extent=(0, 0, 2, 2))
    assert grid_module.mean(g, g2).width == 2
    assert grid_module.std(g, g2).width == 2
    assert grid_module.minimum(g, g2).width == 2
    assert grid_module.maximum(g, g2).width == 2


def test_remaining_grid_branches(monkeypatch):
    g = grid(np.array([[1, 2], [3, 4]], dtype=np.float32), extent=(0, 0, 2, 2))
    assert g.extent_4326.crs.to_epsg() == 4326

    large = grid(np.arange(100).reshape(10, 10), extent=(0, 0, 10, 10))
    with pytest.raises(ValueError, match="Resulting mosaic is empty"):
        large.process_tiles(lambda tile: None, tile_size=1)

    monkeypatch.setattr(
        "glidergun.types.Extent.from_active_map",
        staticmethod(lambda: (_ for _ in ()).throw(RuntimeError("no GIS"))),
    )
    with pytest.raises(ValueError, match="ArcGIS Pro or QGIS"):
        grid(np.array([[1]], dtype=np.float32), extent="#")

    monkeypatch.setattr(
        "glidergun.stac.search_mosaic",
        lambda *args, **kwargs: g,
    )
    assert grid_module.from_dem("alos-dem", (0, 0, 1, 1), 10) is g

    g3857 = g.project(3857)
    with pytest.raises(ValueError, match="same CRS"):
        grid_module.standardize(g, g3857)
    assert grid_module.standardize(g, g, cell_size=2)[0].cell_size == (2.0, 2.0)
    assert grid_module.standardize(g, g, cell_size=(2, 3))[0].cell_size == (2.0, 3.0)
    assert grid_module.standardize(g, g, extent=g.extent)[0].extent == g.extent

    no_points_in_radius = g.interp_idw([(100, 100, 10)], radius=1)
    assert no_points_in_radius.width == 2
