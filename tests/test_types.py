import sys
from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib import pyplot as plt
from rasterio.crs import CRS
from rasterio.transform import Affine

from glidergun.types import CellSize, Chart, Extent, GridCore, PointValue


def test_gridcore():
    data = np.array([[1, 2], [3, 4]])
    transform = Affine.translation(0, 0)
    crs = CRS.from_epsg(4326)
    grid_core = GridCore(data, transform, crs)

    assert np.array_equal(grid_core.data, data)
    assert grid_core.transform == transform
    assert grid_core.crs == crs


def test_extent_intersect():
    extent1 = Extent(0, 0, 10, 10, CRS.from_epsg(3857))
    extent2 = Extent(5, 5, 15, 15, CRS.from_epsg(3857))
    expected = Extent(5, 5, 10, 10, CRS.from_epsg(3857))

    assert extent1.intersect(extent2) == expected


def test_extent_union():
    extent1 = Extent(0, 0, 10, 10, CRS.from_epsg(3857))
    extent2 = Extent(5, 5, 15, 15, CRS.from_epsg(3857))
    expected = Extent(0, 0, 15, 15, CRS.from_epsg(3857))

    assert extent1.union(extent2) == expected


def test_cellsize_mul():
    cell_size = CellSize(2, 3)
    expected = CellSize(4, 6)

    assert cell_size * 2 == expected
    assert 2 * cell_size == expected


def test_cellsize_truediv():
    cell_size = CellSize(4, 6)
    expected = CellSize(2, 3)

    assert cell_size / 2 == expected


def test_pointvalue():
    point_value = PointValue(1, 2, 3)

    assert point_value.x == 1
    assert point_value.y == 2
    assert point_value.value == 3
    assert repr(point_value) == "(1, 2, 3)"


def test_extent_methods():
    extent = Extent(0, 0, 10, 10, 4326)

    assert tuple(extent) == (0.0, 0.0, 10.0, 10.0)
    assert len(extent) == 4
    assert extent[0] == 0
    assert extent.width == 10
    assert extent.height == 10
    assert extent.center == (5, 5)
    assert extent.is_valid
    assert not Extent(0, 0, 0, 0).is_valid
    assert extent.contains((2, 2, 8, 8))
    assert extent.intersects((5, 5, 15, 15))
    assert extent.to_polygon().area == 100
    assert repr(extent) == "(0, 0, 10, 10)"
    assert not extent.contains((2, 2, 11, 11))
    assert not extent.intersects((10, 10, 11, 11))


def test_extent_transform_operations():
    extent = Extent(0, 0, 10, 10, 4326)

    assert extent.project(3857).crs.to_epsg() == 3857
    assert len(extent.tiles(4, 4)) == 9
    assert extent.adjust(1, 1, -1, -1) == Extent(1, 1, 9, 9, 4326)
    assert extent.buffer(1) == Extent(-1, -1, 11, 11, 4326)
    assert extent.buffer_by(0.5) == Extent(2.5, 2.5, 7.5, 7.5, 4326)

    with pytest.raises(ValueError):
        extent.adjust(6, 6, -6, -6)


def test_extent_active_map_without_gis(monkeypatch):
    monkeypatch.delitem(sys.modules, "arcpy", raising=False)
    monkeypatch.delitem(sys.modules, "qgis.core", raising=False)

    with pytest.raises(RuntimeError, match="only supported"):
        Extent.from_active_map()


def test_extent_from_arcgis_active_map(monkeypatch):
    spatial_reference = SimpleNamespace(exportToString=lambda: "EPSG:4326")
    map_extent = SimpleNamespace(XMin=1, YMin=2, XMax=3, YMax=4, spatialReference=spatial_reference)
    active_view = SimpleNamespace(camera=SimpleNamespace(getExtent=lambda: map_extent))
    arcpy = SimpleNamespace(mp=SimpleNamespace(ArcGISProject=lambda _: SimpleNamespace(activeView=active_view)))
    monkeypatch.setitem(sys.modules, "arcpy", arcpy)

    assert Extent.from_active_map() == Extent(1, 2, 3, 4, 4326)


def test_extent_from_qgis_active_map(monkeypatch):
    map_extent = SimpleNamespace(
        xMinimum=lambda: 1,
        yMinimum=lambda: 2,
        xMaximum=lambda: 3,
        yMaximum=lambda: 4,
    )
    destination_crs = SimpleNamespace(toWkt=lambda: "EPSG:4326")
    canvas = SimpleNamespace(
        extent=lambda: map_extent,
        mapSettings=lambda: SimpleNamespace(destinationCrs=lambda: destination_crs),
    )
    qgis_utils = SimpleNamespace(iface=SimpleNamespace(mapCanvas=lambda: canvas))
    monkeypatch.setitem(sys.modules, "qgis.core", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "qgis.utils", qgis_utils)

    assert Extent.from_active_map() == Extent(1, 2, 3, 4, 4326)


def test_cellsize_unsupported_multiplication():
    cell_size = CellSize(2, 3)

    with pytest.raises(TypeError):
        cell_size * "2"
    with pytest.raises(TypeError):
        "2" * cell_size
    assert repr(cell_size) == "(2, 3)"


def test_chart_png_representation():
    figure, axes = plt.subplots()
    axes.plot([0, 1], [0, 1])

    result = Chart(figure, axes)._repr_png_()

    assert result.startswith(b"\x89PNG\r\n\x1a\n")
