import os
import shutil

import numpy as np
import pytest
import rasterio

from glidergun._grid import Grid, con, grid
from glidergun._mosaic import mosaic

dem = grid("./.data/n55_e008_1arc_v3.bil")


def test_distance_1():
    g = (dem.resample(0.01) > 50).distance()
    assert g.round(4).md5 == "c263f513e2c3fc6c0cfc5c468ceb6b48"


def test_distance_2():
    g = dem.distance((8.2, 55.3))
    assert round(g.value(8.2, 55.3), 3) == 0.0
    # assert round(g.value(8.5, 55.3), 3) == 0.3
    # assert round(g.value(8.2, 55.7), 3) == 0.4
    # assert round(g.value(8.5, 55.7), 3) == 0.5


def test_aspect():
    g = dem.aspect()
    assert g.round(4).md5 == "e203f3540ab892ab9c69b386af1b47e9"


def test_bins():
    n = 77
    count = dem.bins.get(n, 0)
    points = list(dem.set_nan(dem != n).to_points())
    assert count == len(points)


def test_bins_2():
    n = 999
    count = dem.bins.get(n, 0)
    points = list(dem.set_nan(dem != n).to_points())
    assert count == len(points)


def test_bins_3():
    assert len(dem.slice(7).bins) == 7
    assert len(dem.set_nan(12).slice(7).bins) == 8


def test_boolean():
    g1 = dem > 20 and dem < 40
    g2 = 20 < dem < 40
    g3 = g1 and g2
    g4 = g1 * 100 - g3 * 100
    assert pytest.approx(g4.min, 0.001) == 0
    assert pytest.approx(g4.max, 0.001) == 0


def test_buffer():
    g1 = (dem.buffer(12, 1) == 12) ^ (dem.buffer(12, 1) == 12)
    g2 = (dem.buffer(12, 4) == 12) ^ (dem.buffer(12, 3) == 12)
    assert pytest.approx(g1.min, 0.001) == 0
    assert pytest.approx(g1.max, 0.001) == 0
    assert pytest.approx(g2.min, 0.001) == 0
    assert pytest.approx(g2.max, 0.001) == 1


def test_buffer_2():
    g1 = dem.resample(0.01).slice(5)
    g2 = g1.buffer(2, 1)
    g3 = g1.buffer(2, 0)
    g4 = g1.buffer(2, -1)
    g5 = g1.set_nan(~g4.is_nan(), g1)
    assert not g2.has_nan
    assert g3.md5 == g1.md5
    assert g4.has_nan
    assert pytest.approx(g5.min, 0.001) == 2
    assert pytest.approx(g5.max, 0.001) == 2


def test_buffer_3():
    g1 = dem.resample(0.01).slice(5)
    g1 = (g1.randomize() < 0.95) * g1
    g2 = g1.buffer(2, 1)
    g3 = g1.buffer(2, 0)
    g4 = g1.buffer(2, -1)
    g5 = g1.set_nan(~g4.is_nan(), g1)
    assert not g2.has_nan
    assert g3.md5 == g1.md5
    assert g4.has_nan
    assert pytest.approx(g5.min, 0.001) == 2
    assert pytest.approx(g5.max, 0.001) == 2


def test_clip():
    xmin, ymin, xmax, ymax = dem.extent
    extent = xmin + 0.02, ymin + 0.03, xmax - 0.04, ymax - 0.05
    for a, b in zip(dem.clip(*extent).extent, extent):
        assert pytest.approx(a, 0.001) == b


def test_clip_2():
    xmin, ymin, xmax, ymax = dem.extent
    extent = xmin - 0.02, ymin - 0.03, xmax + 0.04, ymax + 0.05
    for a, b in zip(dem.clip(*extent).extent, extent):
        assert pytest.approx(a, 0.001) == b


def test_con():
    g1 = con(dem > 50, 3, 7.123)
    g2 = con(30 <= dem < 40, 2.345, g1)
    assert pytest.approx(g2.min, 0.001) == 2.345
    assert pytest.approx(g2.max, 0.001) == 7.123


def test_fill_nan():
    g1 = dem.resample(0.02).set_nan(50)
    assert g1.has_nan
    g2 = g1.fill_nan()
    assert not g2.has_nan


def test_focal_count():
    g1 = dem.set_nan(dem < 5, con(dem > 50, 1, 2))
    g2 = g1.focal_count(1)
    g3 = g1.focal_count(2)
    g4 = g1.focal_count(3)
    assert pytest.approx(g2.min, 0.001) == 0
    assert pytest.approx(g2.max, 0.001) == 9
    assert pytest.approx(g3.min, 0.001) == 0
    assert pytest.approx(g3.max, 0.001) == 9
    assert pytest.approx(g4.min, 0.001) == 0
    assert pytest.approx(g4.max, 0.001) == 0


def test_focal_count_2():
    g1 = dem.set_nan(dem < 5, con(dem > 50, 1, 2))
    g2 = g1.focal_count(1, circle=True)
    g3 = g1.focal_count(2, circle=True)
    g4 = g1.focal_count(3, circle=True)
    assert pytest.approx(g2.min, 0.001) == 0
    assert pytest.approx(g2.max, 0.001) == 5
    assert pytest.approx(g3.min, 0.001) == 0
    assert pytest.approx(g3.max, 0.001) == 5
    assert pytest.approx(g4.min, 0.001) == 0
    assert pytest.approx(g4.max, 0.001) == 0


def test_focal_mean():
    g = dem.focal_mean()
    assert g.md5 == "ed6fb3c2c2423caeb347ba02196d78a7"


def test_focal_mean_2():
    g1 = dem.resample(0.02)
    g2 = g1.focal_mean()
    g3 = g1.focal_python(np.nanmean)
    assert g2.md5 == g3.md5


def test_focal_mean_3():
    g1 = dem.resample(0.02)
    g2 = g1.focal_mean(3, True, False)
    g3 = g1.focal_python(np.mean, 3, True, False)
    assert g2.md5 == g3.md5


def test_focal_mean_4():
    g1 = dem
    g2 = g1.focal_mean(2)
    assert g2.crs == g1.crs
    assert g2.extent == g1.extent
    assert g2.cell_size == g1.cell_size


def test_from_polygons():
    g1 = dem.set_nan(-1 < dem < 10, 123.456)
    g2 = dem.rasterize(g1.to_polygons())
    assert g2.has_nan
    assert pytest.approx(g2.min, 0.001) == 123.456
    assert pytest.approx(g2.max, 0.001) == 123.456


def test_gosper():
    def tick(grid: Grid):
        g = grid.focal_sum() - grid
        return (grid == 1) & (g == 2) | (g == 3)

    gosper = tick(grid(".data/glidergun30.txt"))
    md5s = set()
    while gosper.md5 not in md5s:
        md5s.add(gosper.md5)
        gosper = tick(gosper)
    assert len(md5s) == 60


def test_hillshade():
    g = dem.hillshade()
    assert g.round().md5 == "c8e3c319fe198b0d87456b98b9fe7532"


def test_interp_linear():
    points = [(-120, 50, 100), (-110, 50, 200), (-110, 40, 300), (-120, 40, 400)]
    extent = (-121, 39, -109, 51)
    g = grid(points, extent, 4326, 0.1)
    g = g.interp_linear()
    assert g.value(-100, 45) is np.nan
    assert 100 < g.value(-115, 45) < 400


def test_interp_nearest():
    points = [(-120, 50, 100), (-110, 50, 200), (-110, 40, 300), (-120, 40, 400)]
    extent = (-121, 39, -109, 51)
    g = grid(points, extent, 4326, 0.1)
    g = g.interp_nearest()
    assert g.value(-100, 45) is np.nan
    assert g.value(-112, 42) == 300


def test_interp_rbf():
    points = [(-120, 50, 100), (-110, 50, 200), (-110, 40, 300), (-120, 40, 400)]
    extent = (-121, 39, -109, 51)
    g = grid(points, extent, 4326, 0.1)
    g = g.interp_rbf()
    assert g.value(-100, 45) is np.nan
    assert 100 < g.value(-115, 45) < 400


def test_mosaic():
    g1 = grid("./.data/n55_e009_1arc_v3.bil")
    g2 = dem.mosaic(g1)
    assert g2.crs == dem.crs
    xmin, ymin, xmax, ymax = g2.extent
    assert pytest.approx(xmin, 0.001) == dem.xmin
    assert pytest.approx(ymin, 0.001) == min(dem.ymin, g1.ymin)
    assert pytest.approx(xmax, 0.001) == g1.xmax
    assert pytest.approx(ymax, 0.001) == max(dem.ymax, g1.ymax)


def test_op_mul():
    g = dem * 100
    assert pytest.approx(g.min, 0.001) == dem.min * 100
    assert pytest.approx(g.max, 0.001) == dem.max * 100
    assert g.md5 == "7d8dc93fa345e9929ebebe630f2e1de3"


def test_op_div():
    g = dem / 100
    assert pytest.approx(g.min, 0.001) == dem.min / 100
    assert pytest.approx(g.max, 0.001) == dem.max / 100
    assert g.md5 == "76f6a23611dbb51577003a6b4d157875"


def test_op_add():
    g = dem + 100
    assert pytest.approx(g.min, 0.001) == dem.min + 100
    assert pytest.approx(g.max, 0.001) == dem.max + 100
    assert g.md5 == "7307a641931470f902b299e1b4271ee4"


def test_op_sub():
    g = dem - 100
    assert pytest.approx(g.min, 0.001) == dem.min - 100
    assert pytest.approx(g.max, 0.001) == dem.max - 100
    assert g.md5 == "78cc4e4f5256715c1d37b0a7c0f54312"


def test_op_combined():
    g = 2 * dem - dem / 2 - dem / 4
    assert pytest.approx(g.min, 0.001) == 2 * dem.min - dem.min / 2 - dem.min / 4
    assert pytest.approx(g.max, 0.001) == 2 * dem.max - dem.max / 2 - dem.max / 4


def test_op_pow():
    g = (-dem) ** 2 - dem**2
    assert pytest.approx(g.min, 0.001) == 0
    assert pytest.approx(g.max, 0.001) == 0


def test_op_gt():
    g1 = con(dem > 20, 7, 11)
    g2 = g1 % 3
    assert pytest.approx(g2.min, 0.001) == 1
    assert pytest.approx(g2.max, 0.001) == 2


def test_op__floordiv():
    g = dem // 100
    assert pytest.approx(g.min, 0.001) == dem.min // 100
    assert pytest.approx(g.max, 0.001) == dem.max // 100
    assert g.md5 == "ee1906003e3b983d57f95ea60a059501"


def test_op_neg():
    g = -dem
    assert pytest.approx(g.min, 0.001) == -dem.max
    assert pytest.approx(g.max, 0.001) == -dem.min
    assert g.md5 == "1183b549226dd2858bcf1d62dd5202d1"


def test_op_pow_2():
    g1 = con(dem > 0, dem, 0) ** 2
    g2 = con(dem < 0, dem, 0) ** 2
    assert pytest.approx(g1.min, 0.001) == 0
    assert pytest.approx(g1.max, 0.001) == dem.max**2
    assert pytest.approx(g2.min, 0.001) == 0
    assert pytest.approx(g2.max, 0.001) == dem.min**2
    assert g1.md5 == "0c960de0b43e7b02e743303567f96ce5"
    assert g2.md5 == "10c9394f2b483d07834cdd6e8e9d7604"


def test_op_eq():
    g1 = dem == dem
    g2 = dem == dem * 1
    assert pytest.approx(g1.min, 0.001) == 1
    assert pytest.approx(g1.max, 0.001) == 1
    assert pytest.approx(g2.min, 0.001) == 1
    assert pytest.approx(g2.max, 0.001) == 1
    assert g1.md5 == "d698aba6245e1475c46436f1bb52f46e"
    assert g2.md5 == "d698aba6245e1475c46436f1bb52f46e"


def test_to_points():
    g = (dem.resample(0.01).randomize() < 0.01).set_nan(0).randomize()
    n = 0
    for x, y, value in g.to_points():
        n += 1
        assert g.value(x, y) == value
    assert n > 1000


def test_to_points_2():
    g1 = dem.resample(0.1).randomize()
    g2 = grid(g1.to_points(), g1.extent, g1.crs, g1.cell_size)
    assert g1.md5 == g2.md5


def test_to_points_3():
    g1 = dem.resample(0.01234).randomize()
    g2 = grid(g1.to_points(), g1.extent, g1.crs, g1.cell_size)
    assert g1.md5 == g2.md5


def test_to_stack():
    s = dem.to_stack()
    for g in s.grids:
        assert pytest.approx(g.min, 0.001) == 1
        assert pytest.approx(g.max, 0.001) == 254


def test_to_uint8_range():
    g1 = (dem * 100).to_uint8_range()
    assert pytest.approx(g1.min, 0.001) == 1
    assert pytest.approx(g1.max, 0.001) == 254
    g2 = ((dem.randomize() - 0.5) * 10000).to_uint8_range()
    assert pytest.approx(g2.min, 0.001) == 1
    assert pytest.approx(g2.max, 0.001) == 254


def test_project():
    g = dem.project(3857)
    assert g.crs.wkt.startswith('PROJCS["WGS 84 / Pseudo-Mercator",')


def test_properties():
    assert dem.width == 1801
    assert dem.height == 3601
    assert dem.dtype == "float32"
    assert dem.md5 == "09b41b3363bd79a87f28e3c5c4716724"


def test_ptp():
    g1 = dem.focal_ptp(4, True)
    g2 = dem.focal_max(4, True) - dem.focal_min(4, True)
    g3 = g2 - g1
    assert pytest.approx(g3.min, 0.001) == 0
    assert pytest.approx(g3.max, 0.001) == 0


def test_reclass():
    g = dem.reclass(
        (-9999, 10, 1),
        (10, 20, 2),
        (20, 20, 3),
        (30, 20, 4),
        (40, 20, 5),
        (50, 9999, 6),
    )
    assert pytest.approx(g.min, 0.001) == 1
    assert pytest.approx(g.max, 0.001) == 6
    values = set([0, 1, 2, 3, 4, 5, 6])
    for _, value in g.to_polygons():
        assert value in values


def test_resample():
    g = dem.resample(0.01)
    assert g.cell_size == (0.01, 0.01)


def test_resample_2():
    g = dem.resample((0.02, 0.03))
    assert g.cell_size == (0.02, 0.03)


def test_resample_3():
    g = dem.resample((0.07, 0.04))
    assert g.cell_size == (0.07, 0.04)


def test_set_nan():
    g1 = dem.set_nan(dem < 10, 123.456)
    g2 = con(g1.is_nan(), 234.567, -g1)
    assert pytest.approx(g1.min, 0.001) == 123.456
    assert pytest.approx(g1.max, 0.001) == 123.456
    assert pytest.approx(g2.min, 0.001) == -123.456
    assert pytest.approx(g2.max, 0.001) == 234.567


def test_slope():
    g = dem.slope()
    assert g.round(4).md5 == "4604605be36bfbf1ca83e7ab21002a10"


def test_sin():
    g = dem.sin()
    assert pytest.approx(g.min, 0.001) == -1
    assert pytest.approx(g.max, 0.001) == 1


def test_cos():
    g = dem.cos()
    assert pytest.approx(g.min, 0.001) == -1
    assert pytest.approx(g.max, 0.001) == 1


def test_tan():
    g = dem.tan()
    assert pytest.approx(g.min, 0.001) == -225.951
    assert pytest.approx(g.max, 0.001) == 225.951


def test_round():
    g = dem.resample(0.01).randomize()
    points = g.to_points()
    for p1, p2 in zip(points, g.round().to_points()):
        assert pytest.approx(p2[2], 0.000001) == round(p1[2])
    for p1, p2 in zip(points, g.round(3).to_points()):
        assert pytest.approx(p2[2], 0.000001) == round(p1[2], 3)


def test_zonal():
    zones = dem.slice(10)
    zone_min = dem.zonal_min(zones)
    zone_max = dem.zonal_max(zones)
    assert zone_min.set_nan(zones != 1).max < zone_max.set_nan(zones != 2).min
    assert zone_min.set_nan(zones != 2).max < zone_max.set_nan(zones != 3).min
    assert zone_min.set_nan(zones != 3).max < zone_max.set_nan(zones != 4).min
    assert zone_min.set_nan(zones != 4).max < zone_max.set_nan(zones != 5).min
    assert zone_min.set_nan(zones != 5).max < zone_max.set_nan(zones != 6).min
    assert zone_min.set_nan(zones != 6).max < zone_max.set_nan(zones != 7).min
    assert zone_min.set_nan(zones != 7).max < zone_max.set_nan(zones != 8).min
    assert zone_min.set_nan(zones != 8).max < zone_max.set_nan(zones != 9).min
    assert zone_min.set_nan(zones != 9).max < zone_max.set_nan(zones != 10).min


def save(g1: Grid, file: str, strict: bool = True):
    folder = ".output/test"
    file_path = f"{folder}/{file}"
    os.makedirs(folder, exist_ok=True)
    g1.save(file_path)
    g2 = grid(file_path)
    if strict:
        assert g2.md5 == g1.md5
    assert g2.extent == g1.extent
    shutil.rmtree(folder)


def test_save_memory():
    memory_file = rasterio.MemoryFile()
    dem.save(memory_file)
    g = grid(memory_file)
    assert g.md5 == dem.md5


def test_save_bil():
    save(dem, "test_grid.bil")


def test_save_bt():
    save(dem, "test_grid.bt")


def test_save_img():
    save(dem, "test_grid.img")


def test_save_tif():
    save(dem, "test_grid.tif")


def test_save_jpg():
    save(dem, "test_grid.jpg", strict=False)


def test_save_png():
    save(dem, "test_grid.png", strict=False)


def test_mosaic_dataset():
    m = mosaic(
        "./.data/n55_e008_1arc_v3.bil",
        "./.data/n55_e009_1arc_v3.bil",
    )
    assert m.clip(8, 55, 9, 56).extent == pytest.approx((8, 55, 9, 56), 0.001)
    assert m.clip(8, 55, 10, 56).extent == pytest.approx((8, 55, 10, 56), 0.001)
    assert m.clip(8.2, 55.2, 9.2, 56.2).extent == pytest.approx(
        (8.2, 55.2, 9.2, 56.0), 0.001
    )
    assert m.clip(7.5, 55, 10, 56).extent == pytest.approx((8, 55, 10, 56), 0.001)
    assert m.clip(8, 50, 10, 56).extent == pytest.approx((8, 55, 10, 56), 0.001)
    assert m.clip(8, 55, 15, 56).extent == pytest.approx((8, 55, 10, 56), 0.001)
    assert m.clip(8, 55, 10, 60).extent == pytest.approx((8, 55, 10, 56), 0.001)


def test_mosaic_eager_vs_lazy():
    g = mosaic(
        grid("./.data/n55_e008_1arc_v3.bil"), grid("./.data/n55_e009_1arc_v3.bil")
    )
    m = mosaic("./.data/n55_e008_1arc_v3.bil", "./.data/n55_e009_1arc_v3.bil")

    assert g.clip(8, 55, 8.5, 56).md5 == m.clip(8, 55, 8.5, 56).md5
