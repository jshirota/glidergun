import os
import shutil

import numpy as np
import pytest
import rasterio

from glidergun import Grid, con, grid, mosaic
from glidergun._grid import _to_uint8_range

dem = grid("./tests/input/n55_e008_1arc_v3.bil")


def test_aspect():
    g = dem.aspect(True)
    assert g.round(4).sha256 == "44b804cbe617d622afa160589c5ccb685517c3fa9e3f036d2c7126fb6ec30d5f"


def test_bins():
    n = 77
    count = dem.bins.get(n, 0)
    points = dem.set_nan(dem != n).to_points()
    assert count == len(points)


def test_bins_2():
    n = 999
    count = dem.bins.get(n, 0)
    points = dem.set_nan(dem != n).to_points()
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
    assert g3.sha256 == g1.sha256
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
    assert g3.sha256 == g1.sha256
    assert g4.has_nan
    assert pytest.approx(g5.min, 0.001) == 2
    assert pytest.approx(g5.max, 0.001) == 2


def test_buffer_4():
    g = dem.resample(0.01).randomize() > 0.9
    assert len(g.bins) == 2
    assert g.type("uint8").buffer(1, 3).dtype == "uint8"
    assert g.type("uint8").buffer(1, 0).dtype == "uint8"
    assert g.type("int32").buffer(1, 3).dtype == "int32"
    assert g.type("int32").buffer(1, 0).dtype == "int32"
    assert g.type("float32").buffer(1, 3).dtype == "float32"
    assert g.type("float32").buffer(1, 0).dtype == "float32"


def test_clip():
    xmin, ymin, xmax, ymax = dem.extent
    extent = xmin + 0.02, ymin + 0.03, xmax - 0.04, ymax - 0.05
    for a, b in zip(dem.clip(extent).extent, extent, strict=False):
        assert pytest.approx(a, 0.001) == b


def test_clip_2():
    xmin, ymin, xmax, ymax = dem.extent
    extent = xmin - 0.02, ymin - 0.03, xmax + 0.04, ymax + 0.05
    for a, b in zip(dem.clip(extent).extent, extent, strict=False):
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
    assert g.sha256 == "f110551123e368bc61b3a1ed757341b532d66dc1394bf89432a8cde62f3341d3"


def test_focal_mean_2():
    g1 = dem.resample(0.02)
    g2 = g1.focal_mean()
    g3 = g1.focal_generic(np.nanmean)
    assert g2.sha256 == g3.sha256


def test_focal_mean_3():
    g1 = dem.resample(0.02)
    g2 = g1.focal_mean(3, True, False)
    g3 = g1.focal_generic(np.mean, 3, True, False)
    assert g2.sha256 == g3.sha256


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


def test_hillshade():
    g = dem.hillshade()
    assert g.round().sha256 == "026831fdf7236580d855d55664ce58c84a1589fe7fcb907483d29c4fd4874519"


def test_interp_linear():
    points = [(-120, 50, 100), (-110, 50, 200), (-110, 40, 300), (-120, 40, 400)]
    extent = (-121, 39, -109, 51)
    g = grid(points, extent, 4326, 0.1)
    g = g.interp_linear()
    assert g.value_at(-100, 45) is np.nan
    assert 100 < g.value_at(-115, 45) < 400


def test_interp_nearest():
    points = [(-120, 50, 100), (-110, 50, 200), (-110, 40, 300), (-120, 40, 400)]
    extent = (-121, 39, -109, 51)
    g = grid(points, extent, 4326, 0.1)
    g = g.interp_nearest()
    assert g.value_at(-100, 45) is np.nan
    assert g.value_at(-112, 42) == 300


def test_interp_rbf():
    points = [(-120, 50, 100), (-110, 50, 200), (-110, 40, 300), (-120, 40, 400)]
    extent = (-121, 39, -109, 51)
    g = grid(points, extent, 4326, 0.1)
    g = g.interp_rbf()
    assert g.value_at(-100, 45) is np.nan
    assert 100 < g.value_at(-115, 45) < 400


def test_mosaic():
    g1 = grid("./tests/input/n55_e009_1arc_v3.bil")
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
    assert g.sha256 == "c397703b13b955de8b8a9b01bd7bbd85e67d0c9c36290bf28e01a92a60b9343c"


def test_op_div():
    g = dem / 100
    assert pytest.approx(g.min, 0.001) == dem.min / 100
    assert pytest.approx(g.max, 0.001) == dem.max / 100
    assert g.sha256 == "f96225558e2daa17250aca7dd42d4fd8770da4bcf9c53b49d25365b7688c3a16"


def test_op_add():
    g = dem + 100
    assert pytest.approx(g.min, 0.001) == dem.min + 100
    assert pytest.approx(g.max, 0.001) == dem.max + 100
    assert g.sha256 == "1de1fc6df6036bf3e21bae6ffa787bbeb8e53480ebb48b10bad439afa48ac91e"


def test_op_sub():
    g = dem - 100
    assert pytest.approx(g.min, 0.001) == dem.min - 100
    assert pytest.approx(g.max, 0.001) == dem.max - 100
    assert g.sha256 == "cd7f7359c55df230dfcf60fec97ac56f1cd870699b6b94e117c8e759a6753ed0"


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
    assert g.sha256 == "2f5fe97ac43559b606d67bb90ff037f1f2088a24586a4a0692d218df95d384b7"


def test_op_neg():
    g = -dem
    assert pytest.approx(g.min, 0.001) == -dem.max
    assert pytest.approx(g.max, 0.001) == -dem.min
    assert g.sha256 == "16e092bfac3a53eba37f8859096a92b85a6745c07ac831f7ae77943481ec1cd3"


def test_op_pow_2():
    g1 = con(dem > 0, dem, 0) ** 2
    g2 = con(dem < 0, dem, 0) ** 2
    assert pytest.approx(g1.min, 0.001) == 0
    assert pytest.approx(g1.max, 0.001) == dem.max**2
    assert pytest.approx(g2.min, 0.001) == 0
    assert pytest.approx(g2.max, 0.001) == dem.min**2
    assert g1.sha256 == "13ae32f3cebfdc803fb25572acfda258920236c22ba3f39469117cfc7276063d"
    assert g2.sha256 == "4f061de3459ba7f6dbd8801e50d28fbf43ed8e91b91e2a814f22d94642c7e649"


def test_op_eq():
    g1 = dem == dem
    g2 = dem == dem * 1
    assert pytest.approx(g1.min, 0.001) == 1
    assert pytest.approx(g1.max, 0.001) == 1
    assert pytest.approx(g2.min, 0.001) == 1
    assert pytest.approx(g2.max, 0.001) == 1
    assert g1.sha256 == "1ee570109149867b9908cd549a9b2137ad9b6e115534cfae23089630375f7ba7"
    assert g2.sha256 == "1ee570109149867b9908cd549a9b2137ad9b6e115534cfae23089630375f7ba7"


def test_to_points():
    g = (dem.resample(0.01).randomize() < 0.01).set_nan(0).randomize()
    n = 0
    for x, y, value in g.to_points():
        n += 1
        assert g.value_at(x, y) == value
    assert n > 1000


def test_to_points_2():
    g1 = dem.resample(0.1).randomize()
    g2 = grid(g1.to_points(), g1.extent, g1.crs, g1.cell_size)
    assert g1.sha256 == g2.sha256


def test_to_points_3():
    g1 = dem.resample(0.01234).randomize()
    g2 = grid(g1.to_points(), g1.extent, g1.crs, g1.cell_size)
    assert g1.sha256 == g2.sha256


def test_to_stack():
    s = dem.to_stack()
    for g in s.grids:
        assert pytest.approx(g.min, 0.001) == 1
        assert pytest.approx(g.max, 0.001) == 254


def test_to_uint8_range():
    g1 = _to_uint8_range(dem * 100)
    assert pytest.approx(g1.min, 0.001) == 0
    assert pytest.approx(g1.max, 0.001) == 255
    g2 = _to_uint8_range((dem.randomize() - 0.5) * 10000)
    assert pytest.approx(g2.min, 0.001) == 0
    assert pytest.approx(g2.max, 0.001) == 255


def test_project():
    g = dem.project(3857)
    assert g.crs.wkt.startswith('PROJCS["WGS 84 / Pseudo-Mercator",')


def test_project_2():
    g0 = dem.type("int32")
    g1 = g0.project(3857)
    g2 = g0.project(4326)
    assert g0.dtype == "int32"
    assert g1.dtype == "float32"
    assert g2.dtype == "float32"


def test_properties():
    assert dem.width == 1801
    assert dem.height == 3601
    assert dem.dtype == "float32"
    assert dem.sha256 == "58ba80fc92d5a3f11e0bd366bedd2bf5d3b620a9a56b0312a26e606e2b9b761d"


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
    values = {0, 1, 2, 3, 4, 5, 6}
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


def test_resample_4():
    g0 = dem.type("int32")
    g1 = g0.resample_by(10.0)
    g2 = g0.resample_by(1.0)
    assert g0.dtype == "int32"
    assert g1.dtype == "float32"
    assert g2.dtype == "float32"


def test_set_nan():
    g1 = dem.set_nan(dem < 10, 123.456)
    g2 = con(g1.is_nan(), 234.567, -g1)
    assert pytest.approx(g1.min, 0.001) == 123.456
    assert pytest.approx(g1.max, 0.001) == 123.456
    assert pytest.approx(g2.min, 0.001) == -123.456
    assert pytest.approx(g2.max, 0.001) == 234.567


def test_slope():
    g = dem.slope(True)
    assert g.round(4).sha256 == "5c467561e8b8b85dff2efe2d68bb96a3924586b612e79cc2366685fc9786df7b"


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
    for p1, p2 in zip(points, g.round().to_points(), strict=False):
        assert pytest.approx(p2[2], 0.01) == round(p1[2])
    for p1, p2 in zip(points, g.round(3).to_points(), strict=False):
        assert pytest.approx(p2[2], 0.01) == round(p1[2], 3)


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
    folder = "tests/output/temp1"
    file_path = f"{folder}/{file}"
    os.makedirs(folder, exist_ok=True)
    g1.save(file_path)
    g2 = grid(file_path)
    if strict:
        assert g2.sha256 == g1.sha256
    assert g2.extent == g1.extent
    shutil.rmtree(folder)


def test_save_memory():
    memory_file = rasterio.MemoryFile()
    dem.save(memory_file)
    g = grid(memory_file)
    assert g.sha256 == dem.sha256


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
        "./tests/input/n55_e008_1arc_v3.bil",
        "./tests/input/n55_e009_1arc_v3.bil",
    )

    e = m.extent
    assert e == pytest.approx((8.0, 55.0, 10.0, 56.0), 0.001)

    def clip(xmin, ymin, xmax, ymax) -> Grid:
        g = m.clip((xmin, ymin, xmax, ymax))
        return g  # type: ignore

    assert clip(8, 55, 9, 56).extent == pytest.approx((8, 55, 9, 56), 0.001)
    assert clip(8, 55, 10, 56).extent == pytest.approx((8, 55, 10, 56), 0.001)
    assert clip(8.2, 55.2, 9.2, 56.2).extent == pytest.approx((8.2, 55.2, 9.2, 56.0), 0.001)
    assert clip(7.5, 55, 10, 56).extent == pytest.approx((8.0, 55, 10, 56), 0.001)
    assert clip(8, 50, 10, 56).extent == pytest.approx((8, 55, 10, 56), 0.001)
    assert clip(8, 55, 15, 56).extent == pytest.approx((8, 55, 10, 56), 0.001)
    assert clip(8, 55, 10, 60).extent == pytest.approx((8, 55, 10, 56), 0.001)
    assert clip(2.5, 55.5, 3.5, 55.5) is None


def test_mosaic_eager_vs_lazy():
    g = mosaic(grid("./tests/input/n55_e008_1arc_v3.bil"), grid("./tests/input/n55_e009_1arc_v3.bil"))
    m = mosaic("./tests/input/n55_e008_1arc_v3.bil", "./tests/input/n55_e009_1arc_v3.bil")

    g1 = g.clip((8, 55, 8.5, 56))
    g2 = m.clip((8, 55, 8.5, 56))
    assert g2
    assert g1.sha256 == g2.sha256


def test_tiling():
    g = mosaic(
        grid("./tests/input/n55_e008_1arc_v3.bil"),
        grid("./tests/input/n55_e009_1arc_v3.bil"),
    )

    fmean = g.focal_mean(2)

    assert g.extent == fmean.extent
    assert g.crs == fmean.crs
    assert g.cell_size == fmean.cell_size
    assert g.width == fmean.width
    assert g.height == fmean.height


def test_set_nan_2():
    g = dem.resample(0.02)

    assert g.is_less_than(1).then(np.nan, g).sha256 == g.set_nan(g < 1).sha256
    assert g.resample(0.04).set_nan(g < 1).sha256 == g.resample(0.04).set_nan(lambda g: g < 1).sha256
    assert g.resample(0.04).con(lambda g: g < 1, np.nan).sha256 == g.resample(0.04).set_nan(lambda g: g < 1).sha256


def test_capping():
    g = dem.resample(0.02)
    assert not g.has_nan

    g1 = g.cap_min(10)
    assert g1.min == 10
    assert g1.max == g.max
    assert not g1.has_nan
    assert g1.extent == g.extent

    g2 = g.cap_min(10, set_nan=True)
    assert g2.min == 10
    assert g2.max == g.max
    assert g2.has_nan
    assert g2.extent == g.extent

    g3 = g.cap_max(30)
    assert g3.min == g.min
    assert g3.max == 30
    assert not g3.has_nan
    assert g3.extent == g.extent

    g4 = g.cap_max(30, set_nan=True)
    assert g4.min == g.min
    assert g4.max == 30
    assert g4.has_nan
    assert g4.extent == g.extent

    g5 = g.cap_range(10, 30)
    assert g5.min == 10
    assert g5.max == 30
    assert not g5.has_nan
    assert g5.extent == g.extent

    g6 = g.cap_range(10, 30, True)
    assert g6.min == 10
    assert g6.max == 30
    assert g6.has_nan
    assert g6.extent == g.extent


def test_clustering():
    g = dem.resample(0.02)
    assert not g.has_nan

    g1 = g.slice(4)
    assert len(g1.bins) == 4
    assert g1.min == 1
    assert g1.max == 4
    assert g1.extent == g.extent

    g2 = g.kmeans_cluster(4)
    assert len(g2.bins) == 4
    assert g2.min != 1
    assert g2.max != 4
    assert g2.extent == g.extent


def test_stretching():
    g = dem.resample(0.02)
    assert not g.has_nan

    g1 = g.stretch(10, 30)
    assert g1.min == 10
    assert g1.max == 30
    assert g1.extent == g.extent

    g2 = g.stretch(-777, 888)
    assert g2.min == -777
    assert g2.max == 888
    assert g2.extent == g.extent


def test_con_2():
    g = dem.resample(0.02)

    g1 = g.set_nan(g > 10)
    assert g1.has_nan
    assert g1.min == g.min
    assert g1.max == 10
    assert g1.extent == g.extent

    g2 = g1.is_nan().then(1, 2)
    assert not g2.has_nan
    assert g2.min == 1
    assert g2.max == 2
    assert g2.extent == g.extent

    g3 = g2.con(2, np.nan)
    assert g3.has_nan
    assert g3.min == 1
    assert g3.max == 1
    assert g3.extent == g.extent

    g4 = (~g1.is_nan()).then(2, 1)
    assert g4.sha256 == g2.sha256


def test_tiling_2():
    g1 = dem.resample(0.002)
    g2 = None

    folder = "tests/output/temp2"
    for e in g1.extent.tiles(0.1, 0.1):
        name = f"{folder}/{e}.tif"
        g1.clip(e).save(name)
        g2 = g2.mosaic(grid(name)) if g2 else grid(name)

    assert g2
    assert g2.cell_size == g1.cell_size
    assert g2.crs == g1.crs
    assert g2.sha256 == g1.sha256
    shutil.rmtree(folder)


def test_mosaic_tiling():
    files = ["tests/input/n55_e008_1arc_v3.bil", "tests/input/n55_e009_1arc_v3.bil"]
    m = mosaic(*files)
    g = mosaic(*map(grid, files))
    e = (8.678, 55.2, 8.8, 55.4)
    g1 = m.clip(e)
    g2 = g.clip(e)
    g3 = None
    for tile in m.tiles(0.1, 0.1, e):
        g3 = tile if g3 is None else g3.mosaic(tile)
    assert g1
    assert g3
    assert g1.sha256 == g2.sha256


def test_clip_at():
    g = dem.clip_at(8.43, 55.5, 9, 11)
    assert g.width == 9
    assert g.height == 11
    assert dem.value_at(8.43, 55.5) == 6.0
    assert g.value_at(8.43, 55.5) == 6.0


def test_idw():
    g1 = dem.resample(0.04)
    g2 = (g1.randomize() > 0.9).then(g1, np.nan)
    g3 = g2.interp_idw(max_workers=4)
    assert not g3.has_nan

    g4 = (g3 - g2).round(3)
    assert g4.has_nan
    assert g4.min == 0
    assert g4.max == 0


def test_bytes():
    assert grid(dem.to_bytes()).sha256 == dem.sha256
