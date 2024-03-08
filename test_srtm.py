import os
import pytest
import rasterio
import shutil
from glidergun import *

dem = grid("./.data/n55_e008_1arc_v3.bil")


def test_aspect():
    g = dem.aspect()
    assert g.md5 == "398fd901ffdf84ef51ae50474e3420e4"


def test_boolean():
    g1 = dem > 20 and dem < 40
    g2 = 20 < dem < 40
    g3 = g1 and g2
    g4 = g1 * 100 - g3 * 100
    assert pytest.approx(g4.min, 0.001) == 0
    assert pytest.approx(g4.max, 0.001) == 0


def test_clip():
    xmin, ymin, xmax, ymax = dem.extent
    extent = xmin + 0.02, ymin + 0.03, xmax - 0.04, ymax - 0.05
    for a, b in zip(dem.clip(extent).extent, extent):
        assert pytest.approx(a, 0.001) == b


def test_clip_2():
    xmin, ymin, xmax, ymax = dem.extent
    extent = xmin - 0.02, ymin - 0.03, xmax + 0.04, ymax + 0.05
    for a, b in zip(dem.clip(extent).extent, extent):
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


def test_hillshade():
    g = dem.hillshade()
    assert g.md5 == "b947266de9fd27237405a34093ae513a"


def test_gosper():
    def tick(grid: Grid):
        g = grid.focal_sum() - grid
        return (grid == 1) & (g == 2) | (g == 3)

    gosper = tick(grid(".data/gosper.txt"))
    md5s = set()

    while gosper.md5 not in md5s:
        md5s.add(gosper.md5)
        gosper = tick(gosper)

    assert len(md5s) == 60


def test_mosaic():
    g1 = grid("./.data/n55_e009_1arc_v3.bil")
    g2 = mosaic(dem, g1)
    assert g2.crs == dem.crs
    xmin, ymin, xmax, ymax = g2.extent
    assert pytest.approx(xmin, 0.001) == dem.xmin
    assert pytest.approx(ymin, 0.001) == min(dem.ymin, g1.ymin)
    assert pytest.approx(xmax, 0.001) == g1.xmax
    assert pytest.approx(ymax, 0.001) == max(dem.ymax, g1.ymax)


def test_operators():
    g = dem * 100
    assert pytest.approx(g.min, 0.001) == dem.min * 100
    assert pytest.approx(g.max, 0.001) == dem.max * 100


def test_operators_2():
    g = dem / 100
    assert pytest.approx(g.min, 0.001) == dem.min / 100
    assert pytest.approx(g.max, 0.001) == dem.max / 100


def test_operators_3():
    g = dem + 100
    assert pytest.approx(g.min, 0.001) == dem.min + 100
    assert pytest.approx(g.max, 0.001) == dem.max + 100


def test_operators_4():
    g = dem - 100
    assert pytest.approx(g.min, 0.001) == dem.min - 100
    assert pytest.approx(g.max, 0.001) == dem.max - 100


def test_operators_5():
    g = 2 * dem - dem / 2 - dem / 4
    assert pytest.approx(g.min, 0.001) == 2 * dem.min - dem.min / 2 - dem.min / 4
    assert pytest.approx(g.max, 0.001) == 2 * dem.max - dem.max / 2 - dem.max / 4


def test_operators_6():
    g = (-dem) ** 2 - dem**2
    assert pytest.approx(g.min, 0.001) == 0
    assert pytest.approx(g.max, 0.001) == 0


def test_operators_7():
    g1 = con(dem > 20, 7, 11)
    g2 = g1 % 3
    assert pytest.approx(g2.min, 0.001) == 1
    assert pytest.approx(g2.max, 0.001) == 2


def test_to_points():
    g = (dem.resample(0.01).randomize() < 0.01).set_nan(0).randomize()
    n = 0
    for x, y, value in g.to_points():
        n += 1
        assert g.value(x, y) == value
    assert n > 1000


def test_project():
    g = dem.project(3857)
    assert g.crs.wkt.startswith('PROJCS["WGS 84 / Pseudo-Mercator",')


def test_properties():
    assert dem.width == 1801
    assert dem.height == 3601
    assert dem.dtype == "float32"
    assert dem.md5 == "09b41b3363bd79a87f28e3c5c4716724"


def test_resample():
    g = dem.resample(0.01)
    assert g.cell_size == (0.01, 0.01)


def test_resample_2():
    g = dem.resample((0.02, 0.03))
    assert g.cell_size == (0.02, 0.03)


def test_slope():
    g = dem.slope()
    assert g.md5 == "6d7d0bdf8a2b38035b6fbc9d9c5cbd63"


def test_trig():
    g1 = dem.sin()
    assert pytest.approx(g1.min, 0.001) == -1
    assert pytest.approx(g1.max, 0.001) == 1
    g2 = dem.cos()
    assert pytest.approx(g2.min, 0.001) == -1
    assert pytest.approx(g2.max, 0.001) == 1
    g3 = dem.tan()
    assert pytest.approx(g3.min, 0.001) == -225.951
    assert pytest.approx(g3.max, 0.001) == 225.951


def test_save():
    memory_file = rasterio.MemoryFile()
    dem.save(memory_file)
    g = grid(memory_file)
    assert g.md5 == dem.md5


def save(file: str, strict: bool = True):
    folder = ".output/test"
    file_path = f"{folder}/{file}"
    os.makedirs(folder, exist_ok=True)
    dem.save(file_path)
    g = grid(file_path)
    if strict:
        assert g.md5 == dem.md5
    assert g.extent == dem.extent
    shutil.rmtree(folder)


def test_save_2():
    save("test_save_2.bil")


def test_save_3():
    save("test_save_2.img")


def test_save_4():
    save("test_save_2.tif")


def test_save_5():
    save("test_save_2.jpg", strict=False)


def test_save_6():
    save("test_save_2.png", strict=False)
