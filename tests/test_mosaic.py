import numpy as np
import rasterio
from rasterio.transform import from_origin

from glidergun import Grid, grid, mosaic
from glidergun.mosaic import Mosaic


def test_mosaic_function_with_grids():
    g1 = grid((40, 30), (0, 0, 4, 3))
    g2 = grid((50, 40), (0, 0, 4, 3))
    result_grid = mosaic(g1, g2)
    assert isinstance(result_grid, Grid)


def test_mosaic_from_files(tmp_path):
    paths = []
    for left in (0, 1):
        path = tmp_path / f"tile_{left}.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=2,
            width=2,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=from_origin(left, 2, 1, 1),
        ) as dataset:
            dataset.write(np.array([[left + 1, left + 2], [left + 3, left + 4]], dtype=np.float32), 1)
        paths.append(str(path))

    m = mosaic(*paths)
    assert isinstance(m, Mosaic)
    assert len(m.files) == 2
    clipped = m.clip((0.5, 0.5, 1.5, 1.5), index=1)
    assert clipped is not None
    assert isinstance(clipped, Grid)
    assert len(list(m.tiles(1, 1, clip_extent=(0, 0, 2, 2)))) > 0
