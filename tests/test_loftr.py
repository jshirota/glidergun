import cv2
import numpy as np
import pytest
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject

from glidergun import grid


@pytest.mark.parametrize(
    ("shift_x", "shift_y", "rotation", "scale", "quality", "max_error"),
    [
        (12, -8, 90, 1.0, "clean", 25),
        (50, 30, 180, 0.75, "clean", 25),
        (-25, 20, 20, 1.25, "clean", 25),
        (18, 10, -90, 0.8, "blurred", 25),
        (-15, -22, 35, 1.0, "noisy", 40),
        (8, -14, 270, 0.9, "clean", 25),
        (-30, 12, 0, 1.1, "clean", 25),
        (22, -18, 15, 0.85, "blurred", 25),
        (-10, 25, -35, 1.15, "noisy", 40),
    ],
)
def test_georeference_to_numpy_grid_with_shift_rotation_and_resize(
    shift_x,
    shift_y,
    rotation,
    scale,
    quality,
    max_error,
):
    rng = np.random.default_rng(42)
    image = cv2.GaussianBlur(rng.integers(0, 256, (320, 320), dtype=np.uint8), (3, 3), 0)
    for i in range(8):
        cv2.circle(image, (35 + i * 36, 45 + (i % 3) * 90), 12 + i, 255 - i * 25, 2)
        cv2.putText(image, str(i), (25 + i * 36, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, i * 30, 2)

    reference = grid(image, Affine(1, 0, -123, 0, -1, 49), crs=4326)
    transform = cv2.getRotationMatrix2D((159.5, 159.5), rotation, scale)
    transform[:, 2] += (shift_x, shift_y)
    moved = cv2.warpAffine(image, transform, (320, 320), borderMode=cv2.BORDER_REFLECT)
    if quality == "blurred":
        moved = cv2.GaussianBlur(moved, (7, 7), 1.5)
    elif quality == "noisy":
        noise = rng.normal(0, 8, moved.shape)
        moved = np.clip(moved.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    shifted_rotated = reference.local(lambda _: moved)

    homography = shifted_rotated.georeference_to(reference, homography_only=True)
    assert homography.homography is not None
    assert homography.best_inliers >= 20

    aligned = shifted_rotated.georeference_to(homography)
    reference_on_aligned_grid = np.full(aligned.data.shape, np.nan, dtype=np.float32)
    reproject(
        reference.data,
        reference_on_aligned_grid,
        src_transform=reference.transform,
        src_crs=reference.crs,
        dst_transform=aligned.transform,
        dst_crs=aligned.crs,
        src_nodata=None,
        dst_nodata=np.nan,
        resampling=Resampling.nearest,
    )
    valid = np.isfinite(aligned.data) & np.isfinite(reference_on_aligned_grid)
    assert valid.sum() > 20_000
    assert np.mean(np.abs(aligned.data[valid] - reference_on_aligned_grid[valid])) < max_error
