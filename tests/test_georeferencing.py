from typing import Any, cast

import cv2
import numpy as np
import pytest
import torch
from affine import Affine
from rasterio.transform import from_origin

import glidergun.loftr as georeferencing
from glidergun import CellSize, grid
from glidergun.loftr import georeference_to_reference


def _rotate_image_bound(image: np.ndarray, angle: float) -> np.ndarray:
    h, w = image.shape[:2]
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    matrix = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int(np.ceil((h * sin) + (w * cos)))
    new_h = int(np.ceil((h * cos) + (w * sin)))

    matrix[0, 2] += (new_w / 2.0) - cx
    matrix[1, 2] += (new_h / 2.0) - cy

    return cv2.warpAffine(
        image,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def test_georeference_to_reference_preserves_native_resolution(monkeypatch: pytest.MonkeyPatch):
    source = grid(np.arange(16, dtype=np.float32).reshape(4, 4), from_origin(0, 4, 1, 1), 4326)
    reference = grid(np.arange(100, dtype=np.float32).reshape(10, 10), from_origin(100, 200, 2, 2), 4326)
    homography = np.array(
        [
            [0.5, 0.0, 3.0],
            [0.0, 0.5, 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    monkeypatch.setattr(georeferencing.kornia.feature, "LoFTR", lambda pretrained: object())
    monkeypatch.setattr(georeferencing, "_best_transform_with_scan", lambda *args, **kwargs: (homography, 16))

    result = georeference_to_reference(source, reference)

    assert result.cell_size == CellSize(1, 1)
    assert result.transform == Affine(1.0, 0.0, 106.0, 0.0, -1.0, 196.0)
    assert result.extent == pytest.approx((106.0, 192.0, 110.0, 196.0))
    assert result.data.shape == (4, 4)
    assert np.allclose(result.data, source.data)


def test_best_transform_with_scan_supports_180_rotation(monkeypatch: pytest.MonkeyPatch):
    class FakeMatcher:
        def __call__(self, batch):
            image0 = batch["image0"][0, 0].cpu().numpy()
            # Only provide matches for the 180-degree hypothesis.
            if image0.shape != (6, 8) or float(image0[0, 0]) < 0.15:
                return {
                    "keypoints0": torch.empty((0, 2), dtype=torch.float32),
                    "keypoints1": torch.empty((0, 2), dtype=torch.float32),
                }

            pts_rot = np.array([[1.0, 1.0], [4.0, 1.0], [4.0, 3.0], [1.0, 3.0]], dtype=np.float32)
            w, h = 8.0, 6.0
            pts_orig = np.column_stack(((w - 1.0) - pts_rot[:, 0], (h - 1.0) - pts_rot[:, 1]))
            shift = np.array([30.0, 40.0], dtype=np.float32)
            pts_ref = pts_orig + shift
            return {
                "keypoints0": torch.from_numpy(pts_rot),
                "keypoints1": torch.from_numpy(pts_ref),
            }

    img = np.arange(48, dtype=np.uint8).reshape(6, 8)
    ref = np.zeros((200, 200), dtype=np.uint8)

    monkeypatch.setattr(georeferencing, "_coarse_candidate_windows", lambda *args, **kwargs: [(0, 0, 200, 200)])

    h, inliers = georeferencing._best_transform_with_scan(
        img,
        ref,
        cast(Any, FakeMatcher()),
        max_loftr_side=512,
        tile_size=128,
        tile_overlap=0.2,
        top_k=1,
    )

    assert inliers == 4
    assert h[0, 0] == pytest.approx(1.0, abs=1e-5)
    assert h[1, 1] == pytest.approx(1.0, abs=1e-5)
    assert h[0, 1] == pytest.approx(0.0, abs=1e-5)
    assert h[1, 0] == pytest.approx(0.0, abs=1e-5)
    assert h[0, 2] == pytest.approx(30.0, abs=1e-4)
    assert h[1, 2] == pytest.approx(40.0, abs=1e-4)


@pytest.mark.parametrize("scale", [0.5, 0.8, 1.3, 1.7])
def test_coarse_candidate_windows_handles_reference_input_size_ratio(scale: float):
    rng = np.random.default_rng(123)
    reference = rng.integers(0, 256, size=(900, 1200), dtype=np.uint8)

    x0, y0 = 420, 310
    w, h = 160, 120
    patch = reference[y0 : y0 + h, x0 : x0 + w]
    probe = cv2.resize(patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    tile_size = max(128, int(max(probe.shape) * 1.4))
    windows = georeferencing._coarse_candidate_windows(
        probe,
        reference,
        tile_size=tile_size,
        tile_overlap=0.2,
        top_k=12,
    )

    cx = x0 + (w / 2.0)
    cy = y0 + (h / 2.0)
    assert any(wx0 <= cx <= wx1 and wy0 <= cy <= wy1 for wx0, wy0, wx1, wy1 in windows)


@pytest.mark.parametrize("angle", [30.0, 70.0, 130.0])
def test_coarse_candidate_windows_handles_non_orthogonal_angles(angle: float):
    rng = np.random.default_rng(99)
    reference = rng.integers(0, 256, size=(1000, 1400), dtype=np.uint8)

    x0, y0 = 530, 420
    w, h = 180, 130
    patch = reference[y0 : y0 + h, x0 : x0 + w]
    probe = _rotate_image_bound(patch, angle)

    tile_size = max(128, int(max(probe.shape) * 1.5))
    windows = georeferencing._coarse_candidate_windows(
        probe,
        reference,
        tile_size=tile_size,
        tile_overlap=0.25,
        top_k=16,
    )

    cx = x0 + (w / 2.0)
    cy = y0 + (h / 2.0)
    assert any(wx0 <= cx <= wx1 and wy0 <= cy <= wy1 for wx0, wy0, wx1, wy1 in windows)
