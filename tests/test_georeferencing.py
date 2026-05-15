import numpy as np
import pytest
import torch
from rasterio.transform import from_origin

import glidergun.loftr as georeferencing
from glidergun import Grid, grid
from glidergun.loftr import georeference_to_reference


def test_georeference_to_reference_uses_reference_frame(monkeypatch: pytest.MonkeyPatch):
    source = grid(np.arange(16, dtype=np.float32).reshape(4, 4), from_origin(0, 4, 1, 1))
    reference = grid(np.arange(100, dtype=np.float32).reshape(10, 10), from_origin(100, 200, 2, 2))

    monkeypatch.setattr(
        georeferencing,
        "_best_transform_with_scan",
        lambda *args, **kwargs: (np.eye(3), 10, []),
    )
    monkeypatch.setattr(georeferencing, "_get_matcher", lambda *_: object())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    result: Grid = georeference_to_reference(source, reference) # type: ignore

    assert result.cell_size == reference.cell_size
    assert np.isfinite(result.data).sum() > 0


def test_loftr2_forwards_confidence_threshold(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, float] = {}

    def fake_best_transform(*args, **kwargs):
        captured["confidence_threshold"] = float(args[4])
        return np.eye(3, dtype=np.float64), 10, []

    monkeypatch.setattr(georeferencing, "_best_transform_with_scan", fake_best_transform)
    monkeypatch.setattr(georeferencing, "_get_matcher", lambda *_: object())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    source = grid(np.arange(16, dtype=np.float32).reshape(4, 4), from_origin(0, 4, 1, 1))
    reference = grid(np.arange(100, dtype=np.float32).reshape(10, 10), from_origin(100, 200, 2, 2))

    georeferencing.georeference_to_reference(source, reference, confidence_threshold=0.7)

    assert captured["confidence_threshold"] == 0.7


def test_loftr2_raises_when_no_robust_matches(monkeypatch: pytest.MonkeyPatch):
    def fake_best_transform(*args, **kwargs):
        raise ValueError("Unable to estimate transform: no robust LoFTR matches were found.")

    monkeypatch.setattr(georeferencing, "_best_transform_with_scan", fake_best_transform)
    monkeypatch.setattr(georeferencing, "_get_matcher", lambda *_: object())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    source = grid(np.arange(16, dtype=np.float32).reshape(4, 4), from_origin(0, 4, 1, 1))
    reference = grid(np.arange(100, dtype=np.float32).reshape(10, 10), from_origin(100, 200, 2, 2))

    with pytest.raises(ValueError, match="Unable to estimate transform"):
        georeferencing.georeference_to_reference(source, reference)


def test_resize_for_loftr_scales_and_crops_to_multiple_of_8():
    image = np.zeros((103, 205), dtype=np.uint8)
    resized, x_scale, y_scale = georeferencing._resize_for_loftr(image, max_side=100)

    assert resized.shape == (50, 100)
    assert x_scale == pytest.approx(205 / 100)
    assert y_scale == pytest.approx(103 / 50)


def test_grid_georeference_to_forwards_options_to_loftr2(monkeypatch: pytest.MonkeyPatch):
    source = grid(np.arange(16, dtype=np.float32).reshape(4, 4), from_origin(0, 4, 1, 1))
    reference = grid(np.arange(100, dtype=np.float32).reshape(10, 10), from_origin(100, 200, 2, 2))
    captured: dict[str, object] = {}

    def fake_georeference_to_reference(input, ref, **kwargs):
        captured.update(kwargs)
        return input

    monkeypatch.setattr(georeferencing, "georeference_to_reference", fake_georeference_to_reference)

    result = source.georeference_to(
        reference,
        confidence_threshold=0.55,
        max_long_side=512,
        resampling="bilinear",
    )

    assert result is source
    assert captured["confidence_threshold"] == 0.55
    assert captured["max_long_side"] == 512
    assert captured["resampling"] == "bilinear"


def test_loftr2_accepts_max_long_side_alias(monkeypatch: pytest.MonkeyPatch):
    source = grid(np.arange(16, dtype=np.float32).reshape(4, 4), from_origin(0, 4, 1, 1))
    reference = grid(np.arange(100, dtype=np.float32).reshape(10, 10), from_origin(100, 200, 2, 2))
    captured: dict[str, int] = {}

    def fake_best_transform(*args, **kwargs):
        max_side = kwargs["max_loftr_side"]
        captured["max_loftr_side"] = int(max_side)
        return np.eye(3, dtype=np.float64), 10, []

    monkeypatch.setattr(georeferencing, "_best_transform_with_scan", fake_best_transform)
    monkeypatch.setattr(georeferencing, "_get_matcher", lambda *_: object())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    georeferencing.georeference_to_reference(source, reference, max_long_side=256)
    assert captured["max_loftr_side"] == 256
