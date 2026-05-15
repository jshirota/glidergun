from dataclasses import dataclass, field
from functools import lru_cache
from typing import TypeVar

import cv2
import kornia.feature
import numpy as np
import torch
from affine import Affine
from rasterio.crs import CRS
from rasterio.warp import Resampling

from glidergun.grid import Grid, from_ndarray
from glidergun.literals import ResamplingMethod
from glidergun.stack import Stack, stack

T = TypeVar("T", bound=Grid | Stack)


@dataclass(frozen=True)
class Homography:
    homography: np.ndarray | None
    best_inliers: int
    best_matches: int
    transform: Affine
    crs: CRS
    width: int
    height: int
    is_valid: bool = field(init=False)
    max_shift_px: float | None = field(init=False, default=None)
    mean_shift_px: float | None = field(init=False, default=None)
    max_dx_px: float | None = field(init=False, default=None)
    max_dy_px: float | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        object.__setattr__(self, "is_valid", self.homography is not None)

        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")

        if self.homography is None:
            return

        pts = np.array(
            [[0, 0], [self.width - 1, 0], [self.width - 1, self.height - 1], [0, self.height - 1]],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(pts, self.homography)[:, 0, :]
        src = pts[:, 0, :]
        dxy = warped - src
        mag = np.linalg.norm(dxy, axis=1)
        object.__setattr__(self, "max_shift_px", float(mag.max()))
        object.__setattr__(self, "mean_shift_px", float(mag.mean()))
        object.__setattr__(self, "max_dx_px", float(np.abs(dxy[:, 0]).max()))
        object.__setattr__(self, "max_dy_px", float(np.abs(dxy[:, 1]).max()))


@lru_cache
def _get_matcher(device_type: str) -> kornia.feature.LoFTR:
    matcher = kornia.feature.LoFTR(pretrained="outdoor")
    matcher.eval()
    return matcher.to(torch.device(device_type))


def _affine_matrix(transform: Affine) -> np.ndarray:
    return np.array(
        [
            [transform.a, transform.b, transform.c],
            [transform.d, transform.e, transform.f],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _apply_affine(transform: Affine, points: np.ndarray) -> np.ndarray:
    x = transform.a * points[:, 0] + transform.b * points[:, 1] + transform.c
    y = transform.d * points[:, 0] + transform.e * points[:, 1] + transform.f
    return np.column_stack((x, y))


def _native_cell_size(
    reference_transform: Affine,
    homography: np.ndarray,
    width: int,
    height: int,
) -> tuple[float, float]:
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32).reshape(-1, 1, 2)
    matched_pixels = cv2.perspectiveTransform(corners, homography)[:, 0, :]
    matched_world = _apply_affine(reference_transform, matched_pixels)

    top = np.linalg.norm(matched_world[1] - matched_world[0])
    bottom = np.linalg.norm(matched_world[2] - matched_world[3])
    left = np.linalg.norm(matched_world[3] - matched_world[0])
    right = np.linalg.norm(matched_world[2] - matched_world[1])

    cell_size_x = max(np.mean((top, bottom)) / max(width, 1), 1e-12)
    cell_size_y = max(np.mean((left, right)) / max(height, 1), 1e-12)
    return float(cell_size_x), float(cell_size_y)


def _resize_for_loftr(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float, float]:
    """Resize image for LoFTR while preserving aspect ratio."""
    h, w = image.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return image, 1.0, 1.0

    scale = max_side / float(side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, (w / float(new_w)), (h / float(new_h))


def _rotate_image_orthogonal(image: np.ndarray, rotation: int) -> np.ndarray:
    if rotation == 0:
        return image
    if rotation == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError("rotation must be one of 0, 90, 180, 270")


def _unrotate_points(points: np.ndarray, rotation: int, width: int, height: int) -> np.ndarray:
    if rotation == 0:
        return points

    result = points.copy()

    if rotation == 90:
        result[:, 0] = points[:, 1]
        result[:, 1] = (height - 1.0) - points[:, 0]
        return result

    if rotation == 180:
        result[:, 0] = (width - 1.0) - points[:, 0]
        result[:, 1] = (height - 1.0) - points[:, 1]
        return result

    if rotation == 270:
        result[:, 0] = (width - 1.0) - points[:, 1]
        result[:, 1] = points[:, 0]
        return result

    raise ValueError("rotation must be one of 0, 90, 180, 270")


def _coarse_candidate_windows(
    img_gray: np.ndarray,
    ref_gray: np.ndarray,
    tile_size: int,
    tile_overlap: float,
    top_k: int,
) -> list[tuple[int, int, int, int]]:
    """Find likely reference windows where the input image may match."""
    h_img, w_img = img_gray.shape[:2]
    h_ref, w_ref = ref_gray.shape[:2]

    if h_img >= h_ref or w_img >= w_ref:
        return [(0, 0, w_ref, h_ref)]

    longest = max(h_ref, w_ref)
    scan_scale = min(1.0, 1800.0 / float(longest))
    if scan_scale < 1.0:
        ref_scan = cv2.resize(ref_gray, None, fx=scan_scale, fy=scan_scale, interpolation=cv2.INTER_AREA)
        img_scan = cv2.resize(img_gray, None, fx=scan_scale, fy=scan_scale, interpolation=cv2.INTER_AREA)
    else:
        ref_scan = ref_gray
        img_scan = img_gray

    th, tw = img_scan.shape[:2]
    rh, rw = ref_scan.shape[:2]
    if th >= rh or tw >= rw:
        return [(0, 0, w_ref, h_ref)]

    response = cv2.matchTemplate(ref_scan, img_scan, cv2.TM_CCOEFF_NORMED)
    if response.size == 0:
        return [(0, 0, w_ref, h_ref)]

    k = max(1, min(int(top_k), response.size))
    flat = response.ravel()
    idx = np.argpartition(flat, -k)[-k:]
    idx = idx[np.argsort(flat[idx])[::-1]]
    ys, xs = np.unravel_index(idx, response.shape)

    base_w = max(tile_size, int(round(w_img * (1.0 + tile_overlap))))
    base_h = max(tile_size, int(round(h_img * (1.0 + tile_overlap))))
    windows: list[tuple[int, int, int, int]] = []

    for y_s, x_s in zip(ys.tolist(), xs.tolist(), strict=True):
        center_x = (x_s + tw * 0.5) / scan_scale
        center_y = (y_s + th * 0.5) / scan_scale

        x0 = int(round(center_x - base_w / 2.0))
        y0 = int(round(center_y - base_h / 2.0))
        x1 = x0 + base_w
        y1 = y0 + base_h

        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(w_ref, x1)
        y1 = min(h_ref, y1)

        if x1 - x0 < 8 or y1 - y0 < 8:
            continue

        windows.append((x0, y0, x1, y1))

    seen = set()
    deduped: list[tuple[int, int, int, int]] = []
    for window in windows:
        if window not in seen:
            deduped.append(window)
            seen.add(window)

    deduped.append((0, 0, w_ref, h_ref))
    return deduped


def _best_transform_with_scan(
    img_gray: np.ndarray,
    ref_norm: np.ndarray,
    matcher: kornia.feature.LoFTR,
    device: torch.device,
    confidence_threshold: float,
    max_loftr_side: int,
    tile_size: int,
    tile_overlap: float,
    top_k: int,
    early_stop_inliers: int,
) -> tuple[np.ndarray | None, int, int]:
    """Find best transform from candidate windows using LoFTR and RANSAC."""

    def to_tensor(im: np.ndarray):
        return torch.from_numpy(im / 255.0).float()[None, None].to(device)

    h_img, w_img = img_gray.shape[:2]

    best_h = None
    best_inliers = -1
    best_matches = 0

    for rotation in (0, 180, 90, 270):
        img_hyp = _rotate_image_orthogonal(img_gray, rotation)
        img_lf, img_x_scale, img_y_scale = _resize_for_loftr(img_hyp, max_side=max_loftr_side)
        t_img = to_tensor(img_lf)

        candidates = _coarse_candidate_windows(
            img_hyp,
            ref_norm,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            top_k=top_k,
        )

        if best_inliers >= 40 and rotation != 0:
            candidates = candidates[: min(2, len(candidates))]

        for x0, y0, x1, y1 in candidates:
            ref_crop = ref_norm[y0:y1, x0:x1]
            if ref_crop.size == 0:
                continue

            ref_lf, ref_x_scale, ref_y_scale = _resize_for_loftr(ref_crop, max_side=max_loftr_side)
            t_ref = to_tensor(ref_lf)

            with torch.inference_mode():
                correspondences = matcher({"image0": t_img, "image1": t_ref})

            pts1 = correspondences["keypoints0"].cpu().numpy()
            pts2 = correspondences["keypoints1"].cpu().numpy()
            confidence = correspondences["confidence"].cpu().numpy()
            keep = confidence >= confidence_threshold
            pts1 = pts1[keep]
            pts2 = pts2[keep]
            best_matches = max(best_matches, int(len(pts1)))
            if len(pts1) < 4 or len(pts2) < 4:
                continue

            pts1_full = pts1.copy()
            pts1_full[:, 0] *= img_x_scale
            pts1_full[:, 1] *= img_y_scale
            pts1_full = _unrotate_points(pts1_full, rotation=rotation, width=w_img, height=h_img)

            pts2_full = pts2.copy()
            pts2_full[:, 0] = pts2_full[:, 0] * ref_x_scale + x0
            pts2_full[:, 1] = pts2_full[:, 1] * ref_y_scale + y0

            affine, mask = cv2.estimateAffinePartial2D(
                pts1_full,
                pts2_full,
                method=cv2.RANSAC,
                ransacReprojThreshold=4.0,
            )

            if affine is not None and mask is not None:
                h = np.vstack([affine, [0.0, 0.0, 1.0]])
            else:
                h, mask = cv2.findHomography(pts1_full, pts2_full, cv2.RANSAC, 5.0)
                if h is None or mask is None:
                    continue

            inliers = int(mask.sum())
            if inliers > best_inliers:
                best_inliers = inliers
                best_h = h
                if best_inliers >= early_stop_inliers:
                    return best_h, best_inliers, best_matches

    return best_h, best_inliers, best_matches


def get_homography(
    input: Grid | Stack,
    reference: Grid,
    *,
    max_loftr_side: int = 640,
    confidence_threshold: float = 0.75,
    tile_size: int | None = None,
    tile_overlap: float = 0.2,
    top_k: int = 2,
    max_long_side: int | None = None,
) -> Homography:
    is_stack = isinstance(input, Stack)

    if max_long_side is not None:
        max_loftr_side = max_long_side

    if tile_size is None:
        tile_size = max_loftr_side

    if max_loftr_side < 128:
        raise ValueError("max_loftr_side must be >= 128")
    if tile_size < 128:
        raise ValueError("tile_size must be >= 128")
    if tile_overlap < 0.0 or tile_overlap >= 1.0:
        raise ValueError("tile_overlap must be between 0.0 and 1.0")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if confidence_threshold < 0.0 or confidence_threshold > 1.0:
        raise ValueError("confidence_threshold must be between 0.0 and 1.0")

    ref_norm = cv2.normalize(reference.data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  # type: ignore

    if is_stack:
        img = np.asarray([x.data for x in input.grids[:3]]).transpose(1, 2, 0).astype(np.uint8)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img = input.data.astype(np.uint8)  # type: ignore
        img_gray = img

    h_img, w_img = img_gray.shape[:2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matcher = _get_matcher(device.type)
    early_stop_inliers = 80 if max(h_img, w_img) >= 1024 else 40

    h, best_inliers, best_matches = _best_transform_with_scan(
        img_gray,
        ref_norm,
        matcher,
        device,
        confidence_threshold,
        max_loftr_side=max_loftr_side,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        top_k=top_k,
        early_stop_inliers=early_stop_inliers,
    )

    return Homography(h, max(best_inliers, 0), best_matches, reference.transform, reference.crs, w_img, h_img)


def apply_homography(input: T, homography: Homography, resampling: Resampling | ResamplingMethod = "nearest") -> T:
    img = input.to_array().transpose(1, 2, 0)
    h = homography.homography
    transform = homography.transform
    crs = homography.crs
    if h is None:
        raise ValueError(
            "Unable to estimate transform: no robust LoFTR matches were found "
            f"(best_matches={homography.best_matches}, best_inliers={homography.best_inliers})."
        )

    h_img, w_img = img.shape[:2]
    corners_src = np.float32([[0, 0], [w_img, 0], [w_img, h_img], [0, h_img]]).reshape(-1, 1, 2)  # type: ignore
    matched_pixels = cv2.perspectiveTransform(corners_src, h)[:, 0, :]
    matched_world = _apply_affine(transform, matched_pixels)

    xmin = float(matched_world[:, 0].min())
    xmax = float(matched_world[:, 0].max())
    ymin = float(matched_world[:, 1].min())
    ymax = float(matched_world[:, 1].max())

    cell_size_x, cell_size_y = _native_cell_size(transform, h, w_img, h_img)
    out_w = max(1, int(np.ceil((xmax - xmin) / cell_size_x)))
    out_h = max(1, int(np.ceil((ymax - ymin) / cell_size_y)))
    output_transform = Affine(cell_size_x, 0.0, xmin, 0.0, -cell_size_y, ymax)
    warp_matrix = np.linalg.inv(_affine_matrix(output_transform)) @ _affine_matrix(transform) @ h

    img = img.astype(np.float32)
    if img.ndim == 2:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 1:
        img_rgb = np.repeat(img, 3, axis=2)
    else:
        img_rgb = img[:, :, :3]
    method = Resampling[resampling] if isinstance(resampling, str) else resampling
    interpolation = {
        Resampling.nearest: cv2.INTER_NEAREST,
        Resampling.bilinear: cv2.INTER_LINEAR,
        Resampling.cubic: cv2.INTER_CUBIC,
        Resampling.lanczos: cv2.INTER_LANCZOS4,
    }.get(method, cv2.INTER_LINEAR)

    warped_rgb = cv2.warpPerspective(
        img_rgb,
        warp_matrix,
        (out_w, out_h),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0.0, 0.0, 0.0),
    )
    valid_mask = cv2.warpPerspective(
        np.ones(img.shape[:2], dtype=np.uint8),
        warp_matrix,
        (out_w, out_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    warped_rgb[~valid_mask] = np.nan

    r = warped_rgb[:, :, 0].copy()
    g = warped_rgb[:, :, 1].copy()
    b = warped_rgb[:, :, 2].copy()

    valid: np.ndarray = valid_mask.astype(np.float32)
    valid[valid == 0] = np.nan
    common_extent = from_ndarray(valid, output_transform, crs=crs).data_extent

    grids = []
    for c in (r, g, b):
        x = from_ndarray(c, output_transform, crs=crs)
        grids.append(x.clip(common_extent))

    result = stack(grids)
    return result if isinstance(input, Stack) else result.first  # type: ignore

def georeference_to_reference(
    input: T,
    reference: Grid | Stack | Homography,
    *,
    max_loftr_side: int = 640,
    confidence_threshold: float = 0.75,
    tile_size: int | None = None,
    tile_overlap: float = 0.2,
    top_k: int = 2,
    max_long_side: int | None = None,
    resampling: Resampling | ResamplingMethod = "nearest",
    homography_only: bool = False,
) -> T | Homography:
    if isinstance(reference, Homography):
        homography = reference
    else:
        reference_grid = reference.mean() if isinstance(reference, Stack) else reference
        homography = get_homography(
            input,
            reference_grid,
            max_loftr_side=max_loftr_side,
            confidence_threshold=confidence_threshold,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            top_k=top_k,
            max_long_side=max_long_side,
        )
    if homography_only:
        return homography
    return apply_homography(input, homography, resampling)
