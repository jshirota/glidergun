from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy import ndarray
from numpy.lib.stride_tricks import sliding_window_view

if TYPE_CHECKING:
    from glidergun._grid import Grid


@dataclass(frozen=True)
class Focal:
    def focal(
        self,
        func: Callable[[ndarray], Any],
        buffer: int,
        circle: bool,
        max_workers: int,
    ) -> "Grid":
        """Apply a focal operation over a window.

        Args:
            func: Function receiving windowed array and returning per-cell values.
            buffer: Radius in cells; window size is `2*buffer+1`.
            circle: If True, use a circular mask; otherwise square window.
            max_workers: Thread pool workers for tile processing.

        Returns:
            Grid: New grid with focal operation applied.
        """
        self = cast("Grid", self)

        def f(g: "Grid") -> "Grid":
            size = 2 * buffer + 1
            mask = _mask(buffer) if circle else np.full((size, size), True)
            array = sliding_window_view(_pad(g.data, buffer), (size, size))
            result = func(array[:, :, mask])
            return g.local(result)

        return self.process_tiles(f, 8000 // buffer, buffer, max_workers)

    def focal_generic(
        self,
        func: Callable[[list[float]], float],
        buffer: int = 1,
        circle: bool = False,
        ignore_nan: bool = True,
        max_workers: int = 1,
    ) -> "Grid":
        """Generic focal applying a Python function to each window's flattened values.

        Args:
            func: Function taking a list of values and returning a scalar.
            buffer: Window buffer (radius).
            circle: Use circular mask if True.
            ignore_nan: Exclude NaNs from the input list if True.
            max_workers: Thread pool workers.

        Returns:
            Grid: Result of applying `func` over each window.
        """

        def f(a):
            values = [n for n in a if not np.isnan(n)] if ignore_nan else list(a)
            return func(values)

        return self.focal(lambda a: np.apply_along_axis(f, 2, a), buffer, circle, max_workers)

    def focal_count(
        self,
        value: float | int,
        buffer: int = 1,
        circle: bool = False,
        max_workers: int = 1,
        **kwargs,
    ):
        """Count occurrences of `value` within each window.
        Args:
            value: Value to count.
            buffer: Window buffer (radius).
            circle: Use circular mask if True.
            max_workers: Thread pool workers.
            **kwargs: Additional arguments for `np.count_nonzero`.

        Returns:
            Grid: Count of `value` per window.
        """
        return self.focal(
            lambda a: np.count_nonzero(a == value, axis=2, **kwargs),
            buffer,
            circle,
            max_workers,
        )

    def focal_ptp(self, buffer: int = 1, circle: bool = False, max_workers: int = 1, **kwargs):
        """Peak-to-peak (max - min) within each window.
        Args:
            buffer: Window buffer (radius).
            circle: Use circular mask if True.
            max_workers: Thread pool workers.
            **kwargs: Additional arguments for `np.ptp`.

        Returns:
            Grid: Peak-to-peak per window.
        """
        return self.focal(lambda a: np.ptp(a, axis=2, **kwargs), buffer, circle, max_workers)

    def focal_median(
        self,
        buffer: int = 1,
        circle: bool = False,
        ignore_nan: bool = True,
        max_workers: int = 1,
        **kwargs,
    ):
        """Median within each window (NaN-aware if `ignore_nan`).
        Args:
            buffer: Window buffer (radius).
            circle: Use circular mask if True.
            ignore_nan: Use `np.nanmedian` if True.
            max_workers: Thread pool workers.
            **kwargs: Additional arguments for median function.

        Returns:
            Grid: Median per window.
        """
        f = np.nanmedian if ignore_nan else np.median
        return self.focal(lambda a: f(a, axis=2, **kwargs), buffer, circle, max_workers)

    def focal_mean(
        self,
        buffer: int = 1,
        circle: bool = False,
        ignore_nan: bool = True,
        max_workers: int = 1,
        **kwargs,
    ):
        """Mean within each window (NaN-aware if `ignore_nan`).
        Args:
            buffer: Window buffer (radius).
            circle: Use circular mask if True.
            ignore_nan: Use `np.nanmean` if True.
            max_workers: Thread pool workers.
            **kwargs: Additional arguments for mean function.

        Returns:
            Grid: Mean per window.
        """
        f = np.nanmean if ignore_nan else np.mean
        return self.focal(lambda a: f(a, axis=2, **kwargs), buffer, circle, max_workers)

    def focal_std(
        self,
        buffer: int = 1,
        circle: bool = False,
        ignore_nan: bool = True,
        max_workers: int = 1,
        **kwargs,
    ):
        """Standard deviation within each window (NaN-aware if `ignore_nan`).
        Args:
            buffer: Window buffer (radius).
            circle: Use circular mask if True.
            ignore_nan: Use `np.nanstd` if True.
            max_workers: Thread pool workers.
            **kwargs: Additional arguments for std function.

        Returns:
            Grid: Standard deviation per window.
        """
        f = np.nanstd if ignore_nan else np.std
        return self.focal(lambda a: f(a, axis=2, **kwargs), buffer, circle, max_workers)

    def focal_var(
        self,
        buffer: int = 1,
        circle: bool = False,
        ignore_nan: bool = True,
        max_workers: int = 1,
        **kwargs,
    ):
        """Variance within each window (NaN-aware if `ignore_nan`).
        Args:
            buffer: Window buffer (radius).
            circle: Use circular mask if True.
            ignore_nan: Use `np.nanvar` if True.
            max_workers: Thread pool workers.
            **kwargs: Additional arguments for var function.

        Returns:
            Grid: Variance per window.
        """
        f = np.nanvar if ignore_nan else np.var
        return self.focal(lambda a: f(a, axis=2, **kwargs), buffer, circle, max_workers)

    def focal_min(
        self,
        buffer: int = 1,
        circle: bool = False,
        ignore_nan: bool = True,
        max_workers: int = 1,
        **kwargs,
    ):
        """Minimum within each window (NaN-aware if `ignore_nan`).
        Args:
            buffer: Window buffer (radius).
            circle: Use circular mask if True.
            ignore_nan: Use `np.nanmin` if True.
            max_workers: Thread pool workers.
            **kwargs: Additional arguments for min function.

        Returns:
            Grid: Minimum per window.
        """
        f = np.nanmin if ignore_nan else np.min
        return self.focal(lambda a: f(a, axis=2, **kwargs), buffer, circle, max_workers)

    def focal_max(
        self,
        buffer: int = 1,
        circle: bool = False,
        ignore_nan: bool = True,
        max_workers: int = 1,
        **kwargs,
    ):
        """Maximum within each window (NaN-aware if `ignore_nan`).
        Args:
            buffer: Window buffer (radius).
            circle: Use circular mask if True.
            ignore_nan: Use `np.nanmax` if True.
            max_workers: Thread pool workers.
            **kwargs: Additional arguments for max function.

        Returns:
            Grid: Maximum per window.
        """
        f = np.nanmax if ignore_nan else np.max
        return self.focal(lambda a: f(a, axis=2, **kwargs), buffer, circle, max_workers)

    def focal_sum(
        self,
        buffer: int = 1,
        circle: bool = False,
        ignore_nan: bool = True,
        max_workers: int = 1,
        **kwargs,
    ):
        """Sum within each window (NaN-aware if `ignore_nan`).
        Args:
            buffer: Window buffer (radius).
            circle: Use circular mask if True.
            ignore_nan: Use `np.nansum` if True.
            max_workers: Thread pool workers.
            **kwargs: Additional arguments for sum function.

        Returns:
            Grid: Sum per window.
        """
        f = np.nansum if ignore_nan else np.sum
        return self.focal(lambda a: f(a, axis=2, **kwargs), buffer, circle, max_workers)

    def fill_nan(self, max_exponent: int = 4, max_workers: int = 1):
        """Fill NaNs by iteratively averaging circular neighborhoods.
        Args:
            max_exponent: Use buffers of size `2**n` up to this exponent.
            max_workers: Thread pool workers.

        Returns:
            Grid: Grid with NaNs progressively filled.
        """
        self = cast("Grid", self)

        if not self.has_nan:
            return self.type("float32")

        def f(g: "Grid"):
            n = 0
            while g.has_nan and n <= max_exponent:
                g = g.is_nan().then(g.focal_mean(2**n, True), g)
                n += 1
            return g

        return self.process_tiles(f, 256, 2**max_exponent, max_workers)


def _mask(buffer: int) -> ndarray:
    size = 2 * buffer + 1
    rows = []
    for y in range(size):
        row = []
        for x in range(size):
            d = ((x - buffer) ** 2 + (y - buffer) ** 2) ** (1 / 2)
            row.append(d <= buffer)
        rows.append(row)
    return np.array(rows)


def _pad(data: ndarray, buffer: int):
    row = np.zeros((buffer, data.shape[1])) * np.nan
    col = np.zeros((data.shape[0] + 2 * buffer, buffer)) * np.nan
    return np.hstack([col, np.vstack([row, data, row]), col]).astype("float32")
