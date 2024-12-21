import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, List, Tuple, Union, cast

import numpy as np
import scipy as sp
from numpy import ndarray
from numpy.lib.stride_tricks import sliding_window_view

from glidergun._types import StatsResult

if TYPE_CHECKING:
    from glidergun._grid import Grid


@dataclass(frozen=True)
class Focal:
    def focal(
        self, func: Callable[[ndarray], Any], buffer: int, circle: bool
    ) -> "Grid":
        return _batch(
            lambda g: _focal(func, buffer, circle, *g), buffer, cast("Grid", self)
        )[0]

    def focal_python(
        self,
        func: Callable[[List[float]], float],
        buffer: int = 1,
        circle: bool = False,
        ignore_nan: bool = True,
    ) -> "Grid":
        def f(a):
            values = [n for n in a if n != np.nan] if ignore_nan else list(a)
            return func(values)

        return self.focal(lambda a: np.apply_along_axis(f, 2, a), buffer, circle)

    def focal_count(
        self, value: Union[float, int], buffer: int = 1, circle: bool = False, **kwargs
    ):
        return self.focal(
            lambda a: np.count_nonzero(a == value, axis=2, **kwargs), buffer, circle
        )

    def focal_ptp(self, buffer: int = 1, circle: bool = False, **kwargs):
        return self.focal(lambda a: np.ptp(a, axis=2, **kwargs), buffer, circle)

    def focal_percentile(
        self,
        percentile: float,
        buffer: int = 1,
        circle: bool = False,
        ignore_nan: bool = True,
        **kwargs,
    ):
        f = np.nanpercentile if ignore_nan else np.percentile
        return self.focal(lambda a: f(a, percentile, axis=2, **kwargs), buffer, circle)

    def focal_quantile(
        self,
        probability: float,
        buffer: int = 1,
        circle: bool = False,
        ignore_nan: bool = True,
        **kwargs,
    ):
        f = np.nanquantile if ignore_nan else np.quantile
        return self.focal(lambda a: f(a, probability, axis=2, **kwargs), buffer, circle)

    def focal_median(
        self, buffer: int = 1, circle: bool = False, ignore_nan: bool = True, **kwargs
    ):
        f = np.nanmedian if ignore_nan else np.median
        return self.focal(lambda a: f(a, axis=2, **kwargs), buffer, circle)

    def focal_mean(
        self, buffer: int = 1, circle: bool = False, ignore_nan: bool = True, **kwargs
    ):
        f = np.nanmean if ignore_nan else np.mean
        return self.focal(lambda a: f(a, axis=2, **kwargs), buffer, circle)

    def focal_std(
        self, buffer: int = 1, circle: bool = False, ignore_nan: bool = True, **kwargs
    ):
        f = np.nanstd if ignore_nan else np.std
        return self.focal(lambda a: f(a, axis=2, **kwargs), buffer, circle)

    def focal_var(
        self, buffer: int = 1, circle: bool = False, ignore_nan: bool = True, **kwargs
    ):
        f = np.nanvar if ignore_nan else np.var
        return self.focal(lambda a: f(a, axis=2, **kwargs), buffer, circle)

    def focal_min(
        self, buffer: int = 1, circle: bool = False, ignore_nan: bool = True, **kwargs
    ):
        f = np.nanmin if ignore_nan else np.min
        return self.focal(lambda a: f(a, axis=2, **kwargs), buffer, circle)

    def focal_max(
        self, buffer: int = 1, circle: bool = False, ignore_nan: bool = True, **kwargs
    ):
        f = np.nanmax if ignore_nan else np.max
        return self.focal(lambda a: f(a, axis=2, **kwargs), buffer, circle)

    def focal_sum(
        self, buffer: int = 1, circle: bool = False, ignore_nan: bool = True, **kwargs
    ):
        f = np.nansum if ignore_nan else np.sum
        return self.focal(lambda a: f(a, axis=2, **kwargs), buffer, circle)

    def _kwargs(self, ignore_nan: bool, **kwargs):
        return {
            "axis": 2,
            "nan_policy": "omit" if ignore_nan else "propagate",
            **kwargs,
        }

    def focal_entropy(self, buffer: int = 1, circle: bool = False, **kwargs):
        return self.focal(
            lambda a: sp.stats.entropy(a, axis=2, **kwargs), buffer, circle
        )

    def focal_gmean(
        self, buffer: int = 1, circle: bool = False, ignore_nan: bool = True, **kwargs
    ):
        return self.focal(
            lambda a: sp.stats.gmean(a, **self._kwargs(ignore_nan, **kwargs)),
            buffer,
            circle,
        )

    def focal_hmean(
        self, buffer: int = 1, circle: bool = False, ignore_nan: bool = True, **kwargs
    ):
        return self.focal(
            lambda a: sp.stats.hmean(a, **self._kwargs(ignore_nan, **kwargs)),
            buffer,
            circle,
        )

    def focal_pmean(
        self,
        p: float,
        buffer: int = 1,
        circle: bool = False,
        ignore_nan: bool = True,
        **kwargs,
    ):
        return self.focal(
            lambda a: sp.stats.pmean(a, p, **self._kwargs(ignore_nan, **kwargs)),
            buffer,
            circle,
        )

    def focal_kurtosis(
        self, buffer: int = 1, circle: bool = False, ignore_nan: bool = True, **kwargs
    ):
        return self.focal(
            lambda a: sp.stats.kurtosis(a, **self._kwargs(ignore_nan, **kwargs)),
            buffer,
            circle,
        )

    def focal_iqr(
        self, buffer: int = 1, circle: bool = False, ignore_nan: bool = True, **kwargs
    ):
        return self.focal(
            lambda a: sp.stats.iqr(a, **self._kwargs(ignore_nan, **kwargs)),
            buffer,
            circle,
        )

    def focal_moment(
        self, buffer: int = 1, circle: bool = False, ignore_nan: bool = True, **kwargs
    ):
        return self.focal(
            lambda a: sp.stats.moment(a, **self._kwargs(ignore_nan, **kwargs)),
            buffer,
            circle,
        )

    def focal_skew(
        self, buffer: int = 1, circle: bool = False, ignore_nan: bool = True, **kwargs
    ):
        return self.focal(
            lambda a: sp.stats.skew(a, **self._kwargs(ignore_nan, **kwargs)),
            buffer,
            circle,
        )

    def focal_kstat(
        self, buffer: int = 1, circle: bool = False, ignore_nan: bool = True, **kwargs
    ):
        return self.focal(
            lambda a: sp.stats.kstat(a, **self._kwargs(ignore_nan, **kwargs)),
            buffer,
            circle,
        )

    def focal_kstatvar(
        self, buffer: int = 1, circle: bool = False, ignore_nan: bool = True, **kwargs
    ):
        return self.focal(
            lambda a: sp.stats.kstatvar(a, **self._kwargs(ignore_nan, **kwargs)),
            buffer,
            circle,
        )

    def focal_tmean(
        self, buffer: int = 1, circle: bool = False, ignore_nan: bool = True, **kwargs
    ):
        return self.focal(
            lambda a: sp.stats.tmean(a, **self._kwargs(ignore_nan, **kwargs)),
            buffer,
            circle,
        )

    def focal_tvar(self, buffer: int = 1, circle: bool = False, **kwargs):
        return self.focal(lambda a: sp.stats.tvar(a, axis=2, **kwargs), buffer, circle)

    def focal_tmin(
        self, buffer: int = 1, circle: bool = False, ignore_nan: bool = True, **kwargs
    ):
        return self.focal(
            lambda a: sp.stats.tmin(a, **self._kwargs(ignore_nan, **kwargs)),
            buffer,
            circle,
        )

    def focal_tmax(
        self, buffer: int = 1, circle: bool = False, ignore_nan: bool = True, **kwargs
    ):
        return self.focal(
            lambda a: sp.stats.tmax(a, **self._kwargs(ignore_nan, **kwargs)),
            buffer,
            circle,
        )

    def focal_tstd(self, buffer: int = 1, circle: bool = False, **kwargs):
        return self.focal(lambda a: sp.stats.tstd(a, axis=2, **kwargs), buffer, circle)

    def focal_variation(
        self, buffer: int = 1, circle: bool = False, ignore_nan: bool = True, **kwargs
    ):
        return self.focal(
            lambda a: sp.stats.variation(a, **self._kwargs(ignore_nan, **kwargs)),
            buffer,
            circle,
        )

    def focal_median_abs_deviation(
        self, buffer: int = 1, circle: bool = False, ignore_nan: bool = True, **kwargs
    ):
        return self.focal(
            lambda a: sp.stats.median_abs_deviation(
                a, **self._kwargs(ignore_nan, **kwargs)
            ),
            buffer,
            circle,
        )

    def focal_chisquare(self, buffer: int = 1, circle: bool = False, **kwargs):
        def f(grids):
            return _focal(
                lambda a: sp.stats.chisquare(a, axis=2, **kwargs),
                buffer,
                circle,
                *grids,
            )

        return StatsResult(*_batch(f, buffer, cast("Grid", self)))

    def focal_ttest_ind(
        self, other_grid: "Grid", buffer: int = 1, circle: bool = False, **kwargs
    ):
        def f(grids):
            return _focal(
                lambda a: sp.stats.ttest_ind(*a, axis=2, **kwargs),
                buffer,
                circle,
                *grids,
            )

        return StatsResult(*_batch(f, buffer, cast("Grid", self), other_grid))

    def fill_nan(self, max_exponent: int = 4):
        if not cast("Grid", self).has_nan:
            return self

        def f(grids: Tuple["Grid", ...]):
            g = grids[0]
            n = 0
            while g.has_nan and n <= max_exponent:
                g = g.is_nan().con(g.focal_mean(2**n, True), g)
                n += 1
            return (g,)

        return _batch(f, 2**max_exponent, cast("Grid", self))[0]


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
    return np.hstack([col, np.vstack([row, data, row]), col], dtype="float32")


def _focal(
    func: Callable, buffer: int, circle: bool, *grids: "Grid"
) -> Tuple["Grid", ...]:
    grids_adjusted = grids[0].standardize(*grids[1:])
    size = 2 * buffer + 1
    mask = _mask(buffer) if circle else np.full((size, size), True)

    if len(grids) == 1:
        array = sliding_window_view(_pad(grids[0].data, buffer), (size, size))
        result = func(array[:, :, mask])
    else:
        array = np.stack(
            [
                sliding_window_view(_pad(g.data, buffer), (size, size))
                for g in grids_adjusted
            ]
        )
        transposed = np.transpose(array, axes=(1, 2, 0, 3, 4))[:, :, :, mask]
        result = func(tuple(transposed[:, :, i] for i, _ in enumerate(grids)))

    if isinstance(result, ndarray) and len(result.shape) == 2:
        return (grids_adjusted[0].update(np.array(result)),)

    return tuple([grids_adjusted[0].update(r) for r in result])


def _batch(
    func: Callable[[Tuple["Grid", ...]], Tuple["Grid", ...]],
    buffer: int,
    *grids: "Grid",
):
    stride = 8000 // buffer // len(grids)
    grids1 = grids[0].standardize(*grids[1:])
    g = grids1[0]

    def tile():
        for x in range(0, g.width // stride + 1):
            xmin, xmax = x * stride, min((x + 1) * stride, g.width)
            if xmin < xmax:
                for y in range(0, g.height // stride + 1):
                    ymin, ymax = y * stride, min((y + 1) * stride, g.height)
                    if ymin < ymax:
                        yield xmin, ymin, xmax, ymax

    tiles = list(tile())
    count = len(tiles)

    if count <= 4:
        return func(tuple(grids1))

    results: List["Grid"] = []
    cell_size = g.cell_size
    n = 0

    for xmin, ymin, xmax, ymax in tiles:
        n += 1
        sys.stdout.write(f"\rProcessing {n} of {count} tiles...")
        sys.stdout.flush()
        grids2 = [
            g1.clip(
                g.xmin + (xmin - buffer) * cell_size.x,
                g.ymin + (ymin - buffer) * cell_size.y,
                g.xmin + (xmax + buffer) * cell_size.x,
                g.ymin + (ymax + buffer) * cell_size.y,
            )
            for g1 in grids1
        ]

        grids3 = func(tuple(grids2))

        grids4 = [
            g3.clip(
                g.xmin + xmin * cell_size.x,
                g.ymin + ymin * cell_size.y,
                g.xmin + xmax * cell_size.x,
                g.ymin + ymax * cell_size.y,
            )
            for g3 in grids3
        ]

        if results:
            for i, g4 in enumerate(grids4):
                results[i] = results[i].mosaic(g4)
        else:
            results = grids4

    print()
    return tuple(results)
