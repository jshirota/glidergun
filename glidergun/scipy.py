import typing
import scipy as sp
from glidergun.core import Grid, Value, _batch, _focal


def gaussian_filter(grid: Grid, sigma: float, **kwargs):
    return grid.local(lambda a: sp.ndimage.gaussian_filter(a, sigma, **kwargs))


def gaussian_filter1d(grid: Grid, sigma: float, **kwargs):
    return grid.local(lambda a: sp.ndimage.gaussian_filter1d(a, sigma, **kwargs))


def gaussian_gradient_magnitude(grid: Grid, sigma: float, **kwargs):
    return grid.local(
        lambda a: sp.ndimage.gaussian_gradient_magnitude(a, sigma, **kwargs)
    )


def gaussian_laplace(grid: Grid, sigma: float, **kwargs):
    return grid.local(lambda a: sp.ndimage.gaussian_laplace(a, sigma, **kwargs))


def prewitt(grid: Grid, **kwargs):
    return grid.local(lambda a: sp.ndimage.prewitt(a, **kwargs))


def sobel(grid: Grid, **kwargs):
    return grid.local(lambda a: sp.ndimage.sobel(a, **kwargs))


def uniform_filter(grid: Grid, **kwargs):
    return grid.local(lambda a: sp.ndimage.uniform_filter(a, **kwargs))


def uniform_filter1d(grid: Grid, size: float, **kwargs):
    return grid.local(lambda a: sp.ndimage.uniform_filter1d(a, size, **kwargs))


def _kwargs(ignore_nan: bool, **kwargs):
    return {
        "axis": 2,
        "nan_policy": "omit" if ignore_nan else "propagate",
        **kwargs,
    }


def focal_entropy(grid: Grid, buffer=1, circle: bool = False, **kwargs):
    return grid.focal(lambda a: sp.stats.entropy(a, axis=2, **kwargs), buffer, circle)


def focal_gmean(
    grid: Grid, buffer=1, circle: bool = False, ignore_nan: bool = True, **kwargs
):
    return grid.focal(
        lambda a: sp.stats.gmean(a, **_kwargs(ignore_nan, **kwargs)),
        buffer,
        circle,
    )


def focal_hmean(
    grid: Grid, buffer=1, circle: bool = False, ignore_nan: bool = True, **kwargs
):
    return grid.focal(
        lambda a: sp.stats.hmean(a, **_kwargs(ignore_nan, **kwargs)),
        buffer,
        circle,
    )


def focal_pmean(
    grid: Grid,
    p: Value,
    buffer=1,
    circle: bool = False,
    ignore_nan: bool = True,
    **kwargs
):
    return grid.focal(
        lambda a: sp.stats.pmean(a, p, **_kwargs(ignore_nan, **kwargs)),
        buffer,
        circle,
    )


def focal_kurtosis(
    grid: Grid, buffer=1, circle: bool = False, ignore_nan: bool = True, **kwargs
):
    return grid.focal(
        lambda a: sp.stats.kurtosis(a, **_kwargs(ignore_nan, **kwargs)),
        buffer,
        circle,
    )


def focal_iqr(
    grid: Grid, buffer=1, circle: bool = False, ignore_nan: bool = True, **kwargs
):
    return grid.focal(
        lambda a: sp.stats.iqr(a, **_kwargs(ignore_nan, **kwargs)),
        buffer,
        circle,
    )


def focal_mode(
    grid: Grid, buffer=1, circle: bool = False, ignore_nan: bool = True, **kwargs
):
    return grid.focal(
        lambda a: sp.stats.mode(a, **_kwargs(ignore_nan, **kwargs)),
        buffer,
        circle,
    )


def focal_moment(
    grid: Grid, buffer=1, circle: bool = False, ignore_nan: bool = True, **kwargs
):
    return grid.focal(
        lambda a: sp.stats.moment(a, **_kwargs(ignore_nan, **kwargs)),
        buffer,
        circle,
    )


def focal_skew(
    grid: Grid, buffer=1, circle: bool = False, ignore_nan: bool = True, **kwargs
):
    return grid.focal(
        lambda a: sp.stats.skew(a, **_kwargs(ignore_nan, **kwargs)),
        buffer,
        circle,
    )


def focal_kstat(
    grid: Grid, buffer=1, circle: bool = False, ignore_nan: bool = True, **kwargs
):
    return grid.focal(
        lambda a: sp.stats.kstat(a, **_kwargs(ignore_nan, **kwargs)),
        buffer,
        circle,
    )


def focal_kstatvar(
    grid: Grid, buffer=1, circle: bool = False, ignore_nan: bool = True, **kwargs
):
    return grid.focal(
        lambda a: sp.stats.kstatvar(a, **_kwargs(ignore_nan, **kwargs)),
        buffer,
        circle,
    )


def focal_tmean(
    grid: Grid, buffer=1, circle: bool = False, ignore_nan: bool = True, **kwargs
):
    return grid.focal(
        lambda a: sp.stats.tmean(a, **_kwargs(ignore_nan, **kwargs)),
        buffer,
        circle,
    )


def focal_tvar(grid: Grid, buffer=1, circle: bool = False, **kwargs):
    return grid.focal(lambda a: sp.stats.tvar(a, axis=2, **kwargs), buffer, circle)


def focal_tmin(
    grid: Grid, buffer=1, circle: bool = False, ignore_nan: bool = True, **kwargs
):
    return grid.focal(
        lambda a: sp.stats.tmin(a, **_kwargs(ignore_nan, **kwargs)),
        buffer,
        circle,
    )


def focal_tmax(
    grid: Grid, buffer=1, circle: bool = False, ignore_nan: bool = True, **kwargs
):
    return grid.focal(
        lambda a: sp.stats.tmax(a, **_kwargs(ignore_nan, **kwargs)),
        buffer,
        circle,
    )


def focal_tstd(grid: Grid, buffer=1, circle: bool = False, **kwargs):
    return grid.focal(lambda a: sp.stats.tstd(a, axis=2, **kwargs), buffer, circle)


def focal_variation(
    grid: Grid, buffer=1, circle: bool = False, ignore_nan: bool = True, **kwargs
):
    return grid.focal(
        lambda a: sp.stats.variation(a, **_kwargs(ignore_nan, **kwargs)),
        buffer,
        circle,
    )


def focal_median_abs_deviation(
    grid: Grid, buffer=1, circle: bool = False, ignore_nan: bool = True, **kwargs
):
    return grid.focal(
        lambda a: sp.stats.median_abs_deviation(a, **_kwargs(ignore_nan, **kwargs)),
        buffer,
        circle,
    )


class StatsResult(typing.NamedTuple):
    statistic: Grid
    pvalue: Grid


def focal_chisquare(buffer: int, circle: bool, grid: Grid, **kwargs) -> StatsResult:
    def f(grids):
        return _focal(
            lambda a: sp.stats.chisquare(a, axis=2, **kwargs),
            buffer,
            circle,
            *grids,
        )

    return StatsResult(*_batch(f, buffer, grid))


def focal_f_oneway(buffer: int, circle: bool, *grids: Grid, **kwargs) -> StatsResult:
    def f(grids):
        return _focal(
            lambda a: sp.stats.f_oneway(*a, axis=2, **kwargs),
            buffer,
            circle,
            *grids,
        )

    return StatsResult(*_batch(f, buffer, *grids))


def focal_ttest_ind(
    buffer: int, circle: bool, grid1: Grid, grid2: Grid, **kwargs
) -> StatsResult:
    def f(grids):
        return _focal(
            lambda a: sp.stats.ttest_ind(*a, axis=2, **kwargs),
            buffer,
            circle,
            *grids,
        )

    return StatsResult(*_batch(f, buffer, grid1, grid2))


def zonal_entropy(grid: Grid, zone_grid: "Grid", **kwargs):
    return grid.zonal(lambda a: sp.stats.entropy(a, **kwargs), zone_grid)


def zonal_gmean(grid: Grid, zone_grid: "Grid", **kwargs):
    return grid.zonal(lambda a: sp.stats.gmean(a, **kwargs), zone_grid)


def zonal_hmean(grid: Grid, zone_grid: "Grid", **kwargs):
    return grid.zonal(lambda a: sp.stats.hmean(a, **kwargs), zone_grid)


def zonal_pmean(grid: Grid, p: Value, zone_grid: "Grid", **kwargs):
    return grid.zonal(lambda a: sp.stats.pmean(a, p, **kwargs), zone_grid)


def zonal_kurtosis(grid: Grid, zone_grid: "Grid", **kwargs):
    return grid.zonal(lambda a: sp.stats.kurtosis(a, **kwargs), zone_grid)


def zonal_iqr(grid: Grid, zone_grid: "Grid", **kwargs):
    return grid.zonal(lambda a: sp.stats.iqr(a, **kwargs), zone_grid)


def zonal_mode(grid: Grid, zone_grid: "Grid", **kwargs):
    return grid.zonal(lambda a: sp.stats.mode(a, **kwargs), zone_grid)


def zonal_moment(grid: Grid, zone_grid: "Grid", **kwargs):
    return grid.zonal(lambda a: sp.stats.moment(a, **kwargs), zone_grid)


def zonal_skew(grid: Grid, zone_grid: "Grid", **kwargs):
    return grid.zonal(lambda a: sp.stats.skew(a, **kwargs), zone_grid)


def zonal_kstat(grid: Grid, zone_grid: "Grid", **kwargs):
    return grid.zonal(lambda a: sp.stats.kstat(a, **kwargs), zone_grid)


def zonal_kstatvar(grid: Grid, zone_grid: "Grid", **kwargs):
    return grid.zonal(lambda a: sp.stats.kstatvar(a, **kwargs), zone_grid)


def zonal_tmean(grid: Grid, zone_grid: "Grid", **kwargs):
    return grid.zonal(lambda a: sp.stats.tmean(a, **kwargs), zone_grid)


def zonal_tvar(grid: Grid, zone_grid: "Grid", **kwargs):
    return grid.zonal(lambda a: sp.stats.tvar(a, **kwargs), zone_grid)


def zonal_tmin(grid: Grid, zone_grid: "Grid", **kwargs):
    return grid.zonal(lambda a: sp.stats.tmin(a, **kwargs), zone_grid)


def zonal_tmax(grid: Grid, zone_grid: "Grid", **kwargs):
    return grid.zonal(lambda a: sp.stats.tmax(a, **kwargs), zone_grid)


def zonal_tstd(grid: Grid, zone_grid: "Grid", **kwargs):
    return grid.zonal(lambda a: sp.stats.tstd(a, **kwargs), zone_grid)


def zonal_variation(grid: Grid, zone_grid: "Grid", **kwargs):
    return grid.zonal(lambda a: sp.stats.variation(a, **kwargs), zone_grid)


def zonal_median_abs_deviation(grid: Grid, zone_grid: "Grid", **kwargs):
    return grid.zonal(lambda a: sp.stats.median_abs_deviation(a, **kwargs), zone_grid)
