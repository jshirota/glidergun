from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Union, cast

import numpy as np
import scipy as sp
from numpy import ndarray

if TYPE_CHECKING:
    from glidergun._grid import Grid


@dataclass(frozen=True)
class Zonal:
    def zonal(self, func: Callable[[ndarray], Any], zone_grid: "Grid") -> "Grid":
        g = cast("Grid", self)
        zone_grid = zone_grid.type("int32")
        result = self
        for zone in set(zone_grid.data[np.isfinite(zone_grid.data)]):
            zone_value = int(zone + 0.5)
            data = g.set_nan(zone_grid != zone_value).data
            statistics = func(data[np.isfinite(data)])
            result = (zone_grid == zone_value).con(statistics, result)  # type: ignore
        return cast("Grid", result)

    def zonal_count(self, value: Union[float, int], zone_grid: "Grid", **kwargs):
        return self.zonal(lambda a: np.count_nonzero(a == value, **kwargs), zone_grid)

    def zonal_ptp(self, zone_grid: "Grid", **kwargs):
        return self.zonal(lambda a: np.ptp(a, **kwargs), zone_grid)

    def zonal_percentile(self, percentile: float, zone_grid: "Grid", **kwargs):
        return self.zonal(lambda a: np.percentile(a, percentile, **kwargs), zone_grid)

    def zonal_quantile(self, probability: float, zone_grid: "Grid", **kwargs):
        return self.zonal(lambda a: np.quantile(a, probability, **kwargs), zone_grid)

    def zonal_median(self, zone_grid: "Grid", **kwargs):
        return self.zonal(lambda a: np.median(a, **kwargs), zone_grid)

    def zonal_mean(self, zone_grid: "Grid", **kwargs):
        return self.zonal(lambda a: np.mean(a, **kwargs), zone_grid)

    def zonal_std(self, zone_grid: "Grid", **kwargs):
        return self.zonal(lambda a: np.std(a, **kwargs), zone_grid)

    def zonal_var(self, zone_grid: "Grid", **kwargs):
        return self.zonal(lambda a: np.var(a, **kwargs), zone_grid)

    def zonal_min(self, zone_grid: "Grid", **kwargs):
        return self.zonal(lambda a: np.min(a, **kwargs), zone_grid)

    def zonal_max(self, zone_grid: "Grid", **kwargs):
        return self.zonal(lambda a: np.max(a, **kwargs), zone_grid)

    def zonal_sum(self, zone_grid: "Grid", **kwargs):
        return self.zonal(lambda a: np.sum(a, **kwargs), zone_grid)

    def zonal_entropy(self, zone_grid: "Grid", **kwargs):
        return self.zonal(lambda a: sp.stats.entropy(a, **kwargs), zone_grid)

    def zonal_gmean(self, zone_grid: "Grid", **kwargs):
        return self.zonal(lambda a: sp.stats.gmean(a, **kwargs), zone_grid)

    def zonal_hmean(self, zone_grid: "Grid", **kwargs):
        return self.zonal(lambda a: sp.stats.hmean(a, **kwargs), zone_grid)

    def zonal_pmean(self, p: float, zone_grid: "Grid", **kwargs):
        return self.zonal(lambda a: sp.stats.pmean(a, p, **kwargs), zone_grid)

    def zonal_kurtosis(self, zone_grid: "Grid", **kwargs):
        return self.zonal(lambda a: sp.stats.kurtosis(a, **kwargs), zone_grid)

    def zonal_iqr(self, zone_grid: "Grid", **kwargs):
        return self.zonal(lambda a: sp.stats.iqr(a, **kwargs), zone_grid)

    def zonal_moment(self, zone_grid: "Grid", **kwargs):
        return self.zonal(lambda a: sp.stats.moment(a, **kwargs), zone_grid)

    def zonal_skew(self, zone_grid: "Grid", **kwargs):
        return self.zonal(lambda a: sp.stats.skew(a, **kwargs), zone_grid)

    def zonal_kstat(self, zone_grid: "Grid", **kwargs):
        return self.zonal(lambda a: sp.stats.kstat(a, **kwargs), zone_grid)

    def zonal_kstatvar(self, zone_grid: "Grid", **kwargs):
        return self.zonal(lambda a: sp.stats.kstatvar(a, **kwargs), zone_grid)

    def zonal_tmean(self, zone_grid: "Grid", **kwargs):
        return self.zonal(lambda a: sp.stats.tmean(a, **kwargs), zone_grid)

    def zonal_tvar(self, zone_grid: "Grid", **kwargs):
        return self.zonal(lambda a: sp.stats.tvar(a, **kwargs), zone_grid)

    def zonal_tmin(self, zone_grid: "Grid", **kwargs):
        return self.zonal(lambda a: sp.stats.tmin(a, **kwargs), zone_grid)

    def zonal_tmax(self, zone_grid: "Grid", **kwargs):
        return self.zonal(lambda a: sp.stats.tmax(a, **kwargs), zone_grid)

    def zonal_tstd(self, zone_grid: "Grid", **kwargs):
        return self.zonal(lambda a: sp.stats.tstd(a, **kwargs), zone_grid)

    def zonal_variation(self, zone_grid: "Grid", **kwargs):
        return self.zonal(lambda a: sp.stats.variation(a, **kwargs), zone_grid)

    def zonal_median_abs_deviation(self, zone_grid: "Grid", **kwargs):
        return self.zonal(
            lambda a: sp.stats.median_abs_deviation(a, **kwargs), zone_grid
        )
