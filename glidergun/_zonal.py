from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy import ndarray

if TYPE_CHECKING:
    from glidergun._grid import Grid


@dataclass(frozen=True)
class Zonal:
    def zonal(self, func: Callable[[ndarray], Any], zone_grid: "Grid") -> "Grid":
        """Apply a statistic over values grouped by zones.

        Args:
            func: Function applied to values within each zone.
            zone_grid: Integer grid defining zone IDs.

        Returns:
            Grid: Grid where each zone's cells are assigned the computed statistic.
        """
        self = cast("Grid", self)
        zone_grid = zone_grid.type("int32")
        result = self
        for zone in set(zone_grid.data[np.isfinite(zone_grid.data)]):
            zone_value = int(zone + 0.5)
            data = self.set_nan(zone_grid != zone_value).data
            statistics = func(data[np.isfinite(data)])
            result = (zone_grid == zone_value).then(statistics, result)  # type: ignore
        return cast("Grid", result)

    def zonal_count(self, value: float | int, zone_grid: "Grid", **kwargs):
        """Count of `value` within each zone.
        Args:
            value: Value to count.
            zone_grid: Integer grid defining zone IDs.
            **kwargs: Additional args for `np.count_nonzero`.

        Returns:
            Grid: Count per zone.
        """
        return self.zonal(lambda a: np.count_nonzero(a == value, **kwargs), zone_grid)

    def zonal_ptp(self, zone_grid: "Grid", **kwargs):
        """Peak-to-peak (max - min) within each zone.
        Args:
            zone_grid: Integer grid defining zone IDs.
            **kwargs: Additional args for `np.ptp`.

        Returns:
            Grid: Peak-to-peak per zone.
        """
        return self.zonal(lambda a: np.ptp(a, **kwargs), zone_grid)

    def zonal_median(self, zone_grid: "Grid", **kwargs):
        """Median within each zone.
        Args:
            zone_grid: Integer grid defining zone IDs.
            **kwargs: Additional args for `np.median`.

        Returns:
            Grid: Median per zone.
        """
        return self.zonal(lambda a: np.median(a, **kwargs), zone_grid)

    def zonal_mean(self, zone_grid: "Grid", **kwargs):
        """Mean within each zone.
        Args:
            zone_grid: Integer grid defining zone IDs.
            **kwargs: Additional args for `np.mean`.

        Returns:
            Grid: Mean per zone.
        """
        return self.zonal(lambda a: np.mean(a, **kwargs), zone_grid)

    def zonal_std(self, zone_grid: "Grid", **kwargs):
        """Standard deviation within each zone.
        Args:
            zone_grid: Integer grid defining zone IDs.
            **kwargs: Additional args for `np.std`.

        Returns:
            Grid: Standard deviation per zone.
        """
        return self.zonal(lambda a: np.std(a, **kwargs), zone_grid)

    def zonal_var(self, zone_grid: "Grid", **kwargs):
        """Variance within each zone.
        Args:
            zone_grid: Integer grid defining zone IDs.
            **kwargs: Additional args for `np.var`.

        Returns:
            Grid: Variance per zone.
        """
        return self.zonal(lambda a: np.var(a, **kwargs), zone_grid)

    def zonal_min(self, zone_grid: "Grid", **kwargs):
        """Minimum within each zone.
        Args:
            zone_grid: Integer grid defining zone IDs.
            **kwargs: Additional args for `np.min`.

        Returns:
            Grid: Minimum per zone.
        """
        return self.zonal(lambda a: np.min(a, **kwargs), zone_grid)

    def zonal_max(self, zone_grid: "Grid", **kwargs):
        """Maximum within each zone.
        Args:
            zone_grid: Integer grid defining zone IDs.
            **kwargs: Additional args for `np.max`.

        Returns:
            Grid: Maximum per zone.
        """
        return self.zonal(lambda a: np.max(a, **kwargs), zone_grid)

    def zonal_sum(self, zone_grid: "Grid", **kwargs):
        """Sum within each zone.
        Args:
            zone_grid: Integer grid defining zone IDs.
            **kwargs: Additional args for `np.sum`.

        Returns:
            Grid: Sum per zone.
        """
        return self.zonal(lambda a: np.sum(a, **kwargs), zone_grid)
