from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union, cast

import numpy as np
from scipy.interpolate import (
    LinearNDInterpolator,
    NearestNDInterpolator,
    RBFInterpolator,
)

from glidergun._literals import InterpolationKernel

if TYPE_CHECKING:
    from glidergun._grid import Grid


@dataclass(frozen=True)
class Interpolation:
    def interp_linear(
        self,
        cell_size: Union[Tuple[float, float], float, None] = None,
        fill_value: float = np.nan,
        rescale: bool = False,
    ):
        from glidergun._functions import interpolate

        def f(coords, values):
            return LinearNDInterpolator(coords, values, fill_value, rescale)

        g = cast("Grid", self)
        return interpolate(f, g.to_points(), g.extent, g.crs, cell_size or g.cell_size)

    def interp_nearest(
        self,
        cell_size: Union[Tuple[float, float], float, None] = None,
        rescale: bool = False,
        tree_options: Any = None,
    ):
        from glidergun._functions import interpolate

        def f(coords, values):
            return NearestNDInterpolator(coords, values, rescale, tree_options)

        g = cast("Grid", self)
        return interpolate(f, g.to_points(), g.extent, g.crs, cell_size or g.cell_size)

    def interp_rbf(
        self,
        cell_size: Union[Tuple[float, float], float, None] = None,
        neighbors: Optional[int] = None,
        smoothing: float = 0,
        kernel: InterpolationKernel = "thin_plate_spline",
        epsilon: float = 1,
        degree: Optional[int] = None,
    ):
        from glidergun._functions import interpolate

        def f(coords, values):
            return RBFInterpolator(
                coords, values, neighbors, smoothing, kernel, epsilon, degree
            )

        g = cast("Grid", self)
        return interpolate(f, g.to_points(), g.extent, g.crs, cell_size or g.cell_size)
