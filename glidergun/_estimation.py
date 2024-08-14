import numpy as np
import pickle
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Generic, Optional, Protocol, TypeVar, Union, cast
from glidergun._literals import DataType, EstimatorType

if TYPE_CHECKING:
    from glidergun._grid import Grid


class Estimator(Protocol):
    fit: Callable
    predict: Callable


TEstimator = TypeVar("TEstimator", bound=Estimator)


@dataclass(frozen=True)
class Estimation:
    def fit(self, model: Union[TEstimator, EstimatorType], *explanatory_grids: "Grid", **kwargs: Any):
        if model == "linear_regression":
            from sklearn.linear_model import LinearRegression
            actual_model = LinearRegression()
        elif model == "polynomial_regression":
            from sklearn.linear_model import LinearRegression
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import PolynomialFeatures
            actual_model = Pipeline([("polynomial_features", PolynomialFeatures(degree=2, include_bias=False)),
                                     ("linear_regression", LinearRegression())])
        elif model == "random_forest_classifier":
            from sklearn.ensemble import RandomForestClassifier
            actual_model = RandomForestClassifier()
        elif model == "random_forest_regression":
            from sklearn.ensemble import RandomForestRegressor
            actual_model = RandomForestRegressor()
        else:
            actual_model = model

        if isinstance(actual_model, str):
            raise ValueError(f"'{actual_model}' is not a supported estimator.")

        return GridEstimator(actual_model).fit(cast("Grid", self), *explanatory_grids, **kwargs)


class GridEstimator(Generic[TEstimator]):
    def __init__(self, model: TEstimator) -> None:
        self.model: TEstimator = model
        self._dtype: DataType = "float32"

    def fit(self, dependent_grid: "Grid", *explanatory_grids: "Grid", **kwargs: Any):
        grids = self._flatten(*[dependent_grid, *explanatory_grids])
        head, *tail = grids
        self.model = self.model.fit(
            np.array([g.data.ravel() for g in tail]).transpose(1, 0),
            head.data.ravel(), **kwargs
        )
        self._dtype = dependent_grid.dtype
        return self

    def score(self, dependent_grid: "Grid", *explanatory_grids: "Grid") -> Optional[float]:
        score = getattr(self.model, "score", None)
        if score:
            head, *tail = self._flatten(dependent_grid, *explanatory_grids)
            return score(
                np.array([g.data.ravel() for g in tail]).transpose(
                    1, 0), head.data.ravel()
            )

    def predict(self, *explanatory_grids: "Grid", **kwargs: Any) -> "Grid":
        grids = self._flatten(*explanatory_grids)
        array = self.model.predict(
            np.array([g.data.ravel() for g in grids]).transpose(1, 0), **kwargs
        )
        g = grids[0]
        return g.update(array.reshape((g.height, g.width))).type(self._dtype)

    def _flatten(self, *grids: "Grid"):
        return [g.is_nan().con(float(g.mean), g) for g in grids[0].standardize(*grids[1:])]

    def save(self, file: str):
        with open(file, "wb") as f:
            pickle.dump(self.model, f)

    @classmethod
    def load(cls, file: str):
        with open(file, "rb") as f:
            return GridEstimator(pickle.load(f))


def load_model(file: str) -> GridEstimator[Any]:
    return GridEstimator.load(file)
