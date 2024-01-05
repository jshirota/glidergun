import numpy as np
import pickle
import sklearn.decomposition
import sklearn.ensemble
import sklearn.gaussian_process
import sklearn.linear_model
import sklearn.naive_bayes
import sklearn.neighbors
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.svm
import sklearn.tree
import typing
from glidergun.core import Grid, _standardize, con
from glidergun.literals import DataType


def pca(n_components: int = 1, *grids: Grid) -> typing.Tuple[Grid, ...]:
    grids_adjusted = [con(g.is_nan(), g.mean, g) for g in _standardize(True, *grids)]
    arrays = (
        sklearn.decomposition.PCA(n_components=n_components)
        .fit_transform(
            np.array([g.scale().data.ravel() for g in grids_adjusted]).transpose((1, 0))
        )
        .transpose((1, 0))
    )
    grid = grids_adjusted[0]
    return tuple(grid._create(a.reshape((grid.height, grid.width))) for a in arrays)


class Model(typing.Protocol):
    fit: typing.Callable
    score: typing.Callable
    predict: typing.Callable


T = typing.TypeVar("T", bound=Model)


class Prediction(typing.Generic[T]):
    def __init__(self, model: T) -> None:
        self.model: T = model
        self._dtype: DataType = "float32"

    def fit(self, dependent_grid: Grid, *explanatory_grids: Grid):
        head, *tail = self._flatten(*[dependent_grid, *explanatory_grids])
        self.model = self.model.fit(
            np.array([g.data.ravel() for g in tail]).transpose(1, 0),
            head.data.ravel(),
        )
        self._dtype = dependent_grid.dtype
        return self

    def score(self, dependent_grid: Grid, *explanatory_grids: Grid) -> float:
        head, *tail = self._flatten(dependent_grid, *explanatory_grids)
        return self.model.score(
            np.array([g.data.ravel() for g in tail]).transpose(1, 0), head.data.ravel()
        )

    def predict(self, *explanatory_grids: Grid) -> Grid:
        grids = self._flatten(*explanatory_grids)
        array = self.model.predict(
            np.array([g.data.ravel() for g in grids]).transpose(1, 0)
        )
        grid = grids[0]
        return grid._create(array.reshape((grid.height, grid.width))).type(self._dtype)

    def _flatten(self, *grids: Grid):
        return [con(g.is_nan(), g.mean, g) for g in _standardize(True, *grids)]

    def save(self, file: str):
        with open(file, "wb") as f:
            pickle.dump(self.model, f)


def load(file: str) -> Prediction[typing.Any]:
    with open(file, "rb") as f:
        return Prediction(pickle.load(f))


def fit(model: T, dependent_grid: Grid, *explanatory_grids: Grid) -> Prediction[T]:
    return Prediction(model).fit(dependent_grid, *explanatory_grids)


def decision_tree_classification(
    dependent_grid: Grid, *explanatory_grids: Grid, **kwargs
):
    model = sklearn.tree.DecisionTreeClassifier(**kwargs)
    return fit(model, dependent_grid, *explanatory_grids)


def decision_tree_regression(dependent_grid: Grid, *explanatory_grids: Grid, **kwargs):
    model = sklearn.tree.DecisionTreeRegressor(**kwargs)
    return fit(model, dependent_grid, *explanatory_grids)


def elastic_net_regression(dependent_grid: Grid, *explanatory_grids: Grid, **kwargs):
    model = sklearn.linear_model.ElasticNet(**kwargs)
    return fit(model, dependent_grid, *explanatory_grids)


def gaussian_naive_bayes_classification(
    dependent_grid: Grid, *explanatory_grids: Grid, **kwargs
):
    model = sklearn.naive_bayes.GaussianNB(**kwargs)
    return fit(model, dependent_grid, *explanatory_grids)


def gradient_boosting_classification(
    dependent_grid: Grid, *explanatory_grids: Grid, **kwargs
):
    model = sklearn.ensemble.GradientBoostingClassifier(**kwargs)
    return fit(model, dependent_grid, *explanatory_grids)


def lasso_regression(dependent_grid: Grid, *explanatory_grids: Grid, **kwargs):
    model = sklearn.linear_model.Lasso(**kwargs)
    return fit(model, dependent_grid, *explanatory_grids)


def linear_regression(dependent_grid: Grid, *explanatory_grids: Grid, **kwargs):
    model = sklearn.linear_model.LinearRegression(**kwargs)
    return fit(model, dependent_grid, *explanatory_grids)


def logistic_classification(dependent_grid: Grid, *explanatory_grids: Grid, **kwargs):
    model = sklearn.linear_model.LogisticRegression(**kwargs)
    return fit(model, dependent_grid, *explanatory_grids)


def polynomial_regression(dependent_grid: Grid, *explanatory_grids: Grid, **kwargs):
    model = sklearn.pipeline.make_pipeline(
        sklearn.preprocessing.PolynomialFeatures(**kwargs),
        sklearn.linear_model.LinearRegression(),
    )
    return fit(model, dependent_grid, *explanatory_grids)


def random_forest_classification(
    dependent_grid: Grid, *explanatory_grids: Grid, **kwargs
):
    model = sklearn.ensemble.RandomForestClassifier(**kwargs)
    return fit(model, dependent_grid, *explanatory_grids)


def random_forest_regression(dependent_grid: Grid, *explanatory_grids: Grid, **kwargs):
    model = sklearn.ensemble.RandomForestRegressor(**kwargs)
    return fit(model, dependent_grid, *explanatory_grids)


def ridge_regression(dependent_grid: Grid, *explanatory_grids: Grid, **kwargs):
    model = sklearn.linear_model.Ridge(**kwargs)
    return fit(model, dependent_grid, *explanatory_grids)


def support_vector_classification(
    dependent_grid: Grid, *explanatory_grids: Grid, **kwargs
):
    model = sklearn.pipeline.make_pipeline(
        sklearn.preprocessing.StandardScaler(), sklearn.svm.SVC(**kwargs)
    )
    return fit(model, dependent_grid, *explanatory_grids)
