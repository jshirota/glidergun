import numpy as np
import typing
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
from glidergun.core import Grid, _standardize, con


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


def decision_tree_classification(
    dependent_grid: Grid, *explanatory_grids: Grid, **kwargs
):
    model = sklearn.tree.DecisionTreeClassifier(**kwargs)
    return dependent_grid.fit(model, *explanatory_grids)


def decision_tree_regression(dependent_grid: Grid, *explanatory_grids: Grid, **kwargs):
    model = sklearn.tree.DecisionTreeRegressor(**kwargs)
    return dependent_grid.fit(model, *explanatory_grids)


def elastic_net_regression(dependent_grid: Grid, *explanatory_grids: Grid, **kwargs):
    model = sklearn.linear_model.ElasticNet(**kwargs)
    return dependent_grid.fit(model, *explanatory_grids)


def gaussian_naive_bayes_classification(
    dependent_grid: Grid, *explanatory_grids: Grid, **kwargs
):
    model = sklearn.naive_bayes.GaussianNB(**kwargs)
    return dependent_grid.fit(model, *explanatory_grids)


def gradient_boosting_classification(
    dependent_grid: Grid, *explanatory_grids: Grid, **kwargs
):
    model = sklearn.ensemble.GradientBoostingClassifier(**kwargs)
    return dependent_grid.fit(model, *explanatory_grids)


def lasso_regression(dependent_grid: Grid, *explanatory_grids: Grid, **kwargs):
    model = sklearn.linear_model.Lasso(**kwargs)
    return dependent_grid.fit(model, *explanatory_grids)


def linear_regression(dependent_grid: Grid, *explanatory_grids: Grid, **kwargs):
    model = sklearn.linear_model.LinearRegression(**kwargs)
    return dependent_grid.fit(model, *explanatory_grids)


def logistic_classification(dependent_grid: Grid, *explanatory_grids: Grid, **kwargs):
    model = sklearn.linear_model.LogisticRegression(**kwargs)
    return dependent_grid.fit(model, *explanatory_grids)


def polynomial_regression(dependent_grid: Grid, *explanatory_grids: Grid, **kwargs):
    model = sklearn.pipeline.make_pipeline(
        sklearn.preprocessing.PolynomialFeatures(**kwargs),
        sklearn.linear_model.LinearRegression(),
    )
    return dependent_grid.fit(model, *explanatory_grids)


def random_forest_classification(
    dependent_grid: Grid, *explanatory_grids: Grid, **kwargs
):
    model = sklearn.ensemble.RandomForestClassifier(**kwargs)
    return dependent_grid.fit(model, *explanatory_grids)


def random_forest_regression(dependent_grid: Grid, *explanatory_grids: Grid, **kwargs):
    model = sklearn.ensemble.RandomForestRegressor(**kwargs)
    return dependent_grid.fit(model, *explanatory_grids)


def ridge_regression(dependent_grid: Grid, *explanatory_grids: Grid, **kwargs):
    model = sklearn.linear_model.Ridge(**kwargs)
    return dependent_grid.fit(model, *explanatory_grids)


def support_vector_classification(
    dependent_grid: Grid, *explanatory_grids: Grid, **kwargs
):
    model = sklearn.pipeline.make_pipeline(
        sklearn.preprocessing.StandardScaler(), sklearn.svm.SVC(**kwargs)
    )
    return dependent_grid.fit(model, *explanatory_grids)
