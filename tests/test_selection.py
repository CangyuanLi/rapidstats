import inspect
import tempfile

import catboost
import lightgbm
import numpy as np
import numpy.typing as npt
import polars as pl
import pytest
import xgboost
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression

import rapidstats as rs

N_ROWS = 50_000
SEED = 208
N_ESTIMATORS = 10

ESTIMATORS = [
    catboost.CatBoostClassifier(
        iterations=N_ESTIMATORS,
        verbose=False,
        random_state=SEED,
        allow_writing_files=False,
    ),
    lightgbm.LGBMClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=SEED,
        verbose=-1,
        importance_type="gain",
    ),
    xgboost.XGBClassifier(n_estimators=N_ESTIMATORS, random_state=SEED, verbosity=0),
    GradientBoostingClassifier(n_estimators=N_ESTIMATORS, random_state=SEED, verbose=0),
    RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=SEED, verbose=0),
    catboost.CatBoostRegressor(
        iterations=N_ESTIMATORS,
        verbose=False,
        random_state=SEED,
        allow_writing_files=False,
    ),
    lightgbm.LGBMRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=SEED,
        verbose=-1,
        importance_type="gain",
    ),
    xgboost.XGBRegressor(n_estimators=N_ESTIMATORS, random_state=SEED, verbosity=0),
    GradientBoostingRegressor(n_estimators=N_ESTIMATORS, random_state=SEED, verbose=0),
    RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=SEED, verbose=0),
    LinearRegression(),
    LogisticRegression(random_state=SEED),
]

np.random.seed(SEED)


def _create_correlated_vector(target: npt.NDArray, rho: float) -> npt.NDArray:
    mean_target = np.mean(target)
    std_target = np.std(target)
    n = len(target)

    target_std = (target.astype(float) - mean_target) / std_target

    z = np.random.randn(n)

    factor = np.sqrt(1 - rho**2)
    y = rho * target_std + factor * z

    result = y * std_target + mean_target

    return result


rhos = [0.01, 0.001, 0.001, 0.99]
y = np.random.choice([0, 1], N_ROWS)
X = pl.DataFrame(
    {f"f{rho}": _create_correlated_vector(y, rho) for rho in rhos}
).to_pandas()


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_rfe(estimator):
    fit_kwargs = {}
    if "eval_set" in inspect.signature(estimator.fit).parameters:
        fit_kwargs["eval_set"] = [(X, y)]

    rfe = rs.selection.RFE(estimator=estimator, step=1, quiet=True).fit(
        X, y, **fit_kwargs
    )

    assert rfe.selected_features_ == ["f0.99"]
    assert rfe.transform(X).columns == ["f0.99"]


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_rfe_early_stopping(estimator):
    fit_kwargs = {}
    if "eval_set" in inspect.signature(estimator.fit).parameters:
        fit_kwargs["eval_set"] = [(X, y)]

    def _roc_auc(est, X, y) -> float:
        return rs.metrics.roc_auc(y, est.predict(X))

    early_stopping_kwargs = {}
    if "predict_proba" not in inspect.getmembers(
        estimator, predicate=inspect.isfunction
    ):
        early_stopping_kwargs["metric"] = _roc_auc

    rs.selection.RFE(
        estimator=estimator,
        step=3,
        quiet=True,
        callbacks=[rs.selection.EarlyStopping(**early_stopping_kwargs)],
    ).fit(X, y, **fit_kwargs)


def test_rfe_save_load():
    with tempfile.NamedTemporaryFile() as f:
        x = rs.selection.RFE(estimator=ESTIMATORS[0]).fit(X, y)
        x.save(f.name)

        ref = x.selected_features_

        res = rs.selection.RFE.load(f.name).selected_features_

        assert ref == res


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_nfe(estimator):
    fit_kwargs = {}
    if "eval_set" in inspect.signature(estimator.fit).parameters:
        fit_kwargs["eval_set"] = [(X, y)]

    nfe = rs.selection.NFE(estimator=estimator, seed=SEED).fit(X, y, **fit_kwargs)

    assert "f0.99" in nfe.selected_features_
    assert "f0.99" in nfe.transform(X).columns


def test_nfe_save_load():
    with tempfile.NamedTemporaryFile() as f:
        x = rs.selection.NFE(estimator=ESTIMATORS[0]).fit(X, y)
        x.save(f.name)

        ref = x.selected_features_

        res = rs.selection.NFE.load(f.name).selected_features_

        assert ref == res


def test_cfe():
    corr_mat = pl.DataFrame(
        {
            "": ["a", "b", "c"],
            "a": [1.0, 0.5, 0.7],
            "b": [-0.99, 1, 0.98],
            "c": [float("nan"), None, 1],
        }
    )

    expected = ["a", "c"]
    cfe = rs.selection.CFE(threshold=0.95)

    assert cfe.fit_from_correlation_matrix(corr_mat).selected_features_ == expected

    corr_mat_unpivoted = corr_mat.unpivot(index="").rename(
        {"": "c1", "variable": "c2", "value": "correlation"}
    )

    assert (
        cfe.fit_from_correlation_matrix(
            corr_mat_unpivoted, transform=False
        ).selected_features_
        == expected
    )


def test_cfe_identity_no_drop():
    # Test that identity correlations do not cause a feature to be removed, i.e.
    # corr(a, a) = 1 should not cause feature a to be dropped.
    corr_mat = pl.DataFrame(
        {
            "": ["a", "b", "c"],
            "a": [1.0, 0.5, 0.7],
            "b": [0.5, 1.0, 0.98],
            "c": [float("nan"), None, 1],
        }
    )

    assert rs.selection.CFE(threshold=0.99).fit_from_correlation_matrix(
        corr_mat
    ).selected_features_ == ["a", "b", "c"]


def test_cfe_corr_1_is_removed():
    # Test that a correlation of 1 that is not an identity causes a feature to be
    # correctly removed.
    corr_mat = pl.DataFrame(
        {
            "": ["a", "b", "c"],
            "a": [1.0, 0.5, 0.7],
            "b": [0.5, 1.0, 1.0],
            "c": [float("nan"), None, 1],
        }
    )

    assert rs.selection.CFE(threshold=0.99).fit_from_correlation_matrix(
        corr_mat
    ).selected_features_ == ["a", "c"]


def test_cfe_repro():
    n_cols = 50
    df = pl.DataFrame(
        np.random.rand(1_000, n_cols), schema=[f"col_{i}" for i in range(n_cols)]
    )

    assert (
        rs.selection.CFE().fit(df).selected_features_
        == rs.selection.CFE().fit(df).selected_features_
    )


def test_cfe_save_load():
    with tempfile.NamedTemporaryFile() as f:
        x = rs.selection.CFE().fit(X)
        x.save(f.name)

        ref = x.selected_features_

        res = rs.selection.CFE.load(f.name).selected_features_

        assert ref == res
