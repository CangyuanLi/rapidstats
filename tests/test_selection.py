import inspect

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


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_nfe(estimator):
    fit_kwargs = {}
    if "eval_set" in inspect.signature(estimator.fit).parameters:
        fit_kwargs["eval_set"] = [(X, y)]

    nfe = rs.selection.NFE(estimator=estimator, seed=SEED).fit(X, y, **fit_kwargs)

    assert "f0.99" in nfe.selected_features_
