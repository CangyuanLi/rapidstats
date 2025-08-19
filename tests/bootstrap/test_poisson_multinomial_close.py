import polars as pl
import pytest

import rapidstats as rs
from tests.paths import DAT_PATH

kwargs = {
    "seed": 208,
    "iterations": 1_000,
    "chunksize": 8,
    # "n_jobs": 1,
}

POISSON_BOOTSTRAP = rs.Bootstrap(sampling_method="poisson", **kwargs)
MULTINOMIAL_BOOTSTRAP = rs.Bootstrap(sampling_method="multinomial", **kwargs)
IRIS_SCORES = pl.read_parquet(DAT_PATH / "iris_scores.parquet")
REL_TOL = 1e-2


def test_roc_auc():
    y_true = IRIS_SCORES["y_true"]
    y_score = IRIS_SCORES["y_score"]

    p_res = POISSON_BOOTSTRAP.roc_auc(y_true, y_score)
    m_res = MULTINOMIAL_BOOTSTRAP.roc_auc(y_true, y_score)

    assert pytest.approx(p_res, REL_TOL) == m_res


def test_run():
    p_res = POISSON_BOOTSTRAP.run(
        IRIS_SCORES, lambda df: rs.metrics.mean(df["y_score"])
    )
    m_res = MULTINOMIAL_BOOTSTRAP.run(
        IRIS_SCORES, lambda df: rs.metrics.mean(df["y_score"])
    )

    assert pytest.approx(p_res, REL_TOL) == m_res


# def test_confusion_matrix_at_thresholds():
#     y_true = IRIS_SCORES["y_true"]
#     y_score = IRIS_SCORES["y_score"]

#     p_res = POISSON_BOOTSTRAP.confusion_matrix_at_thresholds(y_true, y_score)
#     m_res = MULTINOMIAL_BOOTSTRAP.confusion_matrix_at_thresholds(y_true, y_score)
