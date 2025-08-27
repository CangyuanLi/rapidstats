import polars as pl
import polars.testing
import pytest

import rapidstats as rs
from tests.paths import DAT_PATH

kwargs = {
    "seed": 208,
    "iterations": 1000,
    "chunksize": 8,
    # "n_jobs": 1,
}

POISSON_BOOTSTRAP = rs.Bootstrap(sampling_method="poisson", **kwargs)
MULTINOMIAL_BOOTSTRAP = rs.Bootstrap(sampling_method="multinomial", **kwargs)
SCORES = pl.read_parquet(DAT_PATH / "scores.parquet")
REL_TOL = 1e-1


def test_roc_auc():
    y_true = SCORES["y_true"]
    y_score = SCORES["y_score"]

    p_res = POISSON_BOOTSTRAP.roc_auc(y_true, y_score)
    m_res = MULTINOMIAL_BOOTSTRAP.roc_auc(y_true, y_score)

    assert pytest.approx(p_res, REL_TOL) == m_res


def test_run():
    p_res = POISSON_BOOTSTRAP.run(SCORES, lambda df: rs.metrics.mean(df["y_score"]))
    m_res = MULTINOMIAL_BOOTSTRAP.run(SCORES, lambda df: rs.metrics.mean(df["y_score"]))

    assert pytest.approx(p_res, REL_TOL) == m_res


def test_confusion_matrix():
    y_true = SCORES["y_true"]
    y_pred = SCORES["y_score"].ge(0.5)

    p_res = (
        POISSON_BOOTSTRAP.confusion_matrix(y_true, y_pred).to_polars().sort("metric")
    )
    m_res = (
        MULTINOMIAL_BOOTSTRAP.confusion_matrix(y_true, y_pred)
        .to_polars()
        .sort("metric")
    )

    polars.testing.assert_frame_equal(p_res, m_res, rtol=REL_TOL)


# def test_confusion_matrix_at_thresholds():
#     tmp = SCORES
#     y_true = tmp["y_true"]
#     y_score = tmp["y_score"]

#     p_res = POISSON_BOOTSTRAP.confusion_matrix_at_thresholds(y_true, y_score).sort(
#         "threshold", "metric"
#     )
#     m_res = MULTINOMIAL_BOOTSTRAP.confusion_matrix_at_thresholds(y_true, y_score).sort(
#         "threshold", "metric"
#     )

#     polars.testing.assert_frame_equal(p_res, m_res, rtol=REL_TOL)
