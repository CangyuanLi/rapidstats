from pathlib import Path

import polars as pl
import pybench
import sklearn.metrics

import rapidstats as rs

BASE_PATH = Path(__file__).resolve().parents[0]


SEED = 208
DF = pl.read_parquet(BASE_PATH / "benchmark_data.parquet")
SIZES = [1_000, 10_000, 100_000, 500_000]


def sample_df(n):
    return {"df": DF.sample(n, seed=SEED)}


@pybench.metadata(group="confusion_matrix")
@pybench.parametrize({"n": SIZES}, setup=sample_df)
def bench_confusion_matrix(df):
    rs.metrics.confusion_matrix(df["y_true"], df["y_pred"])


@pybench.metadata(group="confusion_matrix")
@pybench.parametrize({"n": SIZES}, setup=sample_df)
def bench_sklearn_confusion_matrix(df):
    sklearn.metrics.confusion_matrix(df["y_true"], df["y_pred"])


@pybench.metadata(group="roc_auc")
@pybench.parametrize({"n": SIZES}, setup=sample_df)
def bench_roc_auc(df):
    rs.metrics.roc_auc(df["y_true"], df["y_score"])


@pybench.metadata(group="roc_auc")
@pybench.parametrize({"n": SIZES}, setup=sample_df)
def bench_sklearn_roc_auc(df):
    sklearn.metrics.roc_auc_score(df["y_true"], df["y_score"])


@pybench.metadata(group="mean_squared_error")
@pybench.parametrize({"n": SIZES}, setup=sample_df)
def bench_mean_squared_error(df):
    rs.metrics.mean_squared_error(df["y_score"], df["y_score"])


@pybench.metadata(group="mean_squared_error")
@pybench.parametrize({"n": SIZES}, setup=sample_df)
def bench_sklearn_mean_squared_error(df):
    sklearn.metrics.mean_squared_error(df["y_score"], df["y_score"])


@pybench.metadata(group="r2")
@pybench.parametrize({"n": SIZES}, setup=sample_df)
def bench_r2(df):
    rs.metrics.r2(df["y_score"], df["y_score"])


@pybench.metadata(group="r2")
@pybench.parametrize({"n": SIZES}, setup=sample_df)
def bench_sklearn_r2(df):
    sklearn.metrics.r2_score(df["y_score"], df["y_score"])


def sample_df_thresh(n, n_thresholds, **kwargs):
    thresholds = (
        None
        if n_thresholds == n or n_thresholds is None
        else [t / 100 for t in range(1, n_thresholds + 1)]
    )

    return {"df": DF.sample(n, seed=SEED), "thresholds": thresholds} | kwargs


@pybench.metadata(group="confusion_matrix_at_thresholds")
@pybench.config(repeat=1)
@pybench.parametrize(
    {
        "n": [1_000, 50_000],
        "n_thresholds": [5, 10, 100, 500],
        "strategy": ["loop", "cum_sum"],
    },
    setup=sample_df_thresh,
)
def bench_confusion_matrix_at_thresholds(df, thresholds, strategy):
    rs.metrics.confusion_matrix_at_thresholds(
        df["y_true"],
        df["y_score"],
        thresholds=thresholds,
        strategy=strategy,
    )


@pybench.metadata(group="confusion_matrix_at_thresholds")
@pybench.config(repeat=1)
@pybench.parametrize(
    {
        "n": [1_000, 50_000],
        "n_thresholds": [5, 10, 100, 500],
    },
    setup=sample_df_thresh,
)
def bench_sklearn_confusion_matrix_at_thresholds(df, thresholds):
    if thresholds is None:
        thresholds = df["y_score"].unique()

    nan = float("nan")

    yt = df["y_true"]
    for t in thresholds:
        yp = df["y_score"] >= t

        sklearn.metrics.fbeta_score(yt, yp, beta=1)
        sklearn.metrics.accuracy_score(yt, yp)
        sklearn.metrics.matthews_corrcoef(yt, yp)
        sklearn.metrics.balanced_accuracy_score(yt, yp)
        sklearn.metrics.confusion_matrix(yt, yp)
        sklearn.metrics.fowlkes_mallows_score(yt, yp)
        sklearn.metrics.precision_score(yt, yp, zero_division=nan)
        sklearn.metrics.recall_score(yt, yp)
