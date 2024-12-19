from pathlib import Path

import polars as pl
import pybench
import sklearn.metrics

import rapidstats

BASE_PATH = Path(__file__).resolve().parents[0]


SEED = 208
DF = pl.read_parquet(BASE_PATH / "benchmark_data.parquet")
SIZES = [100, 1_000, 10_000, 100_000, 500_000, DF.height]


def sample_df(n):
    return {"df": DF.sample(n, seed=SEED)}


@pybench.parametrize({"n": SIZES}, setup=sample_df)
def bench_confusion_matrix(df):
    rapidstats.confusion_matrix(df["y_true"], df["y_pred"])


@pybench.parametrize({"n": SIZES}, setup=sample_df)
def bench_sklearn_confusion_matrix(df):
    sklearn.metrics.confusion_matrix(df["y_true"], df["y_pred"])


@pybench.parametrize({"n": SIZES}, setup=sample_df)
def bench_roc_auc(df):
    rapidstats.roc_auc(df["y_true"], df["y_score"])


@pybench.parametrize({"n": SIZES}, setup=sample_df)
def bench_sklearn_roc_auc(df):
    sklearn.metrics.roc_auc_score(df["y_true"], df["y_score"])


@pybench.parametrize({"n": SIZES}, setup=sample_df)
def bench_mean_squared_error(df):
    rapidstats.mean_squared_error(df["y_score"], df["y_score"])


@pybench.parametrize({"n": SIZES}, setup=sample_df)
def bench_sklearn_mean_squared_error(df):
    sklearn.metrics.mean_squared_error(df["y_score"], df["y_score"])


def sample_df_thresh(n, n_thresholds):
    thresholds = (
        None if n_thresholds == n else [t / 100 for t in range(1, n_thresholds + 1)]
    )

    return {"df": DF.sample(n, seed=SEED), "thresholds": thresholds}


@pybench.parametrize(
    {"n": [1_000, 10_000, 50_000], "n_thresholds": [5, 10, 100]},
    setup=sample_df_thresh,
)
def bench_confusion_matrix_at_thresholds_loop(df, thresholds):

    rapidstats.confusion_matrix_at_thresholds(
        df["y_true"], df["y_score"], thresholds=thresholds, strategy="loop"
    )


@pybench.parametrize(
    {"n": [1_000, 10_000, 50_000], "n_thresholds": [5, 10, 100]},
    setup=sample_df_thresh,
)
def bench_confusion_matrix_at_thresholds_cum_sum(df, thresholds):
    rapidstats.confusion_matrix_at_thresholds(
        df["y_true"], df["y_score"], thresholds=thresholds, strategy="cum_sum"
    )
