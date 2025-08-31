from pathlib import Path

import polars as pl
import pybench
import scipy.stats
import sklearn.metrics

import rapidstats as rs

BASE_PATH = Path(__file__).resolve().parents[0]


ITERATIONS = 100
METHOD = "percentile"
SEED = 208
BS = rs.Bootstrap(iterations=ITERATIONS, method=METHOD, seed=SEED)
BS_MULTINOMIAL = rs.Bootstrap(
    iterations=ITERATIONS, method=METHOD, sampling_method="multinomial", seed=SEED
)

DF = (
    pl.scan_parquet(BASE_PATH / "benchmark_data.parquet")
    .head(25_000)
    .select("y_true", "y_score")
    .collect()
)

DF_SMALL = DF.head(1_000)


@pybench.metadata(group="bootstrap_roc_auc")
@pybench.config(repeat=15)
def bench_bootstrap_poisson_roc_auc():
    BS.roc_auc(DF["y_true"], DF["y_score"])


@pybench.metadata(group="bootstrap_roc_auc")
@pybench.config(repeat=15)
def bench_bootstrap_multinomial_roc_auc():
    BS_MULTINOMIAL.roc_auc(DF["y_true"], DF["y_score"])


@pybench.metadata(group="bootstrap_roc_auc")
@pybench.config(repeat=10)
def bench_sklearn_bootstrap_roc_auc():
    data = (DF["y_true"], DF["y_score"])
    scipy.stats.bootstrap(
        data,
        statistic=sklearn.metrics.roc_auc_score,
        n_resamples=ITERATIONS,
        method=METHOD,
        random_state=SEED,
    )


# @pybench.metadata(group="bootstrap_roc_auc")
# @pybench.config(repeat=5)
# def bench_python_multithreaded_bootstrap_roc_auc():
#     with concurrent.futures.ThreadPoolExecutor() as pool:
#         point = rs.metrics.roc_auc(DF["y_true"], DF["y_score"])
#         yt = []
#         ys = []
#         for i in range(ITERATIONS):
#             _df = DF.sample(fraction=1, with_replacement=True, seed=SEED + i)
#             yt.append(_df["y_true"])
#             ys.append(_df["y_score"])

#         bootstrap_stats = list(pool.map(rs.metrics.roc_auc, yt, ys))
#         rs._bootstrap._percentile_interval(
#             point, bootstrap_stats=bootstrap_stats, alpha=(1 - 0.95) / 2
#         )


# @pybench.metadata(group="bootstrap_confusion_matrix_at_thresholds")
# @pybench.config(repeat=5)
# def bench_bootstrap_confusion_matrix_at_thresholds():
#     BS.confusion_matrix_at_thresholds(
#         y_true=DF_SMALL["y_true"], y_score=DF_SMALL["y_score"]
#     )


# @pybench.metadata(group="bootstrap_confusion_matrix_at_thresholds")
# @pybench.config(repeat=5)
# def bench_bootstrap_multinomial_confusion_matrix_at_thresholds():
#     BS_MULTINOMIAL.confusion_matrix_at_thresholds(
#         DF_SMALL["y_true"], DF_SMALL["y_score"]
#     )
