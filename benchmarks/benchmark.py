import concurrent.futures
import functools
import time
import timeit
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl
import scipy.stats
import sklearn.metrics
import tqdm

import rapidstats

np.random.seed(208)


BASE_PATH = Path(__file__).resolve().parents[0]
# BOOTSTRAP = rapidstats.Bootstrap(iterations=500)
N_ROWS = 1_000_000


# class Bench:
#     def __init__(
#         self,
#         df: pl.DataFrame,
#         sizes: list[int] = [100, 1_000, 10_000, 100_000, 500_000, 600_000, 700_000],
#         timing_runs: int = 10,
#     ):
#         self.benchmark_data = {"Function": [], "Size": [], "Time": []}
#         self.df = df
#         self.sizes = sizes
#         self.timing_runs = timing_runs

#     def run(self, funcs: list[Callable]):
#         with tqdm.tqdm(total=len(self.sizes) * len(funcs)) as pbar:
#             for n_rows in self.sizes:
#                 df = self.df.sample(n_rows, seed=208)

#                 for f in funcs:
#                     func_name = (
#                         f.func.__name__
#                         if isinstance(f, functools.partial)
#                         else f.__name__
#                     )
#                     time = timeit.timeit(lambda: f(df), number=self.timing_runs)

#                     self.benchmark_data["Function"].append(func_name)
#                     self.benchmark_data["Size"].append(n_rows)
#                     self.benchmark_data["Time"].append(time)

#                     pbar.update()

#         return self

#     def save(self, file: Path):
#         pl.DataFrame(self.benchmark_data).write_parquet(file)


# def fs_bootstrap_confusion_matrix(df):
#     bs = rapidstats.Bootstrap()
#     bs.confusion_matrix(df["y_true"], df["y_pred"])


# def _cm_wrapper(df):
#     return sklearn.metrics.confusion_matrix(df["y_true"], df["y_pred"]).ravel()[0]


# def py_bootstrap_confusion_matrix(df):
#     rapidstats.bootstrap(df, stat_func=_cm_wrapper, quiet=True)


# def for_loop_bootstrap_confusion_matrix(df):
#     for _ in range(1_000):
#         _cm_wrapper(df.sample(fraction=1, with_replacement=True))


# def roc_auc(df):
#     BOOTSTRAP.roc_auc(y_true=df["y_true"], y_score=df["y_score"])


# def roc_auc2(df):
#     BOOTSTRAP.roc_auc2(y_true=df["y_true"], y_score=df["y_score"])


def main():
    n_rows = 1_100_000
    y_true = np.random.choice([True, False], n_rows)
    y_score = np.random.rand(n_rows)

    benchmark_df = pl.DataFrame({"y_true": y_true, "y_score": y_score})

    data = {
        "function": [],
        "data_size": [],
        "bootstrap_iterations": [],
        "time": [],
        "n_jobs": [],
        "chunksize": [],
    }
    for data_size in tqdm.tqdm(
        [
            # 100,
            # 1_000,
            # 10_000,
            # 100_000,
            500_000,
            # 700_000,
            # 1_000_000,
            # 1_100_000,
        ]
    ):
        df = benchmark_df.sample(n=data_size, seed=208)
        yt = df["y_true"]
        ys = df["y_score"]
        for bootstrap_iterations in [600]:
            for n_jobs in [None]:
                if n_jobs == 1:
                    chunksizes = [None]
                else:
                    chunksizes = [None, 2, 100]
                for chunksize in chunksizes:
                    bs = rapidstats.Bootstrap(
                        iterations=bootstrap_iterations,
                        seed=208,
                        n_jobs=n_jobs,
                        chunksize=chunksize,
                    )
                    # I know I should use timeit, but the latter functions are long-running
                    # enough that I feel I can get away with perf counter. Also, I don't want
                    # my laptop to explode.
                    for func in [bs.roc_auc, bs.roc_auc2]:
                        data["function"].append(func.__name__)
                        data["bootstrap_iterations"].append(bootstrap_iterations)
                        data["data_size"].append(data_size)
                        data["n_jobs"].append(n_jobs)
                        data["chunksize"].append(chunksize)

                        start = time.perf_counter()
                        func(yt, ys)
                        end = time.perf_counter()

                        data["time"].append(end - start)

    pl.DataFrame(data).write_parquet(BASE_PATH / "bootstrap_investigation3.parquet")

    # Bench(benchmark_df).run([roc_auc, roc_auc2]).save(
    #     BASE_PATH / "benchmark_data.parquet"
    # )


if __name__ == "__main__":
    main()
