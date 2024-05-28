import concurrent.futures
import functools
import timeit
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl
import rapidstats
import scipy.stats
import sklearn.metrics
import tqdm

np.random.seed(208)

BASE_PATH = Path(__file__).resolve().parents[0]

N_ROWS = 1_000_000


class Bench:
    def __init__(
        self,
        df: pl.DataFrame,
        sizes: list[int] = [100, 1_000, 10_000, 100_000, 1_000_000],
        timing_runs: int = 10,
    ):
        self.benchmark_data = {"Function": [], "Size": [], "Time": []}
        self.df = df
        self.sizes = sizes
        self.timing_runs = timing_runs

    def run(self, funcs: list[Callable]):
        with tqdm.tqdm(total=len(self.sizes) * len(funcs)) as pbar:
            for n_rows in self.sizes:
                df = self.df.sample(n_rows, seed=208)

                for f in funcs:
                    func_name = (
                        f.func.__name__
                        if isinstance(f, functools.partial)
                        else f.__name__
                    )
                    time = timeit.timeit(lambda: f(df), number=self.timing_runs)

                    self.benchmark_data["Function"].append(func_name)
                    self.benchmark_data["Size"].append(n_rows)
                    self.benchmark_data["Time"].append(time)

                    pbar.update()

        return self

    def save(self, file: Path):
        pl.DataFrame(self.benchmark_data).write_parquet(file)


def fs_bootstrap_confusion_matrix(df):
    bs = rapidstats.Bootstrap()
    bs.confusion_matrix(df["y_true"], df["y_pred"])


def _cm_wrapper(df):
    return sklearn.metrics.confusion_matrix(df["y_true"], df["y_pred"]).ravel()[0]


def py_bootstrap_confusion_matrix(df):
    rapidstats.bootstrap(df, stat_func=_cm_wrapper, quiet=True)


def for_loop_bootstrap_confusion_matrix(df):
    for _ in range(1_000):
        _cm_wrapper(df.sample(fraction=1, with_replacement=True))


def main():
    y_true = np.random.choice([True, False], N_ROWS)
    y_score = np.random.rand(N_ROWS)
    y_pred = y_score > 0.5

    benchmark_df = pl.DataFrame(
        {"y_true": y_true, "y_score": y_score, "y_pred": y_pred}
    )

    Bench(benchmark_df).run(
        [
            fs_bootstrap_confusion_matrix,
            py_bootstrap_confusion_matrix,
            for_loop_bootstrap_confusion_matrix,
        ]
    ).save(BASE_PATH / "benchmark_data.parquet")


if __name__ == "__main__":
    main()
