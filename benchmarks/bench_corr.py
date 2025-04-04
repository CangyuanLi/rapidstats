import numpy as np
import pandas as pd
import polars as pl
import pybench

import rapidstats

SEED = 208

np.random.seed(SEED)


N_ROWS = 50_000
N_COLS = 150
N_NAN = int((N_ROWS * N_COLS) / 10)

A = np.random.randn(N_ROWS, N_COLS)
A.ravel()[np.random.choice(A.size, N_NAN, replace=False)] = np.nan

DF = pl.DataFrame(A)

ROW_SIZES = [100, 1_000, 10_000, 50_000]
COL_SIZES = [10, 50, 100, 150]


def polars_setup(n_rows, n_cols):
    cols = np.random.choice(DF.columns, n_cols, replace=False).tolist()
    df = DF.select(cols).sample(n_rows, seed=SEED)

    return {"df": df}


def pandas_setup(n_rows, n_cols):
    return {"df": polars_setup(n_rows, n_cols)["df"].to_pandas()}


@pybench.metadata(group="correlation_matrix")
@pybench.config(repeat=5)
@pybench.parametrize({"n_rows": ROW_SIZES, "n_cols": COL_SIZES}, setup=pandas_setup)
def bench_pandas_correlation_matrix(df: pd.DataFrame):
    df.corr()


@pybench.metadata(group="correlation_matrix")
@pybench.config(repeat=5)
@pybench.parametrize({"n_rows": ROW_SIZES, "n_cols": COL_SIZES}, setup=polars_setup)
def bench_correlation_matrix(df: pl.DataFrame):
    rapidstats.correlation_matrix(df)
