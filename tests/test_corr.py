import numpy as np
import pandas as pd
import polars as pl

import rapidstats

np.random.seed(208)


N_ROWS = 100
N_COLS = 10
N_NAN = int((N_ROWS * N_COLS) / 10)

A = np.random.randn(N_ROWS, N_COLS)
A.ravel()[np.random.choice(A.size, N_NAN, replace=False)] = np.nan

DF = pl.DataFrame(A)


def remove_diagonal(array):
    return array[~np.eye(len(array), dtype=bool)].reshape(len(array), -1)


def fill_triu(array, fill_value=np.nan):
    array[np.triu_indices(array.shape[0], 1)] = fill_value

    return array


def delete_row(array, index=0):
    return np.delete(array, (index), axis=0)


def reference_correlation_matrix(df: pd.DataFrame):
    corr_mat = df.corr().to_numpy()
    corr_mat = np.tril(corr_mat)  # rapidstats only returns lower triangular part
    corr_mat = fill_triu(corr_mat)  # np.tril replace upper with 0, we need it to be NaN
    corr_mat = remove_diagonal(corr_mat)  # rapidstats doesn't compute the diagonal
    corr_mat = delete_row(corr_mat)  # delete extra row of all NaNs from the top

    return corr_mat


def test_correlation_matrix():
    ref = reference_correlation_matrix(DF.to_pandas())
    rs = rapidstats.correlation_matrix(DF).drop("").to_numpy()

    assert np.allclose(ref, rs, equal_nan=True)


def test_correlation_matrix_filter():
    df = pl.DataFrame(
        {
            "a": ["x", "y", "z"],
            "b": [1, 2, 3],
            "c": [4, 5, 6],
            "d": [7, 8, 9],
        }
    )

    rapidstats.correlation_matrix(df)

    rapidstats.correlation_matrix(df, ["a", "b"], ["c", "d"])
