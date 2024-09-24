import itertools

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


def test_correlation_matrix_combinations():
    # Test if user passes in combinations directly

    # Reference
    combinations = list(itertools.combinations(DF.columns, r=2))[:5]
    ref_corr_mats = []
    for a, b in combinations:
        x = (
            pd.DataFrame({a: DF[a], b: DF[b]})
            .corr()
            .reset_index()
            .melt("index")
            .rename(columns={"index": "col1", "variable": "col2", "value": "ref"})
        )
        x = x.loc[(x["col1"] == b) & (x["col2"] == a)]

        ref_corr_mats.append(x)

    ref_corr_mat = pd.concat(ref_corr_mats, axis=0)

    corr_mat = (
        rapidstats.correlation_matrix(DF, combinations)
        .unpivot(index="")
        .rename({"": "col1", "variable": "col2"})
        .filter(pl.col("col1") != pl.col("col2"))
    )

    assert ref_corr_mat.shape == corr_mat.shape

    res = corr_mat.join(
        pl.from_pandas(ref_corr_mat), on=["col1", "col2"], how="inner", validate="1:1"
    )

    assert res.height == corr_mat.height

    assert np.allclose(res["ref"], res["value"])


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
