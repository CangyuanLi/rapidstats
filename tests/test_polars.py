import numpy as np
import polars as pl
import pytest

import rapidstats

np.random.seed(208)

N_ROWS = 1_000
X = np.random.rand(N_ROWS)
Y = np.random.rand(N_ROWS)


def sort_arrays(*arrays):
    index = np.argsort(arrays[0])

    return [x[index] for x in arrays]


@pytest.mark.parametrize("x,y", [(X, Y)])
def test_auc(x, y):
    def reference_auc(x, y):
        x, y = sort_arrays(x, y)
        return np.trapz(y, x)

    def rs_auc(x, y, method="trapezoidal"):
        df = pl.DataFrame({"x": x, "y": y})

        return df.select(rapidstats.polars.auc("x", "y", method=method)).item()

    ref_auc = reference_auc(x, y)

    assert pytest.approx(ref_auc) == rs_auc(x, y)

    # Test group-by

    df = pl.concat(
        pl.LazyFrame({"x": x, "y": y}).with_columns(pl.lit(i).alias("idx"))
        for i in range(5)
    )

    res = (
        df.group_by("idx")
        .agg(rapidstats.polars.auc("x", "y", method="trapezoidal").alias("auc"))
        .collect()["auc"]
        .to_list()
    )

    assert pytest.approx([ref_auc for _ in range(5)]) == res

    # Test that rectangular works
    # TODO: Find library that implements rectangular AUC to test against
    rs_auc(x, y, method="rectangular")


def test_is_close():
    df = pl.DataFrame(
        {
            "a": [float("nan"), None, 0.00001, 0.01, 0.01],
            "b": [float("nan"), None, 0.000010000000000067, 0.02, 0.01],
            "correct": [True, None, True, False, True],
            "correct_null_equal": [True, True, True, False, True],
        }
    ).with_columns(
        rapidstats.polars.is_close("a", "b", null_equal=False).alias("is_close"),
        rapidstats.polars.is_close("a", "b", null_equal=True).alias(
            "is_close_null_equal"
        ),
    )

    assert df["correct"].eq(df["is_close"]).sum() == 4
    assert df["correct_null_equal"].eq(df["is_close_null_equal"]).sum() == 5
