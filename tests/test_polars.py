import numpy as np
import polars as pl
import polars.testing as plt
import pytest

import rapidstats as rs

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

        return df.select(rs.polars.auc("x", "y", method=method)).item()

    ref_auc = reference_auc(x, y)

    assert pytest.approx(ref_auc) == rs_auc(x, y)

    # Test group-by

    df = pl.concat(
        pl.LazyFrame({"x": x, "y": y}).with_columns(pl.lit(i).alias("idx"))
        for i in range(5)
    )

    res = (
        df.group_by("idx")
        .agg(rs.polars.auc("x", "y", method="trapezoidal").alias("auc"))
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
        rs.polars.is_close("a", "b", null_equal=False).alias("is_close"),
        rs.polars.is_close("a", "b", null_equal=True).alias("is_close_null_equal"),
    )

    assert df["correct"].eq(df["is_close"]).sum() == 4
    assert df["correct_null_equal"].eq(df["is_close_null_equal"]).sum() == 5


def test_format():
    x = 10_000_000
    df = pl.DataFrame({"a": [100.05989, -0.99999], "b": [10000, 25]}).with_columns(
        rs.polars.format(
            "{:.5f} {{another}} {:+_.3%} some space {}",
            pl.col("a"),
            pl.col("b"),
            x,
        ).alias("formatted"),
        pl.struct(a="a", b="b", x=pl.lit(x))
        .map_elements(
            lambda dct: f"{dct['a']:.5f} {{another}} {dct['b']:+_.3%} some space {dct['x']}",
            return_dtype=pl.String,
        )
        .alias("formatted_correct"),
    )

    plt.assert_series_equal(df["formatted_correct"], df["formatted"], check_names=False)
