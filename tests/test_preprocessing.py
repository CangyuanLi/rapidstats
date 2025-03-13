import numpy as np
import polars as pl
import polars.selectors as cs
import polars.testing
import sklearn.preprocessing

import rapidstats

SEED = 208

np.random.seed(SEED)


def test_min_max_scaler():
    n_rows = 1_000
    cols = ["A", "B"]
    data = (
        pl.DataFrame({c: np.random.rand(n_rows) for c in cols})
        .with_columns(
            pl.int_range(n_rows)
            .sample(pl.len(), with_replacement=True, seed=SEED + i)
            .truediv(n_rows)
            .alias(f"prob_{c}")
            for i, c in enumerate(cols)
        )
        .with_columns(
            pl.when(pl.col(f"prob_{c}").gt(0.9))
            .then(None)
            .when(pl.col(f"prob_{c}").lt(0.1))
            .then(float("nan"))
            .otherwise(pl.col(c))
            .alias(c)
            for c in cols
        )
        .drop(cs.starts_with("prob_"))
    )

    np.testing.assert_allclose(
        sklearn.preprocessing.MinMaxScaler().fit_transform(data),
        rapidstats.preprocessing.MinMaxScaler().fit_transform(data).to_numpy(),
    )

    # test inverse transform
    scaler = rapidstats.preprocessing.MinMaxScaler()

    polars.testing.assert_frame_equal(
        data, scaler.inverse_transform(scaler.fit_transform(data))
    )


def test_one_hot_encoder():
    df1 = pl.DataFrame({"x": ["a", None, "b"]})
    df2 = pl.DataFrame({"x": ["a", None, "a"]})

    encoder = rapidstats.preprocessing.OneHotEncoder().fit(df1)

    polars.testing.assert_frame_equal(
        df1.with_columns(
            pl.Series("x_a", [True, None, False]),
            pl.Series("x_b", [False, None, True]),
        ),
        df1.pipe(encoder.transform).select("x", "x_a", "x_b"),
    )

    polars.testing.assert_frame_equal(
        df2.with_columns(
            pl.Series("x_a", [True, None, True]), pl.Series("x_b", [False, None, False])
        ),
        df2.pipe(encoder.transform).select("x", "x_a", "x_b"),
    )
