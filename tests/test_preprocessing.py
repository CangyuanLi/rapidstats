import tempfile

import numpy as np
import polars as pl
import polars.selectors as cs
import polars.testing
import sklearn.preprocessing

import rapidstats as rs

SEED = 208

np.random.seed(SEED)

n_rows = 1_000
cols = ["A", "B"]
DATA = (
    pl.DataFrame({c: np.random.rand(n_rows) for c in cols})
    .with_columns(
        pl.int_range(n_rows)
        .sample(pl.len(), with_replacement=True, seed=SEED + i)
        .truediv(n_rows)
        .alias(f"prob_{c}")
        for i, c in enumerate(cols)
    )
    .with_columns(
        pl.when(pl.col(f"prob_{c}").gt(0.9)).then(None).otherwise(pl.col(c)).alias(c)
        for c in cols
    )
    .drop(cs.starts_with("prob_"))
)


def test_min_max_scaler():
    ref = sklearn.preprocessing.MinMaxScaler().fit_transform(DATA)

    np.testing.assert_allclose(
        ref,
        rs.preprocessing.MinMaxScaler().fit_transform(DATA).to_numpy(),
    )

    np.testing.assert_allclose(
        ref,
        rs.preprocessing.MinMaxScaler().run(DATA).to_numpy(),
    )

    # test inverse transform
    scaler = rs.preprocessing.MinMaxScaler()

    polars.testing.assert_frame_equal(
        DATA, scaler.inverse_transform(scaler.fit_transform(DATA))
    )


def test_min_max_scaler_save():
    with tempfile.TemporaryFile() as f:
        scaler = rs.preprocessing.MinMaxScaler(feature_range=(1, 2))
        scaler.fit(DATA)
        scaler.save(f)

        scaler_loaded = rs.preprocessing.MinMaxScaler().load(f)

        polars.testing.assert_frame_equal(
            scaler.min_.to_polars(), scaler_loaded.min_.to_polars()
        )
        polars.testing.assert_frame_equal(
            scaler.scale_.to_polars(), scaler_loaded.scale_.to_polars()
        )
        assert scaler.feature_names_in_ == scaler_loaded.feature_names_in_
        assert scaler.feature_range == scaler_loaded.feature_range

        polars.testing.assert_frame_equal(
            scaler.transform(DATA), scaler_loaded.transform(DATA)
        )

        polars.testing.assert_frame_equal(scaler.run(DATA), scaler_loaded.run(DATA))


def test_standard_scaler():
    # If using polars backend, as here, NaN is not ignored during mean. Added a note to
    # the docstring to warn users of this. Also, use a default ddof of 1, but use 0 here
    # to match sklearn's behavior.
    data = DATA.fill_nan(None)

    reference = sklearn.preprocessing.StandardScaler().fit_transform(data)

    np.testing.assert_allclose(
        reference,
        rs.preprocessing.StandardScaler(ddof=0).fit_transform(data).to_numpy(),
    )

    np.testing.assert_allclose(
        reference, rs.preprocessing.StandardScaler(ddof=0).run(data).to_numpy()
    )

    # test inverse transform
    scaler = rs.preprocessing.StandardScaler()
    polars.testing.assert_frame_equal(
        data,
        scaler.inverse_transform(scaler.fit_transform(data)),
    )

    # test save and load
    with tempfile.TemporaryFile() as f:
        scaler = rs.preprocessing.StandardScaler(ddof=10)
        scaler.fit(data)
        scaler.save(f)

        scaler_loaded = rs.preprocessing.StandardScaler().load(f)

        polars.testing.assert_frame_equal(
            scaler.mean_.to_polars(), scaler_loaded.mean_.to_polars()
        )
        polars.testing.assert_frame_equal(
            scaler.std_.to_polars(), scaler_loaded.std_.to_polars()
        )
        assert scaler.feature_names_in_ == scaler_loaded.feature_names_in_
        assert scaler.ddof == scaler_loaded.ddof

        polars.testing.assert_frame_equal(
            scaler.transform(data), scaler_loaded.transform(data)
        )

        polars.testing.assert_frame_equal(scaler.run(data), scaler_loaded.run(data))


def test_robust_scaler():
    data = DATA.fill_nan(None)

    reference = sklearn.preprocessing.RobustScaler().fit_transform(data)

    np.testing.assert_allclose(
        reference,
        rs.preprocessing.RobustScaler().fit_transform(data).to_numpy(),
    )

    np.testing.assert_allclose(
        reference, rs.preprocessing.RobustScaler().run(data).to_numpy()
    )

    # test inverse transform
    scaler = rs.preprocessing.RobustScaler()
    polars.testing.assert_frame_equal(
        data,
        scaler.inverse_transform(scaler.fit_transform(data)),
    )

    # test save and load
    with tempfile.TemporaryFile() as f:
        scaler = rs.preprocessing.RobustScaler(quantile_range=(0.1, 0.9))
        scaler.fit(data)
        scaler.save(f)

        scaler_loaded = rs.preprocessing.RobustScaler().load(f)

        polars.testing.assert_frame_equal(
            scaler.median_.to_polars(), scaler_loaded.median_.to_polars()
        )
        polars.testing.assert_frame_equal(
            scaler.scale_.to_polars(), scaler_loaded.scale_.to_polars()
        )
        assert scaler.feature_names_in_ == scaler_loaded.feature_names_in_
        assert scaler.quantile_range == scaler_loaded.quantile_range

        polars.testing.assert_frame_equal(
            scaler.transform(data), scaler_loaded.transform(data)
        )

        polars.testing.assert_frame_equal(scaler.run(data), scaler_loaded.run(data))


def test_one_hot_encoder():
    df1 = pl.DataFrame({"x": ["a", None, "b"]})
    df2 = pl.DataFrame({"x": ["a", None, "a"]})

    encoder = rs.preprocessing.OneHotEncoder().fit(df1)

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

    # test save and load
    with tempfile.TemporaryFile() as f:
        encoder = rs.preprocessing.OneHotEncoder().fit(df1)
        encoder.save(f)

        encoder_loaded = rs.preprocessing.OneHotEncoder().load(f)

        for k, v in encoder.categories_.items():
            polars.testing.assert_series_equal(
                v.to_polars(), encoder_loaded.categories_[k].to_polars()
            )
