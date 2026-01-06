from __future__ import annotations

import json
import tempfile
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Optional, Union

import narwhals.stable.v1 as nw
import narwhals.typing as nwt
import polars as pl

PathLike = Union[str, Path]


def _read_json(path: PathLike) -> dict:
    with open(path) as f:
        return json.load(f)


def _write_json(obj, path: PathLike):
    with open(path, "w") as f:
        json.dump(obj, f)


def _resolve_columns(
    X: nwt.DataFrame, columns: Optional[str | Iterable[str]]
) -> Iterable[str]:
    if columns is None:
        return X.columns
    elif isinstance(columns, str):
        return [columns]
    else:
        return columns


class MinMaxScaler:
    """Scale data using min-max scaling.

    Parameters
    ----------
    feature_range : tuple[float, float], optional
        The range to scale the data to, by default (0, 1)

    Added in version 0.1.0
    ----------------------
    """

    def __init__(self, feature_range: tuple[float, float] = (0, 1)):
        self.feature_range = feature_range
        self._set_range_vars()

    def _set_range_vars(self):
        self._range_min, self._range_max = self.feature_range
        self._range_diff = self._range_max - self._range_min

        return self

    def fit(self, X: nwt.IntoDataFrameT, columns: Optional[str | Iterable[str]] = None):
        """Computes the min(s) and max(es) to be used for scaling.

        Parameters
        ----------
        X : nwt.IntoDataFrameT
        columns : Optional[str | Iterable[str]], optional
            The columns to apply scaling to. If None, all columns are scaled, by default
            None

        Attributes
        ----------
        feature_names_in : list[str]
        min_: nwt.DataFrameT
        scale_ : nwt.DataFrameT

        Returns
        -------
        Self
        """
        X = nw.from_native(X, eager_only=True)

        self.feature_names_in_ = _resolve_columns(X, columns)
        data_min = X.select(nw.col(self.feature_names_in_).min())
        data_max = X.select(nw.col(self.feature_names_in_).max())
        data_range: nwt.DataFrame = data_max.select(
            nw.col(c).__sub__(data_min[c]) for c in self.feature_names_in_
        )

        self.scale_ = data_range.with_columns(
            nw.lit(self._range_diff).__truediv__(nw.col(c)).alias(c)
            for c in self.feature_names_in_
        )

        self.min_ = data_min.select(
            nw.lit(self._range_min).__sub__(nw.col(c).__mul__(self.scale_[c])).alias(c)
            for c in self.feature_names_in_
        )

        return self

    @nw.narwhalify
    def transform(self, X: nwt.IntoFrameT) -> nwt.IntoFrameT:
        return X.with_columns(
            nw.col(c).__mul__(self.scale_[c]).__add__(self.min_[c])
            for c in self.feature_names_in_
        )

    @nw.narwhalify
    def fit_transform(self, X: nwt.IntoDataFrameT) -> nwt.IntoDataFrameT:
        return self.fit(X).transform(X)

    @nw.narwhalify
    def inverse_transform(self, X: nwt.IntoFrameT) -> nwt.IntoFrameT:
        return X.with_columns(
            nw.col(c).__sub__(self.min_[c]).__truediv__(self.scale_[c])
            for c in self.feature_names_in_
        )

    def _run_one(self, c: str) -> nw.Expr:
        expr = nw.col(c)
        min_ = expr.min()

        return (
            expr.__sub__(min_)
            .__truediv__(expr.max().__sub__(min_))
            .__mul__(self._range_diff)
            .__add__(self._range_min)
        )

    @nw.narwhalify
    def run(
        self, X: nwt.IntoFrameT, columns: Optional[str | Iterable[str]] = None
    ) -> nwt.IntoFrameT:
        return X.with_columns(self._run_one(c) for c in _resolve_columns(X, columns))

    def save(self, path: PathLike):
        """_summary_

        Parameters
        ----------
        path : PathLike
            _description_

        Returns
        -------
        _type_
            _description_

        Added in version 0.2.0
        ----------------------
        """
        with (
            zipfile.ZipFile(path, "w") as archive,
            tempfile.TemporaryDirectory() as tmpdir,
        ):
            tmpdir = Path(tmpdir)

            self.min_.write_parquet(tmpdir / "min_.parquet")
            self.scale_.write_parquet(tmpdir / "scale_.parquet")
            _write_json(
                {
                    "feature_names_in_": self.feature_names_in_,
                    "feature_range": self.feature_range,
                },
                tmpdir / "instance_vars.json",
            )

            archive.write(tmpdir / "min_.parquet", "min_.parquet")
            archive.write(tmpdir / "scale_.parquet", "scale_.parquet")
            archive.write(tmpdir / "instance_vars.json", "instance_vars.json")

        return self

    def load(self, path: PathLike):
        """_summary_

        Parameters
        ----------
        path : PathLike
            _description_

        Returns
        -------
        _type_
            _description_

        Added in version 0.2.0
        ----------------------
        """
        with (
            zipfile.ZipFile(path, "r") as archive,
            tempfile.TemporaryDirectory() as tmpdir,
        ):
            archive.extractall(tmpdir)

            self.min_ = nw.read_parquet(f"{tmpdir}/min_.parquet", native_namespace=pl)
            self.scale_ = nw.read_parquet(
                f"{tmpdir}/scale_.parquet", native_namespace=pl
            )
            instance_vars = _read_json(f"{tmpdir}/instance_vars.json")
            self.feature_names_in_ = instance_vars["feature_names_in_"]
            self.feature_range = tuple(instance_vars["feature_range"])

            self._set_range_vars()

        return self


class StandardScaler:
    """_summary_

    !!! Null Handling
        `rapidstats` uses [narwhals](https://narwhals-dev.github.io/narwhals/) to ingest
        supported DataFrames. However, null-handling can differ across backends. For
        example, if using a Polars backend, NaNs are valid numbers, not missing.
        Therefore, the mean / standard deviation of a column with NaNs will be NaN.
        Ensure your input is sanitized according to your specific backend before using
        `StandardScaler`.
    """

    def __init__(self, ddof: int = 1):
        self.ddof = ddof

    def fit(self, X: nwt.IntoDataFrame, columns: Optional[str | Iterable[str]] = None):
        X = nw.from_native(X, eager_only=True)
        self.feature_names_in_ = _resolve_columns(X, columns)
        selector = nw.col(self.feature_names_in_)

        self.mean_ = X.select(selector.mean())
        self.std_ = X.select(selector.std(ddof=self.ddof))

        return self

    @nw.narwhalify
    def transform(self, X: nwt.IntoFrameT) -> nwt.IntoFrameT:
        return X.with_columns(
            nw.col(c).__sub__(self.mean_[c]).__truediv__(self.std_[c])
            for c in self.feature_names_in_
        )

    @nw.narwhalify
    def fit_transform(
        self, X: nwt.IntoDataFrameT, columns: Optional[str | Iterable[str]] = None
    ) -> nwt.IntoDataFrameT:
        return self.fit(X, columns=columns).transform(X)

    @nw.narwhalify
    def inverse_transform(self, X: nwt.IntoFrameT) -> nwt.IntoFrameT:
        return X.with_columns(
            nw.col(c).__mul__(self.std_[c]).__add__(self.mean_[c])
            for c in self.feature_names_in_
        )

    def _run_one(self, c: str) -> nw.Expr:
        expr = nw.col(c)

        return expr.__sub__(expr.mean()).__truediv__(expr.std(ddof=self.ddof))

    @nw.narwhalify
    def run(
        self, X: nwt.IntoFrameT, columns: Optional[str | Iterable[str]] = None
    ) -> nwt.IntoFrameT:
        return X.with_columns(self._run_one(c) for c in _resolve_columns(X, columns))

    def save(self, path: PathLike):
        with (
            zipfile.ZipFile(path, "w") as archive,
            tempfile.TemporaryDirectory() as tmpdir,
        ):
            tmpdir = Path(tmpdir)

            self.mean_.write_parquet(tmpdir / "mean_.parquet")
            self.std_.write_parquet(tmpdir / "std_.parquet")
            _write_json(
                {
                    "feature_names_in_": self.feature_names_in_,
                    "ddof": self.ddof,
                },
                tmpdir / "instance_vars.json",
            )

            archive.write(tmpdir / "mean_.parquet", "mean_.parquet")
            archive.write(tmpdir / "std_.parquet", "std_.parquet")
            archive.write(tmpdir / "instance_vars.json", "instance_vars.json")

        return self

    def load(self, path: PathLike):
        with (
            zipfile.ZipFile(path, "r") as archive,
            tempfile.TemporaryDirectory() as tmpdir,
        ):
            archive.extractall(tmpdir)

            self.mean_ = nw.read_parquet(f"{tmpdir}/mean_.parquet", native_namespace=pl)
            self.std_ = nw.read_parquet(f"{tmpdir}/std_.parquet", native_namespace=pl)
            instance_vars = _read_json(f"{tmpdir}/instance_vars.json")
            self.feature_names_in_ = instance_vars["feature_names_in_"]
            self.ddof = instance_vars["ddof"]

        return self


class RobustScaler:
    def __init__(self, quantile_range: tuple[float, float] = (0.25, 0.75)):
        self.quantile_range = quantile_range

    def fit(self, X: nwt.IntoDataFrame, columns: Optional[Iterable[str]] = None):
        X = nw.from_native(X, eager_only=True)

        self.feature_names_in_ = _resolve_columns(X, columns)
        selector = nw.col(self.feature_names_in_)

        self.median_ = X.select(selector.median())
        self.scale_ = X.select(
            nw.col(c)
            .quantile(self.quantile_range[1], interpolation="linear")
            .__sub__(nw.col(c).quantile(self.quantile_range[0], interpolation="linear"))
            for c in self.feature_names_in_
        )

        return self

    @nw.narwhalify
    def transform(self, X: nwt.IntoFrameT) -> nwt.IntoFrameT:
        return X.with_columns(
            nw.col(c).__sub__(self.median_[c]).__truediv__(self.scale_[c])
            for c in self.feature_names_in_
        )

    @nw.narwhalify
    def fit_transform(
        self, X: nwt.IntoFrameT, columns: Optional[str | Iterable[str]] = None
    ) -> nwt.IntoFrameT:
        return self.fit(X, columns=columns).transform(X)

    @nw.narwhalify
    def inverse_transform(self, X: nwt.IntoFrameT) -> nwt.IntoFrameT:
        return X.with_columns(
            nw.col(c).__mul__(self.scale_[c]).__add__(self.median_[c])
            for c in self.feature_names_in_
        )

    def _run_one(self, c: str) -> nw.Expr:
        expr = nw.col(c)

        return expr.__sub__(expr.median()).__truediv__(
            expr.quantile(self.quantile_range[1], interpolation="linear").__sub__(
                expr.quantile(self.quantile_range[0], interpolation="linear")
            )
        )

    @nw.narwhalify
    def run(
        self, X: nwt.IntoFrameT, columns: Optional[str | Iterable[str]] = None
    ) -> nwt.IntoFrameT:
        return X.with_columns(self._run_one(c) for c in _resolve_columns(X, columns))

    def save(self, path: PathLike):
        with (
            zipfile.ZipFile(path, "w") as archive,
            tempfile.TemporaryDirectory() as tmpdir,
        ):
            tmpdir = Path(tmpdir)

            self.median_.write_parquet(tmpdir / "median_.parquet")
            self.scale_.write_parquet(tmpdir / "scale_.parquet")
            _write_json(
                {
                    "feature_names_in_": self.feature_names_in_,
                    "quantile_range": self.quantile_range,
                },
                tmpdir / "instance_vars.json",
            )

            archive.write(tmpdir / "median_.parquet", "median_.parquet")
            archive.write(tmpdir / "scale_.parquet", "scale_.parquet")
            archive.write(tmpdir / "instance_vars.json", "instance_vars.json")

        return self

    def load(self, path: PathLike):
        with (
            zipfile.ZipFile(path, "r") as archive,
            tempfile.TemporaryDirectory() as tmpdir,
        ):
            archive.extractall(tmpdir)

            self.median_ = nw.read_parquet(
                f"{tmpdir}/median_.parquet", native_namespace=pl
            )
            self.scale_ = nw.read_parquet(
                f"{tmpdir}/scale_.parquet", native_namespace=pl
            )
            instance_vars = _read_json(f"{tmpdir}/instance_vars.json")
            self.feature_names_in_ = instance_vars["feature_names_in_"]
            self.quantile_range = tuple(instance_vars["quantile_range"])

        return self


class OneHotEncoder:
    """One-hot encodes data.

    Added in version 0.1.0
    ----------------------
    """

    def __init__(self):
        pass

    def fit(self, X: nwt.IntoDataFrameT, columns: Optional[str | Iterable[str]] = None):
        X = nw.from_native(X, eager_only=True)

        self.categories_ = {
            c: X[c].drop_nulls().unique() for c in _resolve_columns(X, columns)
        }

        return self

    @nw.narwhalify
    def transform(self, X: nwt.IntoFrameT) -> nwt.IntoFrameT:
        for c, unique_vals in self.categories_.items():
            for val in unique_vals:
                X = X.with_columns(nw.col(c).__eq__(val).alias(f"{c}_{val}"))

        return X

    @nw.narwhalify
    def fit_transform(
        self, X: nwt.IntoDataFrameT, columns: Optional[str | Iterable[str]] = None
    ) -> nwt.IntoDataFrameT:
        return self.fit(X, columns=columns).transform(X)

    def save(self, path: PathLike):
        with (
            zipfile.ZipFile(path, "w") as archive,
            tempfile.TemporaryDirectory() as tmpdir,
        ):
            tmpdir = Path(tmpdir)

            for k, v in self.categories_.items():
                v.to_frame().write_parquet(tmpdir / k)
                archive.write(tmpdir / k, k)

        return self

    def load(self, path: PathLike):
        """_summary_

        Parameters
        ----------
        path : PathLike
            _description_

        Returns
        -------
        Self
            _description_

        Added in version 0.2.0
        ----------------------
        """
        with (
            zipfile.ZipFile(path, "r") as archive,
            tempfile.TemporaryDirectory() as tmpdir,
        ):
            archive.extractall(tmpdir)

            self.categories_ = {
                file: nw.read_parquet(f"{tmpdir}/{file}", native_namespace=pl)[file]
                for file in archive.namelist()
            }

        return self
