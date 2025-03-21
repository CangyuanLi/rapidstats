import json
import tempfile
import zipfile
from pathlib import Path
from typing import Self, Union

import narwhals as nw
import narwhals.selectors as nws
import narwhals.typing as nwt
import polars as pl

PathLike = Union[str, Path]


def _read_list(path: PathLike) -> list:
    with open(path) as f:
        return [x.strip() for x in f.readlines()]


def _write_list(lst: list, path: PathLike):
    with open(path, "w") as f:
        for x in lst:
            f.write(f"{x}\n")


def _read_json(path: PathLike) -> dict:
    with open(path) as f:
        return json.load(f)


def _write_json(obj, path: PathLike):
    with open(path, "w") as f:
        json.dump(obj, f)


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

    def _set_range_vars(self) -> Self:
        self._range_min, self._range_max = self.feature_range
        self._range_diff = self._range_max - self._range_min

        return self

    def fit(self, X: nwt.IntoDataFrameT) -> Self:
        """_summary_

        Parameters
        ----------
        X : nwt.IntoDataFrameT
            _description_

        Attributes
        ----------
        feature_names_in : list[str]
        min_: nwt.DataFrameT
        scale_ : nwt.DataFrameT

        Returns
        -------
        self
            Fitted MinMaxScaler
        """
        X = nw.from_native(X, eager_only=True)

        self.feature_names_in_ = X.columns
        data_min = X.select(nws.all().min())
        data_max = X.select(nws.all().max())
        data_range: nwt.DataFrameT = data_max.select(
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
        return X.select(
            nw.col(c).__mul__(self.scale_[c]).__add__(self.min_[c])
            for c in self.feature_names_in_
        )

    @nw.narwhalify
    def fit_transform(self, X: nwt.IntoDataFrameT) -> nwt.IntoDataFrameT:
        return self.fit(X).transform(X)

    @nw.narwhalify
    def run(self, X: nwt.IntoFrameT) -> nwt.IntoFrameT:
        return X.select(
            nws.all()
            .__sub__(nws.all().min())
            .__truediv__(nws.all().max().__sub__(nws.all().min()))
            .__mul__(self._range_diff)
            .__add__(self._range_min)
        )

    @nw.narwhalify
    def inverse_transform(self, X: nwt.IntoFrameT) -> nwt.IntoFrameT:
        return X.select(
            nw.col(c).__sub__(self.min_[c]).__truediv__(self.scale_[c])
            for c in self.feature_names_in_
        )

    def save(self, path: PathLike) -> Self:
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
        with zipfile.ZipFile(
            path, "w"
        ) as archive, tempfile.TemporaryDirectory() as tmpdir:
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

    def load(self, path: PathLike) -> Self:
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
        with zipfile.ZipFile(
            path, "r"
        ) as archive, tempfile.TemporaryDirectory() as tmpdir:
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

    @nw.narwhalify
    def fit(self, X: nwt.IntoDataFrame) -> Self:
        self.mean_ = X.select(nws.all().mean())
        self.std_ = X.select(nws.all().std(ddof=self.ddof))
        self.feature_names_in_ = X.columns

        return self

    @nw.narwhalify
    def transform(self, X: nwt.IntoDataFrameT) -> nwt.IntoDataFrameT:
        return X.select(
            nw.col(c).__sub__(self.mean_[c]).__truediv__(self.std_[c])
            for c in self.feature_names_in_
        )

    @nw.narwhalify
    def fit_transform(self, X: nwt.IntoDataFrameT) -> nwt.IntoDataFrameT:
        return self.fit(X).transform(X)

    @nw.narwhalify
    def inverse_transform(self, X: nwt.IntoDataFrameT) -> nwt.IntoDataFrameT:
        return X.select(
            nw.col(c).__mul__(self.std_[c]).__add__(self.mean_[c])
            for c in self.feature_names_in_
        )

    @nw.narwhalify
    def run(self, X: nwt.IntoFrameT) -> nwt.IntoFrameT:
        return X.select(
            nws.all()
            .__sub__(nws.all().mean())
            .__truediv__(nws.all().std(ddof=self.ddof))
        )

    def save(self, path: PathLike) -> Self:
        with zipfile.ZipFile(
            path, "w"
        ) as archive, tempfile.TemporaryDirectory() as tmpdir:
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

    def load(self, path: PathLike) -> Self:
        with zipfile.ZipFile(
            path, "r"
        ) as archive, tempfile.TemporaryDirectory() as tmpdir:
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

    @nw.narwhalify
    def fit(self, X: nwt.IntoDataFrame) -> Self:
        self.feature_names_in_ = X.columns
        self.median_ = X.select(nws.all().median())
        self.scale_ = X.select(
            nws.all()
            .quantile(self.quantile_range[1], interpolation="linear")
            .__sub__(nws.all().quantile(self.quantile_range[0], interpolation="linear"))
        )

        return self

    @nw.narwhalify
    def transform(self, X: nwt.IntoDataFrameT) -> nwt.IntoDataFrameT:
        return X.select(
            nw.col(c).__sub__(self.median_[c]).__truediv__(self.scale_[c])
            for c in self.feature_names_in_
        )

    @nw.narwhalify
    def fit_transform(self, X: nwt.IntoDataFrameT) -> nwt.IntoDataFrameT:
        return self.fit(X).transform(X)

    @nw.narwhalify
    def inverse_transform(self, X: nwt.IntoDataFrameT) -> nwt.IntoDataFrameT:
        return X.select(
            nw.col(c).__mul__(self.scale_[c]).__add__(self.median_[c])
            for c in self.feature_names_in_
        )

    @nw.narwhalify
    def run(self, X: nwt.IntoFrameT) -> nwt.IntoFrameT:
        return X.select(
            nws.all()
            .__sub__(nws.all().median())
            .__truediv__(
                nws.all()
                .quantile(self.quantile_range[1], interpolation="linear")
                .__sub__(
                    nws.all().quantile(self.quantile_range[0], interpolation="linear")
                )
            )
        )

    def save(self, path: PathLike) -> Self:
        with zipfile.ZipFile(
            path, "w"
        ) as archive, tempfile.TemporaryDirectory() as tmpdir:
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

    def load(self, path: PathLike) -> Self:
        with zipfile.ZipFile(
            path, "r"
        ) as archive, tempfile.TemporaryDirectory() as tmpdir:
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

    def fit(self, X: nwt.IntoDataFrameT) -> Self:
        X = nw.from_native(X, eager_only=True)

        self.categories_ = {c: X[c].drop_nulls().unique() for c in X.columns}

        return self

    @nw.narwhalify
    def transform(self, X: nwt.IntoDataFrameT) -> nwt.IntoDataFrameT:
        for c, unique_vals in self.categories_.items():
            for val in unique_vals:
                X = X.with_columns(nw.col(c).__eq__(val).alias(f"{c}_{val}"))

        return X

    @nw.narwhalify
    def fit_transform(self, X: nwt.IntoDataFrameT) -> nwt.IntoDataFrameT:
        return self.fit(X).transform(X)

    def save(self, path: PathLike) -> Self:
        with zipfile.ZipFile(
            path, "w"
        ) as archive, tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            for k, v in self.categories_.items():
                v.to_frame().write_parquet(tmpdir / k)
                archive.write(tmpdir / k, k)

        return self

    def load(self, path: PathLike) -> Self:
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
        with zipfile.ZipFile(
            path, "r"
        ) as archive, tempfile.TemporaryDirectory() as tmpdir:
            archive.extractall(tmpdir)

            self.categories_ = {
                file: nw.read_parquet(f"{tmpdir}/{file}", native_namespace=pl)[file]
                for file in archive.namelist()
            }

        return self
