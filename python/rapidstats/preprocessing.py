import narwhals as nw
import narwhals.selectors as nws
import narwhals.typing as nwt


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
        self._feature_range = feature_range
        self._range_min, self._range_max = feature_range
        self._range_diff = self._range_max - self._range_min

    def fit(self, X: nwt.IntoDataFrameT):
        """_summary_

        Parameters
        ----------
        X : nwt.IntoDataFrameT
            _description_

        Attributes
        ----------
        data_min_ : nwt.DataFrameT
        data_max_ : nwt.DataFrameT
        data_range_ : nwt.DataFrameT
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
        self.data_min_ = X.select(nws.all().min())
        self.data_max_ = X.select(nws.all().max())
        self.data_range_: nwt.DataFrameT = self.data_max_.select(
            nw.col(c).__sub__(self.data_min_[c]) for c in self.feature_names_in_
        )

        self.scale_ = self.data_range_.with_columns(
            nw.lit(self._range_diff).__truediv__(nw.col(c)).alias(c)
            for c in self.feature_names_in_
        )

        self.min_ = self.data_min_.select(
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


class OneHotEncoder:
    """One-hot encodes data.

    Added in version 0.1.0
    ----------------------
    """

    def __init__(self):
        pass

    def fit(self, X: nwt.IntoDataFrameT):
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
