import narwhals as nw
import narwhals.selectors as nws
import narwhals.typing as nwt


class MinMaxScaler:
    def __init__(self, feature_range: tuple[float, float] = (0, 1)):
        self._feature_range = feature_range
        self._range_min, self._range_max = feature_range
        self._range_diff = self._range_max - self._range_min

    def fit(self, X: nwt.IntoFrameT):
        X = nw.from_native(X)

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
    def fit_transform(self, X: nwt.IntoFrameT) -> nwt.IntoFrameT:
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
