from typing import Literal

import polars as pl
from polars.series.series import ArrayLike

from ._rustystats import _rectangular_auc, _trapezoidal_auc


def auc(
    x: ArrayLike,
    y: ArrayLike,
    method: Literal["rectangular", "trapezoidal"] = "trapezoidal",
    sorted: bool = False,
):
    df = pl.DataFrame({"x": x, "y": y}).drop_nulls().cast(pl.Float64)

    if not sorted:
        df = df.sort("x", descending=False)

    if method == "rectangular":
        return _rectangular_auc(df)
    elif method == "trapezoidal":
        return _trapezoidal_auc(df)
    else:
        raise ValueError("`method` must be one of `rectangular` or `trapezoidal`")
