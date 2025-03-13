from typing import Literal

import polars as pl
from polars.series.series import ArrayLike

from ._rustystats import _rectangular_auc, _trapezoidal_auc


def auc(
    x: ArrayLike,
    y: ArrayLike,
    method: Literal["rectangular", "trapezoidal"] = "trapezoidal",
    sorted: bool = False,
) -> float:
    """Computes the Area Under the Curve (AUC) via numerical integration.

    Parameters
    ----------
    x : ArrayLike
        x points
    y : ArrayLike
        y points
    method : Literal["rectangular", "trapezoidal"], optional
        Integration method, by default "trapezoidal"
    sorted : bool, optional
        If True, assumes arrays are already sorted by `x`, by default False

    Returns
    -------
    float
        The AUC

    Raises
    ------
    ValueError
        If `method` is not one of `rectangular` or `trapezoidal`

    Added in version 0.1.0
    ----------------------
    """
    df = pl.DataFrame({"x": x, "y": y}).drop_nulls().cast(pl.Float64)

    if not sorted:
        df = df.sort("x", descending=False)

    if method == "rectangular":
        return _rectangular_auc(df)
    elif method == "trapezoidal":
        return _trapezoidal_auc(df)
    else:
        raise ValueError("`method` must be one of `rectangular` or `trapezoidal`")
