from __future__ import annotations

from typing import Literal

import polars as pl

from ._utils import (
    _PLUGIN_PATH,
    IntoExprColumn,
    NumericLiteral,
    _numeric_to_expr,
    _str_to_expr,
)


def auc(
    x: IntoExprColumn,
    y: IntoExprColumn,
    method: Literal["rectangular", "trapezoidal"] = "trapezoidal",
) -> pl.Expr:
    """Computes the area under the curve (AUC) via numerical integration.

    Parameters
    ----------
    x : pl.Expr | str
        The x-axis
    y : pl.Expr | str
        The y-axis
    method : Literal["rectangular", "trapezoidal"], optional
        If "rectangular", use rectangular integration, if "trapezoidal", use
        trapezoidal integration, by default "trapezoidal"

    Returns
    -------
    pl.Expr

    Raises
    ------
    ValueError
        If `method` is not one of `rectangular` or `trapezoidal`
    """
    if method == "trapezoidal":
        is_trapezoidal = True
    elif method == "rectangular":
        is_trapezoidal = False
    else:
        raise ValueError("`method` must be one of `rectangular` or `trapezoidal`")

    return pl.plugins.register_plugin_function(
        plugin_path=_PLUGIN_PATH,
        function_name="pl_auc",
        args=[
            _str_to_expr(x).cast(pl.Float64),
            _str_to_expr(y).cast(pl.Float64),
            pl.lit(is_trapezoidal),
        ],
        returns_scalar=True,
    )


def is_close(
    x: IntoExprColumn | NumericLiteral,
    y: IntoExprColumn | NumericLiteral,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    null_equal: bool = False,
) -> pl.Expr:
    """Compares the relative equality of the inputs.

    Parameters
    ----------
    x : pl.Expr | str | float
    y : pl.Expr | str | float
    rtol : float, optional
        Relative tolerance, by default 1e-05
    atol : float, optional
        Absolute tolerance, by default 1e-08
    null_equal : bool, optional
        If True, considers nulls to be equal, by default False

    Returns
    -------
    pl.Expr
    """
    x = _numeric_to_expr(x)
    y = _numeric_to_expr(y)

    res = x.sub(y).abs().le(pl.lit(atol).add(rtol).mul(y.abs()))

    if null_equal:
        res = res.or_(x.is_null().and_(y.is_null()))

    return res
