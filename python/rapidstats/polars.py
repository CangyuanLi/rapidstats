from pathlib import Path
from typing import Literal, Union

import polars as pl

PolarsFrame = Union[pl.DataFrame, pl.LazyFrame]
IntoExpr = Union[str, pl.Expr]

_PLUGIN_PATH = Path(__file__).resolve().parent


def _into_expr(expr: IntoExpr) -> pl.Expr:
    if isinstance(expr, str):
        return pl.col(expr)
    elif isinstance(expr, pl.Expr):
        return expr
    else:
        raise ValueError("Must be of type `str` or `pl.Expr`")


def auc(
    x: IntoExpr,
    y: IntoExpr,
    method: Literal["rectangular", "trapezoidal"] = "trapezoidal",
) -> pl.Expr:
    """Computes the area under the curve (AUC) via numerical integration.

    Parameters
    ----------
    x : IntoExpr
        The x-axis
    y : IntoExpr
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
            _into_expr(x).cast(pl.Float64),
            _into_expr(y).cast(pl.Float64),
            pl.lit(is_trapezoidal),
        ],
        returns_scalar=True,
    )


def is_close(
    x: IntoExpr,
    y: IntoExpr,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    null_equal: bool = False,
) -> pl.Expr:
    """Compares the relative equality of the inputs.

    Parameters
    ----------
    x : IntoExpr
    y : IntoExpr
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
    x = _into_expr(x)
    y = _into_expr(y)

    res = x.sub(y).abs().le(pl.lit(atol).add(rtol).mul(y.abs()))

    if null_equal:
        res = res.or_(x.is_null().and_(y.is_null()))

    return res
