from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import polars as pl

from ._utils import (
    _PLUGIN_PATH,
    IntoExpr,
    IntoExprColumn,
    NumericLiteral,
    _numeric_to_expr,
    _parse_into_list_of_exprs,
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

    Examples
    --------
    ``` py
    import polars as pl
    import rapidstats.polars as rps

    df = pl.DataFrame({"x": [1, 2, 3], "y": [5, 6, 7]})
    df.select(rps.auc("x", "y"))
    ```
    ``` title="output"
    shape: (1, 1)
    ┌──────┐
    │ x    │
    │ ---  │
    │ f64  │
    ╞══════╡
    │ 12.0 │
    └──────┘
    ```

    Added in version 0.2.0
    ----------------------
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

    Examples
    --------
    ``` py
    import polars as pl
    import rapidstats.polars as rps

    df = pl.DataFrame({"x": [1.0, 1.1], "y": [.999999999, 5]})
    df.select(rps.is_close("x", "y"))
    ```
    ``` title="output"
    shape: (2, 1)
    ┌───────┐
    │ x     │
    │ ---   │
    │ bool  │
    ╞═══════╡
    │ true  │
    │ false │
    └───────┘
    ```

    Added in version 0.2.0
    ----------------------
    """
    x = _numeric_to_expr(x)
    y = _numeric_to_expr(y)

    res = x.sub(y).abs().le(pl.lit(atol).add(rtol).mul(y.abs()))

    if null_equal:
        res = res.or_(x.is_null().and_(y.is_null()))

    return res


def sum_horizontal(
    *exprs: IntoExpr | Iterable[IntoExpr],
    null_method: Literal["kleene", "ignore", "propagate"] = "kleene",
) -> pl.Expr:
    if null_method == "kleene":
        parsed_exprs = _parse_into_list_of_exprs(*exprs)

        return (
            pl.when(pl.all_horizontal(expr.is_null()) for expr in parsed_exprs)
            .then(None)
            .otherwise(pl.sum_horizontal(parsed_exprs))
        )
    elif null_method == "ignore":
        return pl.sum_horizontal(exprs)
    elif null_method == "propagate":
        parsed_exprs = _parse_into_list_of_exprs(*exprs)

        return (
            pl.when(pl.any_horizontal(expr.is_null() for expr in parsed_exprs))
            .then(None)
            .otherwise(pl.sum_horizontal(parsed_exprs))
        )
    else:
        raise ValueError(
            f"Invalid `null_strategy` {null_method}, must be one of `kleene`, `ignore`, or `propagate`"
        )
