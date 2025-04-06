from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

import polars as pl

if TYPE_CHECKING:
    from datetime import date, datetime, time, timedelta
    from decimal import Decimal

PolarsFrame = Union[pl.DataFrame, pl.LazyFrame]
IntoExprColumn = Union[pl.Expr, str]
NumericLiteral = Union[float, int, "Decimal"]
IntoExpr = Union[
    IntoExprColumn, NumericLiteral, bool, bytes, "date", "datetime", "time", "timedelta"
]

_PLUGIN_PATH = Path(__file__).resolve().parents[1]


def _str_to_expr(expr: IntoExprColumn) -> pl.Expr:
    if isinstance(expr, str):
        return pl.col(expr)
    elif isinstance(expr, pl.Expr):
        return expr
    else:
        raise TypeError("Must be of type `str` or `pl.Expr`")


def _numeric_to_expr(expr: IntoExprColumn | NumericLiteral) -> pl.Expr:
    if isinstance(expr, pl.Expr):
        return expr
    elif isinstance(expr, str):
        return pl.col(expr)
    elif isinstance(expr, float) or isinstance(expr, int):
        return pl.lit(expr)
    else:
        raise TypeError("Must be of type `int`, `float`, `str`, or `pl.Expr`")


def _parse_into_expr(
    expr, str_as_lit: bool = False, list_as_series: bool = False
) -> pl.Expr:
    if isinstance(expr, pl.Expr):
        return expr
    elif isinstance(expr, str) and not str_as_lit:
        return pl.col(expr)
    elif isinstance(expr, list) and list_as_series:
        return pl.lit(pl.Series(expr))
    else:
        return pl.lit(expr)


def _parse_inputs_as_iterable(
    inputs: tuple[Any, ...] | tuple[Iterable[Any]],
) -> Iterable[Any]:
    if not inputs:
        return []

    # Treat elements of a single iterable as separate inputs
    if len(inputs) == 1 and _is_iterable(inputs[0]):
        return inputs[0]

    return inputs


def _is_iterable(input: Any | Iterable[Any]) -> bool:
    return isinstance(input, Iterable) and not isinstance(
        input, (str, bytes, pl.Series)
    )


def _parse_into_list_of_exprs(*inputs: IntoExpr | Iterable[IntoExpr]) -> list[pl.Expr]:
    return [_parse_into_expr(e) for e in _parse_inputs_as_iterable(inputs)]
