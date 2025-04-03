from __future__ import annotations

from pathlib import Path
from typing import Union

import polars as pl

PolarsFrame = Union[pl.DataFrame, pl.LazyFrame]
IntoExprColumn = Union[pl.Expr, str]
NumericLiteral = Union[float, int]

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
