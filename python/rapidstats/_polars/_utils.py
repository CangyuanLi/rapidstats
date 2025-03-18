from pathlib import Path
from typing import Union

import polars as pl

PolarsFrame = Union[pl.DataFrame, pl.LazyFrame]
IntoExpr = Union[str, pl.Expr]

_PLUGIN_PATH = Path(__file__).resolve().parents[1]


def _into_expr(expr: IntoExpr) -> pl.Expr:
    if isinstance(expr, str):
        return pl.col(expr)
    elif isinstance(expr, pl.Expr):
        return expr
    else:
        raise ValueError("Must be of type `str` or `pl.Expr`")
