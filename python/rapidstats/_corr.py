import itertools
from typing import Union

import polars as pl
import polars.selectors as cs
from polars.type_aliases import CorrelationMethod

PolarsFrame = Union[pl.DataFrame, pl.LazyFrame]


def _pairs(l1, l2) -> list[tuple]:
    return [(x, y) for x in l1 for y in l2]


def _corr_expr(c1, c2, method: CorrelationMethod):
    return pl.corr(c1, c2, method=method).alias(f"{c1}_{c2}")


def correlation_matrix(
    pf: PolarsFrame,
    l1: list[str] = None,
    l2: list[str] = None,
    method: CorrelationMethod = "pearson",
):
    if l1 is None and l2 is None:
        original = pf.select(cs.numeric() | cs.boolean()).columns
        new_columns = [f"{i}" for i, _ in enumerate(original)]
        combinations = itertools.combinations(new_columns, r=2)
        l1 = original[:-1]
        l2 = original[1:]
    else:
        new_l1 = [f"l{i}" for i, _ in enumerate(l1)]
        new_l2 = [f"r{i}" for i, _ in enumerate(l2)]
        new_columns = new_l1 + new_l2
        combinations = _pairs(new_l1, new_l2)
        original = l1 + l2

    corr_mat = (
        pf.lazy()
        .select(original)
        .rename({old: new for old, new in zip(original, new_columns)})
        .select(_corr_expr(c1, c2, method=method) for c1, c2 in combinations)
        .melt()
        .with_columns(pl.col("variable").str.split("_"))
        .with_columns(
            pl.col("variable").list.get(0).alias("c1"),
            pl.col("variable").list.get(1).alias("c2"),
        )
        .drop("variable")
        .collect()
        .pivot(index="c2", columns="c1", values="value")
        .drop("c2")
    )

    corr_mat.columns = l1
    corr_mat = corr_mat.with_columns(pl.Series("", l2)).select("", *l1)

    return corr_mat
