from __future__ import annotations

import typing
from typing import Literal

import polars as pl
from polars.series.series import ArrayLike

from .bin import doane, freedman_diaconis, rice, scott, sqrt, sturges

BinMethod = Literal["doane", "fd", "rice", "sturges", "scott", "sqrt"]


def _bin_count(x: pl.Series, bin_count: int | BinMethod) -> int:
    if isinstance(bin_count, int):
        return bin_count
    elif bin_count == "doane":
        return doane(x)
    elif bin_count == "fd":
        return freedman_diaconis(x)
    elif bin_count == "rice":
        return rice(x)
    elif bin_count == "sturges":
        return sturges(x)
    elif bin_count == "scott":
        return scott(x)
    elif bin_count == "sqrt":
        return sqrt(x)
    else:
        raise ValueError(
            f"Invalid argument {bin_count} for `bin_count`; must either be an integer or one of {typing.get_args(BinMethod)}"
        )


def _hist(
    x: pl.Series, bins: list[float] | None, bin_count: int | BinMethod
) -> pl.DataFrame:
    if bins is not None:
        return x.hist(bins=bins, include_category=False)

    return x.hist(bin_count=_bin_count(x, bin_count), include_category=False)


def _psi(
    reference_hist: pl.DataFrame | pl.LazyFrame,
    current_hist: pl.DataFrame | pl.LazyFrame,
    reference_len: int,
    current_len: int,
    epsilon: float | None,
) -> float:
    def _fill_zero(e: pl.Expr, value: float) -> pl.Expr:
        return pl.when(e.eq(0)).then(value).otherwise(e)

    res = (
        reference_hist.lazy()
        .rename({"count": "reference_count"})
        .join(
            current_hist.lazy().rename({"count": "current_count"}),
            on="breakpoint",
            how="full",
            validate="1:1",
            coalesce=True,
            nulls_equal=True,
        )
        .with_columns(
            pl.col("reference_count", "current_count").fill_null(0),
            pl.lit(reference_len).alias("reference_len"),
            pl.lit(current_len).alias("current_len"),
        )
        .with_columns(
            pl.col("reference_count")
            .truediv(pl.col("reference_len"))
            .pipe(_fill_zero, epsilon)
            .alias("reference_pct"),
            pl.col("current_count")
            .truediv(pl.col("current_len"))
            .pipe(_fill_zero, epsilon)
            .alias("current_pct"),
        )
    )

    res = res.select(
        pl.col("current_pct")
        .sub(pl.col("reference_pct"))
        .mul(pl.col("current_pct").truediv(pl.col("reference_pct")).log())
        .sum()
        .alias("res")
    ).collect()

    return res["res"].item()


def _numeric_psi(
    reference: pl.Series,
    current: pl.Series,
    *,
    bins: list[float] | None,
    bin_count: int | BinMethod,
    include_nulls: bool,
    epsilon: float | None,
) -> float:
    def _potentially_add_null_bin_and_len(
        x: pl.Series, x_hist: pl.DataFrame, include_nulls: bool
    ) -> tuple[int, pl.DataFrame]:
        if include_nulls:
            x_len = x.len()
            x_hist = pl.concat(
                [
                    x_hist,
                    pl.DataFrame(
                        {"breakpoint": None, "count": x.null_count()}
                    ).with_columns(pl.col("count").cast(pl.UInt32)),
                ],
                how="vertical",
            )
        else:
            x_len = x.count()

        return x_len, x_hist

    reference_hist = _hist(reference, bins=bins, bin_count=bin_count)
    reference_breakpoints = (
        bins
        if bins is not None
        # polars does not report the leftmost breakpoint, which is set the min
        else [reference.min()] + reference_hist["breakpoint"].to_list()
    )
    reference_len, reference_hist = _potentially_add_null_bin_and_len(
        reference, reference_hist, include_nulls=include_nulls
    )

    current_len, current_hist = _potentially_add_null_bin_and_len(
        current,
        current.hist(bins=reference_breakpoints, include_category=False),
        include_nulls=include_nulls,
    )

    return _psi(
        reference_hist=reference_hist,
        current_hist=current_hist,
        reference_len=reference_len,
        current_len=current_len,
        epsilon=epsilon,
    )


def _categorical_psi(
    reference: pl.Series,
    current: pl.Series,
    include_nulls: bool,
    epsilon: float | None,
) -> float:
    if not include_nulls:
        reference = reference.drop_nulls()
        current = current.drop_nulls()

    reference_hist = reference.value_counts().rename({"reference": "breakpoint"})
    current_hist = current.value_counts().rename({"current": "breakpoint"})

    return _psi(
        reference_hist=reference_hist,
        current_hist=current_hist,
        reference_len=reference.len(),
        current_len=current.len(),
        epsilon=epsilon,
    )


def psi(
    reference: ArrayLike,
    current: ArrayLike,
    *,
    bins: list[float] | None = None,
    bin_count: int | BinMethod = "fd",
    include_nulls: bool = True,
    epsilon: float | None = 1e-4,
) -> float:
    reference = pl.Series("reference", reference)
    current = pl.Series("current", current)

    if reference.dtype.is_numeric() and current.dtype.is_numeric():
        return _numeric_psi(
            reference=reference,
            current=current,
            bins=bins,
            bin_count=bin_count,
            include_nulls=include_nulls,
            epsilon=epsilon,
        )

    return _categorical_psi(
        reference=reference,
        current=current,
        include_nulls=include_nulls,
        epsilon=epsilon,
    )
