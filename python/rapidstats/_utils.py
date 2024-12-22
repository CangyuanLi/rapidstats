import concurrent.futures
import multiprocessing
from typing import Literal, Union

import polars as pl
from polars.interchange.protocol import SupportsInterchange
from polars.series.series import ArrayLike
from tqdm.auto import tqdm

DataFrame = Union[pl.DataFrame, SupportsInterchange]
PolarsFrame = Union[pl.LazyFrame, pl.DataFrame]


def _to_polars(df: DataFrame) -> pl.DataFrame:
    if isinstance(df, pl.DataFrame):
        return df
    elif hasattr(df, "to_polars"):
        return df.to_polars()
    elif hasattr(df, "__dataframe__"):
        return pl.from_dataframe(df)
    else:
        raise ValueError("Input must be convertible to a Polars DataFrame")


def _regression_to_df(y_true: ArrayLike, y_score: ArrayLike) -> pl.DataFrame:
    return (
        pl.DataFrame({"y_true": y_true, "y_score": y_score})
        .with_columns(pl.col("y_true", "y_score").cast(pl.Float64))
        .drop_nulls()
    )


def _y_true_y_score_to_df(y_true: ArrayLike, y_score: ArrayLike) -> pl.DataFrame:
    return (
        pl.DataFrame({"y_true": y_true, "y_score": y_score})
        .with_columns(
            pl.col("y_true").cast(pl.Boolean), pl.col("y_score").cast(pl.Float64)
        )
        .drop_nulls()
    )


def _y_true_y_pred_to_df(y_true: ArrayLike, y_pred: ArrayLike) -> pl.DataFrame:
    return (
        pl.DataFrame({"y_true": y_true, "y_pred": y_pred})
        .with_columns(pl.col("y_true", "y_pred").cast(pl.Boolean))
        .drop_nulls()
    )


def _fill_infinite(
    pf: PolarsFrame, value: Union[pl.Expr, int, float, None] = None
) -> PolarsFrame:
    return pf.with_columns(
        pl.when(pl.selectors.float().is_infinite())
        .then(value)
        .otherwise(pl.selectors.float())
        .name.keep()
    )


def _expr_fill_infinite(
    expr: pl.Expr, value: Union[pl.Expr, int, float, None] = None
) -> pl.Expr:
    return pl.when(expr.is_infinite()).then(value).otherwise(expr)


def _run_concurrent(
    fn,
    iterable,
    executor: Union[
        Literal["threads", "processes"],
        concurrent.futures.ThreadPoolExecutor,
        concurrent.futures.ProcessPoolExecutor,
    ] = "threads",
    preserve_order: bool = False,
    quiet: bool = False,
    **executor_kwargs,
):
    if executor_kwargs.get("max_workers") == 1:
        return [fn(i) for i in tqdm(iterable, disable=quiet)]

    if executor == "threads":
        executor = concurrent.futures.ThreadPoolExecutor(**executor_kwargs)
    elif executor == "processes":
        if "context" not in executor_kwargs:
            executor_kwargs["context"] = multiprocessing.get_context("spawn")
        executor = concurrent.futures.ProcessPoolExecutor(**executor_kwargs)

    if preserve_order:
        with executor as pool:
            res = pool.map(fn, iterable)

        return res

    with executor as pool:
        futures = [pool.submit(fn, i) for i in iterable]
        res = []
        for future in concurrent.futures.as_completed(futures):
            res.append(future.result())

    return res
