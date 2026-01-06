from __future__ import annotations

import itertools
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union

import narwhals.stable.v1 as nw
import narwhals.stable.v1.typing as nwt
import polars as pl
from tqdm.auto import tqdm

CorrelationMethod = Literal["pearson", "spearman"]
CorrelationMatrixFormat = Literal["wide", "long"]


@dataclass
class CorrelationBatchOptions:
    """Options to control batching in [rapidstats.correlation_matrix][].

    Parameters
    ----------
    batch_size : int | float, optional
        The number of combinations (where a combination is a pair of features) to
        compute each batch. If a float between 0 and 1, it is interpreted as a percent,
        by default = 0.1
    cache_dir : str | Path | None, optional
        The directory to save out the results of each batch. If None, creates a folder
        called "__rapidstats_correlation_cache__" in the current working directory, by
        default None
    start_iteration : int | None, optional
        The iteration to start at. If None, will start at the latest iteration available
        in `cache_dir`, by default None
    delete_ok : bool, optional
        Whether to delete `cache_dir` after the correlation matrix is computed, by
        default False
    quiet : bool
        Whether to print progress information, by default False
    """

    batch_size: int | float = 0.1
    cache_dir: str | Path | None = None
    start_iteration: int | None = None
    delete_ok: bool = False
    quiet: bool = False


def _pairs(l1, l2) -> list[tuple]:
    return [(x, y) for x in l1 for y in l2]


def _corr_expr(c1, c2, method: CorrelationMethod) -> pl.Expr:
    return pl.corr(c1, c2, method=method).alias(f"{c1}_{c2}")


def _prepare_inputs(data, l1, l2):
    pf = nw.from_native(data).to_polars().lazy()

    if l1 is None and l2 is None:
        original = pf.collect_schema().names()
        new_columns = [f"{i}" for i, _ in enumerate(original)]
        combinations = itertools.combinations(new_columns, r=2)
        l1 = original[:-1]
        l2 = original[1:]
    elif l1 is not None and l2 is None:
        # In this case the user should pass in the combinations directly as a list of
        # 2-tuples.
        original = set()
        for a, b in l1:
            original.add(a)
            original.add(b)
        original = list(original)
        mapper = {name: f"{i}" for i, name in enumerate(original)}
        combinations = [(mapper[a], mapper[b]) for a, b in l1]
        new_columns = list(mapper.values())

        l1 = original
        l2 = original
    else:
        assert l1 is not None
        assert l2 is not None
        valid_cols = set(pf.collect_schema().names())
        l1 = [c for c in l1 if c in valid_cols]
        l2 = [c for c in l2 if c in valid_cols]

        new_l1 = [f"l{i}" for i, _ in enumerate(l1)]
        new_l2 = [f"r{i}" for i, _ in enumerate(l2)]
        new_columns = new_l1 + new_l2
        combinations = _pairs(new_l1, new_l2)
        original = l1 + l2

    return pf, original, new_columns, combinations


def _correlation_matrix(
    pf: pl.LazyFrame,
    original: list[str],
    new_columns: list[str],
    combinations: list[tuple[str, str]],
    method: CorrelationMethod,
    index: str,
    format: CorrelationMatrixFormat,
):
    old_to_new_mapper = {old: new for old, new in zip(original, new_columns)}
    new_to_old_mapper = {new: old for new, old in zip(new_columns, original)}

    corr_mat = (
        pf.select(original)
        .rename(old_to_new_mapper)
        .select(_corr_expr(c1, c2, method=method) for c1, c2 in combinations)
        .unpivot()
        .with_columns(pl.col("variable").str.split("_"))
        .with_columns(
            pl.col("variable").list.get(0).alias("c1"),
            pl.col("variable").list.get(1).alias("c2"),
        )
        .drop("variable")
        .rename({"value": "correlation"})
        .select("c1", "c2", "correlation")
        .collect()
    )

    if format == "wide":
        # Why do this contortion to rename things? I think I thought that replace might
        # be slow if I rename the long dataframe. I don't know if that's actually true
        # though...
        corr_mat = corr_mat.pivot(index="c2", on="c1", values="correlation")
        new_row_names = corr_mat["c2"]
        corr_mat = corr_mat.drop("c2")

        # Set the column names
        valid_old_names = [new_to_old_mapper[c] for c in corr_mat.columns]
        corr_mat.columns = valid_old_names

        # Set the row names
        valid_old_row_names = [new_to_old_mapper[c] for c in new_row_names]
        corr_mat = corr_mat.with_columns(pl.Series(index, valid_old_row_names)).select(
            index, *valid_old_names
        )

        return corr_mat
    elif format == "long":
        return corr_mat.with_columns(
            pl.col("c1").replace_strict(new_to_old_mapper),
            pl.col("c2").replace_strict(new_to_old_mapper),
        )
    else:
        raise ValueError(f"Unexpected format {format}")


def _batched(iterable, n: int):
    if n < 1:
        raise ValueError("n must be at least 1")

    iterator = iter(iterable)
    while batch := list(itertools.islice(iterator, n)):
        yield batch


def _batched_correlation_matrix(
    pf,
    original,
    new_columns,
    combinations,
    batch_options: CorrelationBatchOptions,
    method: CorrelationMethod = "pearson",
    format: CorrelationMatrixFormat = "wide",
    index: str = "",
):
    batch_size = batch_options.batch_size
    cache_dir = batch_options.cache_dir
    start_iteration = batch_options.start_iteration
    delete_ok = batch_options.delete_ok
    quiet = batch_options.quiet

    if cache_dir is None:
        cache_dir = Path.cwd() / "__rapidstats_correlation_cache__"

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if start_iteration is None:
        # Try to pick up where we left off
        try:
            start_iteration = max(int(f.stem) for f in cache_dir.glob("*.parquet")) + 1
        except ValueError:
            start_iteration = 0

    combinations = list(combinations)

    if 0 < batch_size < 1:
        batch_size = int(len(combinations) / batch_size)

    if isinstance(batch_size, float):
        raise ValueError(
            "`batch_size` must either be an integer or a float between 0 and 1 representing a percentage."
        )

    batches = list(_batched(combinations, batch_size))[start_iteration:]

    for i, batch in tqdm(
        enumerate(batches, start=start_iteration), total=len(batches), disable=quiet
    ):
        corr_mat = _correlation_matrix(
            pf=pf,
            original=original,
            new_columns=new_columns,
            combinations=batch,
            method=method,
            index="",
            format="long",
        )

        corr_mat.write_parquet(cache_dir / f"{i}.parquet")

    corr_mats = []
    for f in cache_dir.glob("*.parquet"):
        corr_mat = pl.scan_parquet(f)
        corr_mats.append(corr_mat)

    corr_mat = pl.concat(corr_mats, how="vertical_relaxed").collect()

    if format == "wide":
        corr_mat = corr_mat.pivot(on="c1", index="c2", values="correlation").rename(
            {"c2": index}
        )

    if delete_ok and cache_dir.exists():
        shutil.rmtree(cache_dir)

    return corr_mat


def correlation_matrix(
    data: nwt.IntoFrame,
    l1: Optional[Union[list[str], list[tuple[str, str]]]] = None,
    l2: Optional[list[str]] = None,
    method: CorrelationMethod = "pearson",
    format: CorrelationMatrixFormat = "wide",
    index: str = "",
    batch_options: CorrelationBatchOptions | None = None,
) -> pl.DataFrame:
    """
    !!! warning

        If you know that your data has no nulls, you should use `np.corrcoef` instead.
        While this function will return the correct result and is reasonably fast,
        computing the null-aware correlation matrix will always be slower than assuming
        that there are no nulls.

    Compute the null-aware correlation matrix between two lists of columns. If both
    lists are None, then the correlation matrix is over all columns in the input
    DataFrame. If `l1` is not None, and is a list of 2-tuples, `l1` is interpreted
    as the combinations of columns to compute the correlation for.

    Parameters
    ----------
    data : nwt.IntoFrameT
        The input data
    l1 : Union[list[str], list[tuple[str, str]]], optional
        A list of columns to appear as the columns of the correlation matrix,
        by default None
    l2 : list[str], optional
        A list of columns to appear as the rows of the correlation matrix,
        by default None
    method : Literal["pearson", "spearman"], optional
        How to calculate the correlation, by default "pearson"
    format : Literal["wide", "long"], optional
        The format the correlation matrix is returned in. If "wide", it is the classic
        correlation matrix. If "long", it is a DataFrame with the columns `c1`, `c2`,
        and `correlation`, by default "wide"

        !!! Added in version 0.4.0
    index : str, optional
        The name of the `l2` column in the final output. Ignored if the format is
        "long", by default ""

        !!! Added in version 0.2.0
        !!! Renamed from "index_name" to "index" in version 0.4.0
    batch_options : CorrelationBatchOptions | None, optional
        Parameters that control how to compute the correlation matrix in a batched
        manner. If None, does not use batching, by default None

    Returns
    -------
    pl.DataFrame
        A correlation matrix with `l1` as the columns and `l2` as the rows

    Added in version 0.0.24
    -----------------------
    """
    pf, original, new_columns, combinations = _prepare_inputs(data, l1, l2)

    if batch_options is None:
        return _correlation_matrix(
            pf,
            original=original,
            new_columns=new_columns,
            combinations=combinations,
            method=method,
            index=index,
            format=format,
        )
    else:
        return _batched_correlation_matrix(
            pf=pf,
            original=original,
            new_columns=new_columns,
            combinations=combinations,
            batch_options=batch_options,
            method=method,
            format=format,
            index=index,
        )
