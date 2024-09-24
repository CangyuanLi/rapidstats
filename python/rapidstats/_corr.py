import itertools
from typing import Optional, Union

import polars as pl
import polars.selectors as cs
from polars._typing import CorrelationMethod, FrameInitTypes
from polars.interchange.protocol import SupportsInterchange

ConvertibleToPolars = Union[SupportsInterchange, FrameInitTypes]


def _pairs(l1, l2) -> list[tuple]:
    return [(x, y) for x in l1 for y in l2]


def _corr_expr(c1, c2, method: CorrelationMethod) -> pl.Expr:
    return pl.corr(c1, c2, method=method).alias(f"{c1}_{c2}")


def _to_polars(
    data: Union[pl.LazyFrame, pl.DataFrame, ConvertibleToPolars]
) -> Union[pl.LazyFrame, pl.DataFrame]:
    if isinstance(data, pl.DataFrame):
        return data
    elif isinstance(data, pl.LazyFrame):
        return data
    elif hasattr(data, "to_polars"):
        return data.to_polars()

    try:
        return pl.DataFrame(data)
    except TypeError:
        if hasattr(data, "__dataframe__"):
            return pl.from_dataframe(data)
        else:
            raise TypeError("`data` must be convertible to a Polars DataFrames")


def correlation_matrix(
    data: Union[pl.LazyFrame, pl.DataFrame, ConvertibleToPolars],
    l1: Optional[Union[list[str], list[tuple[str, str]]]] = None,
    l2: Optional[list[str]] = None,
    method: CorrelationMethod = "pearson",
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
    data : Union[pl.LazyFrame, pl.DataFrame, ConvertibleToPolars]
        The input DataFrame. It must be either a Polars Frame or something convertible
        to a Polars Frame.
    l1 : Union[list[str], list[tuple[str, str]]], optional
        A list of columns to appear as the columns of the correlation matrix,
        by default None
    l2 : list[str], optional
        A list of columns to appear as the rows of the correlation matrix,
        by default None
    method : CorrelationMethod, optional
        How to calculate the correlation, by default "pearson"

    Returns
    -------
    pl.DataFrame
        A correlation matrix with `l1` as the columns and `l2` as the rows
    """
    # pl.corr works with nulls but NOT NaNs
    pf = _to_polars(data).select(cs.numeric() | cs.boolean()).fill_nan(None)

    if l1 is None and l2 is None:
        original = pf.columns
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
        valid_cols = set(pf.columns)
        l1 = [c for c in l1 if c in valid_cols]
        l2 = [c for c in l2 if c in valid_cols]

        new_l1 = [f"l{i}" for i, _ in enumerate(l1)]
        new_l2 = [f"r{i}" for i, _ in enumerate(l2)]
        new_columns = new_l1 + new_l2
        combinations = _pairs(new_l1, new_l2)
        original = l1 + l2

    old_to_new_mapper = {old: new for old, new in zip(original, new_columns)}
    new_to_old_mapper = {new: old for new, old in zip(new_columns, original)}

    corr_mat = (
        pf.lazy()
        .select(original)
        .rename(old_to_new_mapper)
        .select(_corr_expr(c1, c2, method=method) for c1, c2 in combinations)
        .unpivot()
        .with_columns(pl.col("variable").str.split("_"))
        .with_columns(
            pl.col("variable").list.get(0).alias("c1"),
            pl.col("variable").list.get(1).alias("c2"),
        )
        .drop("variable")
        .collect()
        .pivot(index="c2", on="c1", values="value")
    )

    new_row_names = corr_mat["c2"]
    corr_mat = corr_mat.drop("c2")

    # Set the column names
    valid_old_names = [new_to_old_mapper[c] for c in corr_mat.columns]
    corr_mat.columns = valid_old_names

    # Set the row names
    valid_old_row_names = [new_to_old_mapper[c] for c in new_row_names]
    corr_mat = corr_mat.with_columns(pl.Series("", valid_old_row_names)).select(
        "", *valid_old_names
    )

    return corr_mat
