from typing import TYPE_CHECKING, Any, TypeVar

import polars as pl

PolarsFrameT = TypeVar("PolarsFrameT", pl.DataFrame, pl.LazyFrame)


if TYPE_CHECKING:
    from polars._typing import ArrayLike as _ArrayLike
else:
    _ArrayLike = Any


ArrayLike = _ArrayLike
