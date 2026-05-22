import polars as pl
from typing import TypeVar

PolarsFrameT = TypeVar("PolarsFrameT", pl.DataFrame, pl.LazyFrame)
