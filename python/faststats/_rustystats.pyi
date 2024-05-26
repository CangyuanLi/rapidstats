from typing import Optional

import polars as pl

def _confusion_matrix(df: pl.DataFrame) -> list[float]: ...
def _bootstrap_confusion_matrix(
    df: pl.DataFrame, iterations: int, z: float, seed: Optional[int]
) -> list[tuple[float, float, float]]: ...
