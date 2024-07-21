from typing import Literal, Optional

import polars as pl

ConfidenceInterval = tuple[float, float, float]
BootstrapMethod = Literal["standard", "percentile", "basic", "BCa"]

def _confusion_matrix(df: pl.DataFrame) -> list[float]: ...
def _bootstrap_confusion_matrix(
    df: pl.DataFrame,
    iterations: int,
    alpha: float,
    method: BootstrapMethod,
    seed: Optional[int],
) -> list[ConfidenceInterval]: ...
def _roc_auc(df: pl.DataFrame) -> float: ...
def _bootstrap_roc_auc(
    df: pl.DataFrame,
    iterations: int,
    alpha: float,
    method: BootstrapMethod,
    seed: Optional[int],
) -> ConfidenceInterval: ...
def _max_ks(df: pl.DataFrame) -> float: ...
def _bootstrap_max_ks(
    df: pl.DataFrame,
    iterations: int,
    alpha: float,
    method: BootstrapMethod,
    seed: Optional[int],
) -> ConfidenceInterval: ...
def _brier_loss(df: pl.DataFrame) -> float: ...
def _bootstrap_brier_loss(
    df: pl.DataFrame,
    iterations: int,
    alpha: float,
    method: BootstrapMethod,
    seed: Optional[int],
) -> ConfidenceInterval: ...
def _mean(df: pl.DataFrame) -> float: ...
def _bootstrap_mean(
    df: pl.DataFrame,
    iterations: int,
    alpha: float,
    method: BootstrapMethod,
    seed: Optional[int],
) -> ConfidenceInterval: ...
def _adverse_impact_ratio(df: pl.DataFrame) -> float: ...
def _bootstrap_adverse_impact_ratio(
    df: pl.DataFrame,
    iterations: int,
    alpha: float,
    method: BootstrapMethod,
    seed: Optional[int],
) -> ConfidenceInterval: ...
def _mean_squared_error(df: pl.DataFrame) -> float: ...
def _bootstrap_mean_squared_error(
    df: pl.DataFrame,
    iterations: int,
    alpha: float,
    method: BootstrapMethod,
    seed: Optional[int],
) -> ConfidenceInterval: ...
def _root_mean_squared_error(df: pl.DataFrame) -> float: ...
def _bootstrap_root_mean_squared_error(
    df: pl.DataFrame,
    iterations: int,
    alpha: float,
    method: BootstrapMethod,
    seed: Optional[int],
) -> ConfidenceInterval: ...
def _standard_interval(
    bootstrap_stats: list[float], alpha: float
) -> ConfidenceInterval: ...
def _percentile_interval(
    bootstrap_stats: list[float], alpha: float
) -> ConfidenceInterval: ...
def _basic_interval(
    original_stat: float, bootstrap_stats: list[float], alpha: float
) -> ConfidenceInterval: ...
def _bca_interval(
    original_stat: float,
    bootstrap_stats: list[float],
    jacknife_stats: list[float],
    alpha: float,
) -> ConfidenceInterval: ...
def _norm_ppf(q: float) -> float: ...
def _norm_cdf(x: float) -> float: ...
