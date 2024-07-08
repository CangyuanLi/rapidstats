import dataclasses
import functools
import math
import statistics
from typing import Callable, Literal, Optional

import polars as pl
from polars.series.series import ArrayLike

from ._distributions import norm
from ._rustystats import (
    _bootstrap_adverse_impact_ratio,
    _bootstrap_brier_loss,
    _bootstrap_confusion_matrix,
    _bootstrap_max_ks,
    _bootstrap_mean,
    _bootstrap_roc_auc,
)
from ._utils import _run_concurrent, _y_true_y_pred_to_df, _y_true_y_score_to_df

ConfidenceInterval = tuple[float, float, float]
StatFunc = Callable[[pl.DataFrame], float]


@dataclasses.dataclass
class BootstrappedConfusionMatrix:
    tn: ConfidenceInterval
    fp: ConfidenceInterval
    fn: ConfidenceInterval
    tp: ConfidenceInterval
    tpr: ConfidenceInterval
    fpr: ConfidenceInterval
    fnr: ConfidenceInterval
    tnr: ConfidenceInterval
    prevalence: ConfidenceInterval
    prevalence_threshold: ConfidenceInterval
    informedness: ConfidenceInterval
    precision: ConfidenceInterval
    false_omission_rate: ConfidenceInterval
    plr: ConfidenceInterval
    nlr: ConfidenceInterval
    acc: ConfidenceInterval
    balanced_accuracy: ConfidenceInterval
    f1: ConfidenceInterval
    folkes_mallows_index: ConfidenceInterval
    mcc: ConfidenceInterval
    threat_score: ConfidenceInterval
    markedness: ConfidenceInterval
    fdr: ConfidenceInterval
    npv: ConfidenceInterval
    dor: ConfidenceInterval

    def to_polars(self) -> pl.DataFrame:
        dct = self.__dict__
        lower = []
        mean = []
        upper = []
        for l, m, u in dct.values():
            lower.append(l)
            mean.append(m)
            upper.append(u)

        return pl.DataFrame(
            {
                "metric": dct.keys(),
                "lower": lower,
                "mean": mean,
                "upper": upper,
            }
        )


def _bs_func(i: int, df: pl.DataFrame, stat_func: StatFunc) -> float:
    return stat_func(df.sample(fraction=1, with_replacement=True, seed=i))


def _js_func(i: int, df: pl.DataFrame, index: pl.Series, stat_func: StatFunc) -> float:
    return stat_func(df.filter(index.ne(i)))


def _percentile(a: list[float], q: float) -> float:
    if not a:
        return math.nan

    k = (len(a) - 1) * q
    f = math.floor(k)
    c = math.ceil(k)

    if f == c:
        return a[int(k)]

    d0 = a[int(f)] * (c - k)
    d1 = a[int(c)] * (k - f)

    return d0 + d1


def _percentile_interval(bootstrap_stats: list[float], z: float) -> ConfidenceInterval:
    iterations = len(bootstrap_stats)
    std = statistics.stdev(bootstrap_stats)
    mean = statistics.fmean(bootstrap_stats)
    x = z * std / math.sqrt(iterations)

    return (mean - x, mean, mean + x)


def _jacknife(
    df: pl.DataFrame, stat_func: Callable[[pl.DataFrame], float]
) -> list[float]:
    df_height = df.height
    index = pl.Series("index", [i for i in range(df_height)])
    func = functools.partial(_js_func, df=df, index=index, stat_func=stat_func)

    return [x for x in _run_concurrent(func, range(df_height)) if not math.isnan(x)]


def _bca_interval(
    original_stat: float,
    bootstrap_stats: list[float],
    jacknife_stats: list[float],
    z: tuple[float, float],
) -> ConfidenceInterval:
    z1, z2 = z

    bias_correction_factor = norm.ppf(
        statistics.fmean(x < original_stat for x in bootstrap_stats)
    )
    jacknife_mean = statistics.fmean(jacknife_stats)
    diff = [jacknife_mean - x for x in jacknife_stats]

    acceleration_factor = sum(x**3 for x in diff) / (
        6 * (sum(x**2 for x in diff) ** 1.5)
    )

    lower_p = norm.cdf(
        bias_correction_factor
        + (
            (bias_correction_factor + z1)
            / (1 - acceleration_factor * (bias_correction_factor + z1))
        )
    )

    upper_p = norm.cdf(
        bias_correction_factor
        + (
            (bias_correction_factor + z2)
            / (1 - acceleration_factor * (bias_correction_factor + z2))
        )
    )

    return (
        _percentile(bootstrap_stats, lower_p),
        statistics.fmean(bootstrap_stats),
        _percentile(bootstrap_stats, upper_p),
    )


class Bootstrap:
    def __init__(
        self,
        iterations: int = 1_000,
        confidence: float = 0.95,
        seed: Optional[int] = None,
        method: Literal["percentile", "bCa"] = "percentile",
    ) -> None:
        self.iterations = iterations
        self.confidence = confidence
        self.seed = seed
        self.z = (math.nan, math.nan)

        if method == "percentile":
            self.z[0] = norm.ppf(confidence)
        elif method == "bCa":
            alpha = (1.0 - confidence) / 2.0
            self.z[0] = norm.ppf(alpha)
            self.z[1] = norm.ppf(1 - alpha)

        self.method = method

    def run(
        self, df: pl.DataFrame, stat_func: StatFunc, **kwargs
    ) -> ConfidenceInterval:
        default = {"executor": "threads", "preserve_order": False}
        for k, v in default.items():
            if k not in kwargs:
                kwargs[k] = v

        func = functools.partial(_bs_func, df=df, stat_func=stat_func)

        if self.seed is None:
            iterable = (None for _ in range(self.iterations))
        else:
            iterable = (self.seed + i for i in range(self.iterations))

        bootstrap_stats = [
            x for x in _run_concurrent(func, iterable, **kwargs) if not math.isnan(x)
        ]

        if len(bootstrap_stats) == 0:
            return (math.nan, math.nan, math.nan)

        if self.method == "percentile":
            return _percentile_interval(bootstrap_stats, self.z[0])
        elif self.method == "bCa":
            original_stat = stat_func(df)
            jacknife_stats = _jacknife(df, stat_func)

            return _bca_interval(original_stat, bootstrap_stats, jacknife_stats, self.z)

    def confusion_matrix(
        self,
        y_true: ArrayLike,
        y_pred: ArrayLike,
    ) -> BootstrappedConfusionMatrix:
        df = _y_true_y_pred_to_df(y_true, y_pred)

        return BootstrappedConfusionMatrix(
            *_bootstrap_confusion_matrix(df, self.iterations, self.z, self.seed)
        )

    def roc_auc(self, y_true: ArrayLike, y_score: ArrayLike) -> ConfidenceInterval:
        df = _y_true_y_score_to_df(y_true, y_score)

        return _bootstrap_roc_auc(df, self.iterations, self.z, self.seed)

    def max_ks(self, y_true: ArrayLike, y_score: ArrayLike) -> ConfidenceInterval:
        df = _y_true_y_score_to_df(y_true, y_score)

        return _bootstrap_max_ks(df, self.iterations, self.z, self.seed)

    def brier_loss(self, y_true: ArrayLike, y_score: ArrayLike) -> ConfidenceInterval:
        df = _y_true_y_score_to_df(y_true, y_score)

        return _bootstrap_brier_loss(df, self.iterations, self.z, self.seed)

    def mean(self, y: ArrayLike) -> ConfidenceInterval:
        df = pl.DataFrame({"y": y})

        return _bootstrap_mean(df, self.iterations, self.z, self.seed)

    def adverse_impact_ratio(
        self, y_pred: ArrayLike, protected: ArrayLike, control: ArrayLike
    ):
        df = pl.DataFrame(
            {"y_pred": y_pred, "protected": protected, "control": control}
        ).cast(pl.Boolean)

        return _bootstrap_adverse_impact_ratio(
            df,
            self.iterations,
            self.z,
            self.seed,
        )
