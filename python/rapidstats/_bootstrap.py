import dataclasses
import functools
import math
from typing import Callable, Literal, Optional

import polars as pl
from polars.series.series import ArrayLike

from ._distributions import norm
from ._rustystats import (
    _bca_interval,
    _bootstrap_adverse_impact_ratio,
    _bootstrap_brier_loss,
    _bootstrap_confusion_matrix,
    _bootstrap_max_ks,
    _bootstrap_mean,
    _bootstrap_roc_auc,
    _percentile_interval,
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


def _jacknife(
    df: pl.DataFrame, stat_func: Callable[[pl.DataFrame], float]
) -> list[float]:
    df_height = df.height
    index = pl.Series("index", [i for i in range(df_height)])
    func = functools.partial(_js_func, df=df, index=index, stat_func=stat_func)

    return [
        x
        for x in _run_concurrent(func, range(df_height), quiet=True)
        if not math.isnan(x)
    ]


class Bootstrap:
    def __init__(
        self,
        iterations: int = 1_000,
        confidence: float = 0.95,
        method: Literal["percentile", "BCa"] = "percentile",
        seed: Optional[int] = None,
    ) -> None:
        self.iterations = iterations
        self.confidence = confidence
        self.seed = seed
        self.alpha = (1 - confidence) / 2
        self.method = method

        self._params = {
            "iterations": self.iterations,
            "alpha": self.alpha,
            "method": self.method,
            "seed": self.seed,
        }

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
            return _percentile_interval(bootstrap_stats, self.alpha)
        elif self.method == "bCa":
            original_stat = stat_func(df)
            jacknife_stats = _jacknife(df, stat_func)

            return _bca_interval(
                original_stat, bootstrap_stats, jacknife_stats, self.alpha
            )

    def confusion_matrix(
        self,
        y_true: ArrayLike,
        y_pred: ArrayLike,
    ) -> BootstrappedConfusionMatrix:
        df = _y_true_y_pred_to_df(y_true, y_pred)

        return BootstrappedConfusionMatrix(
            *_bootstrap_confusion_matrix(
                df, self.iterations, self.alpha, self.method, self.seed
            )
        )

    def roc_auc(self, y_true: ArrayLike, y_score: ArrayLike) -> ConfidenceInterval:
        df = _y_true_y_score_to_df(y_true, y_score)

        return _bootstrap_roc_auc(df, **self._params)

    def max_ks(self, y_true: ArrayLike, y_score: ArrayLike) -> ConfidenceInterval:
        df = _y_true_y_score_to_df(y_true, y_score)

        return _bootstrap_max_ks(df, **self._params)

    def brier_loss(self, y_true: ArrayLike, y_score: ArrayLike) -> ConfidenceInterval:
        df = _y_true_y_score_to_df(y_true, y_score)

        return _bootstrap_brier_loss(df, **self._params)

    def mean(self, y: ArrayLike) -> ConfidenceInterval:
        df = pl.DataFrame({"y": y})

        return _bootstrap_mean(df, **self._params)

    def adverse_impact_ratio(
        self, y_pred: ArrayLike, protected: ArrayLike, control: ArrayLike
    ):
        df = pl.DataFrame(
            {"y_pred": y_pred, "protected": protected, "control": control}
        ).cast(pl.Boolean)

        return _bootstrap_adverse_impact_ratio(df, **self._params)
