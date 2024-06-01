import dataclasses
import functools
from typing import Optional

import numpy as np
import polars as pl
import scipy.stats
from polars.series.series import ArrayLike

from ._rustystats import (
    _bootstrap_brier_loss,
    _bootstrap_confusion_matrix,
    _bootstrap_max_ks,
    _bootstrap_roc_auc,
)
from ._utils import _run_concurrent, _y_true_y_pred_to_df, _y_true_y_score_to_df

ConfidenceInterval = tuple[float, float, float]


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


def _x(i, data, stat_func):
    return stat_func(data.sample(fraction=1, with_replacement=True, seed=i))


def bootstrap(
    data: pl.DataFrame,
    stat_func,
    iterations: int = 1_000,
    confidence: float = 0.95,
    seed: int = None,
    **kwargs,
) -> ConfidenceInterval:
    default = {"executor": "threads", "preserve_order": False}
    for k, v in default.items():
        if k not in kwargs:
            kwargs[k] = v

    func = functools.partial(_x, data=data, stat_func=stat_func)

    if seed is None:
        iterable = (None for _ in range(iterations))
    else:
        iterable = (seed + i for i in range(iterations))

    runs = np.array(_run_concurrent(func, iterable, **kwargs))
    z = scipy.stats.norm.ppf(confidence)

    mean = np.nanmean(runs)
    std = np.nanstd(runs)
    x = z * std / iterations ** (1 / 2)

    return (mean - x, mean, mean + x)


class Bootstrap:
    def __init__(
        self,
        iterations: int = 1_000,
        confidence: float = 0.95,
        seed: Optional[int] = None,
    ) -> None:
        self.iterations = iterations
        self.confidence = confidence
        self.z = scipy.stats.norm.ppf(confidence)
        self.seed = seed

    def run(self, data, stat_func) -> ConfidenceInterval:
        return bootstrap(data, stat_func, self.iterations, self.confidence, self.seed)

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
