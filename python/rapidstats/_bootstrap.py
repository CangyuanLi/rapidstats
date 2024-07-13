from __future__ import annotations

import dataclasses
import functools
import math
from typing import Callable, Literal, Optional

import polars as pl
from polars.series.series import ArrayLike

from ._rustystats import (
    _basic_interval,
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
        """Transform the dataclass to a long Polars dataframe with columns
        `metric`, `lower`, `mean`, and `upper`.

        Returns
        -------
        pl.DataFrame
        """
        dct = self.__dict__
        lower = []
        mean = []
        upper = []
        for l, m, u in dct.values():  # noqa: E741
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
    """Computes a two-sided bootstrap confidence interval of a statistic. The
    process is as follows:

    1. Resample 100% of the data with replacement for `iterations`
    2. Compute the statistic on each resample

    If the method is `percentile`, we stop here and compute the interval of the
    bootstrap distribution that is symmetric about the median and contains
    `confidence` of the bootstrap statistics.

    If the method is `basic`, compute the statistic on the original data and
    generate the "Reverse Percentile Interval."

    If the method is `BCa`,

    3. Compute the statistic on the original data
    4. Compute the statistic on the data with the ith row deleted (jacknife)

    and generate the "Bias Corrected and Accelerated Interval."

    The result of each method will be a three-tuple of (lower, mean, upper).

    Parameters
    ----------
    iterations : int, optional
        How many times to resample the data, by default 1_000
    confidence : float, optional
        The confidence level, by default 0.95
    method : Literal[&quot;percentile&quot;, &quot;basic&quot;, &quot;BCa&quot;], optional
        Whether to return the Percentile, Basic / Reverse Percentile, or
        Bias Corrected and Accelerated Interval, by default "percentile"
    seed : Optional[int], optional
        Seed that controls resampling. Set this to any integer to make results
        reproducible, by default None

    Raises
    ------
    ValueError
        If the method is not one of `percentile`, `basic`, or `BCa`

    Examples
    --------
    ``` py
    import rapidstats
    ci = rapidstats.Bootstrap(seed=208).mean([1, 2, 3])
    ```
    (1.0, 1.9783333333333328, 3.0)
    """

    def __init__(
        self,
        iterations: int = 1_000,
        confidence: float = 0.95,
        method: Literal["percentile", "basic", "BCa"] = "percentile",
        seed: Optional[int] = None,
    ) -> None:
        if method not in ("percentile", "basic", "BCa"):
            raise ValueError(
                f"Invalid confidence interval method `{method}`, only `percentile`, `basic`, and `BCa` are supported",
            )

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
        elif self.method == "basic":
            original_stat = stat_func(df)
            return _basic_interval(original_stat, bootstrap_stats, self.alpha)
        elif self.method == "BCa":
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
            *_bootstrap_confusion_matrix(df, **self._params)
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
