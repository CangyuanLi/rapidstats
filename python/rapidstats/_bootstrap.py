from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Iterable
from typing import Callable, Literal, Optional

import polars as pl
from polars.lazyframe.group_by import LazyGroupBy
from polars.series.series import ArrayLike
from tqdm.auto import tqdm

from ._distributions import norm
from ._metrics import (
    ConfusionMatrixMetric,
    DefaultConfusionMatrixMetrics,
    LoopStrategy,
    PolarsFrame,
    _air_at_thresholds_core,
    _base_confusion_matrix_at_thresholds,
    _full_confusion_matrix_from_base,
    _map_to_thresholds,
    _set_loop_strategy,
)
from ._rustystats import (
    _basic_interval,
    _bca_interval,
    _bootstrap_adverse_impact_ratio,
    _bootstrap_brier_loss,
    _bootstrap_confusion_matrix,
    _bootstrap_max_ks,
    _bootstrap_mean,
    _bootstrap_mean_squared_error,
    _bootstrap_roc_auc,
    _bootstrap_root_mean_squared_error,
    _percentile_interval,
    _standard_interval,
)
from ._utils import (
    _expr_fill_infinite,
    _fill_infinite,
    _regression_to_df,
    _run_concurrent,
    _y_true_y_pred_to_df,
    _y_true_y_score_to_df,
)

ConfidenceInterval = tuple[float, float, float]
StatFunc = Callable[[pl.DataFrame], float]


@dataclasses.dataclass
class BootstrappedConfusionMatrix:
    """Result object returned by `rapidstats.Bootstrap().confusion_matrix`.

    See [rapidstats._metrics.ConfusionMatrix][] for a detailed breakdown of the attributes stored in
    this class. However, instead of storing the statistic, it stores the bootstrapped
    confidence interval as (lower, mean, upper).
    """

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
    ppr: ConfidenceInterval
    pnr: ConfidenceInterval

    def to_polars(self) -> pl.DataFrame:
        """Transform the dataclass to a long Polars DataFrame with columns
        `metric`, `lower`, `mean`, and `upper`.

        Returns
        -------
        pl.DataFrame
            A DataFrame with columns `metric`, `lower`, `mean`, and `upper`
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


def _standard_interval_polars(lf: LazyGroupBy, alpha: float) -> pl.LazyFrame:
    z = norm.ppf(1 - alpha)

    return (
        lf.agg(
            pl.col("value").mean().alias("mean"),
            pl.col("value").std().alias("std"),
        )
        .with_columns(pl.col("std").mul(z).alias("x"))
        .with_columns(
            pl.col("mean").sub(pl.col("x")).alias("lower"),
            pl.col("mean").add(pl.col("x")).alias("upper"),
        )
    )


def _percentile_interval_polars(lf: LazyGroupBy, alpha: float) -> pl.LazyFrame:
    return lf.agg(
        pl.col("value").quantile(alpha).alias("lower"),
        pl.col("value").mean().alias("mean"),
        pl.col("value").quantile(1 - alpha).alias("upper"),
    )


def _basic_interval_polars(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.with_columns(pl.col("original").mul(2).alias("x")).with_columns(
        pl.col("x").sub(pl.col("upper")).alias("lower"),
        pl.col("x").sub(pl.col("lower")).alias("upper"),
    )


class Bootstrap:
    r"""Computes a two-sided bootstrap confidence interval of a statistic. Note that
    \( \alpha \) is then defined as \( \frac{1 - \text{confidence}}{2} \). Regardless
    of method, the result will be a three-tuple of (lower, mean, upper). The process is
    as follows:

    - Resample 100% of the data with replacement for `iterations`
    - Compute the statistic on each resample

    If the method is `standard`,

    - Compute the mean \( \hat{\theta} \) of the bootstrap statistics
    - Compute the standard error of the bootstrap statistics. Note that the standard
    error of any statistic is defined as the standard deviation of its sampling
    distribution.
    - Compute the Z-score

        \[ z_{\alpha} = \phi^{-1}(\alpha) \]

        where \( \phi^{-1} \) is the quantile, inverse CDF, or percent-point function

    Then the "Standard" or "First-Order Normal Approximation" interval is

    \[ \hat{\theta} \pm z_{\alpha} \times \hat{\sigma} \]

    If the method is `percentile`, we stop here and compute the interval of the
    bootstrap distribution that is symmetric about the median and contains
    `confidence` of the bootstrap statistics. Then the "Percentile" interval is

    \[
        [\text{percentile}(\hat{\theta}^{*}, \alpha),
        \text{percentile}(\hat{\theta}^{*}, 1 - \alpha)]
    \]

    where \( \hat{\theta}^{*} \) is the vector of bootstrap statistics.

    If the method is `basic`,

    - Compute the statistic on the original data
    - Compute the "Percentile" interval

    Then the "Basic" or "Reverse Percentile" interval is

    \[
        [2\hat{\theta} - PCI_u,
        2\hat{\theta} - PCI_l,]
    \]

    where \( \hat{\theta} \) is the statistic on the original data, \( PCI_u \) is the
    upper bound of the "Percentile" interval, and \( PCI_l \) is the lower bound of the
    "Percentile" interval.

    If the method is `BCa`,

    - Compute the statistic on the original data \( \hat{\theta} \)
    - Compute the statistic on the data with the \( i^{th} \) row deleted (jacknife)
    - Compute the bias correction factor as

        \[
            \hat{z_0} = \phi^{-1}(
                \frac{\sum_{i=1}^B \hat{\theta_i}^{*} \le \hat{\theta}
                + \sum_{i=1}^B \hat{\theta_i}^{*} \leq \hat{\theta}}{2 * B}
            )
        \]

        where \( \hat{\theta}^{*} \) is the vector of bootstrap statistics and \( B \)
        is the length of that vector.

    - Compute the acceleration factor as

        \[
            \hat{a} = \frac{1}{6} \frac{
                \sum_{i=1}^{N} (\hat{\theta_{(.)}} - \hat{\theta_i})^3
            }{
                \sum_{i=1}^{N} [(\hat{\theta_{(.)}} - \hat{\theta_i})^2]^{1.5}
            }
        \]

        where \( \hat{\theta_{(.)}} \) is the mean of the jacknife statistics and
        \( \hat{\theta_i} \) is the \( i^{th} \) element of the jacknife vector.

    - Compute the lower and upper percentiles as

        \[
            \alpha_l = \phi(
                \hat{z_0} + \frac{\hat{z_0} + z_{\alpha}}{1 - \hat{a}(\hat{z} + z_{\alpha})}
            )
        \]

        and

        \[
            \alpha_u = \phi(
                \hat{z_0} + \frac{
                    \hat{z_0} + z_{1 - \alpha}
                }{
                    1 - \hat{a}(\hat{z} + z_{1-\alpha})
                }
            )
        \]

    Then the "BCa" or "Bias-Corrected and Accelerated" interval is

    \[
        [\text{percentile}(\hat{\theta}^{*}, \alpha_l),
        \text{percentile}(\hat{\theta}^{*}, \alpha_u)]
    \]

    where \( \hat{\theta}^{*} \) is the vector of bootstrap statistics.

    Parameters
    ----------
    iterations : int, optional
        How many times to resample the data, by default 1_000
    confidence : float, optional
        The confidence level, by default 0.95
    method : Literal["standard", "percentile", "basic", "BCa"], optional
        Whether to return the Percentile, Basic / Reverse Percentile, or
        Bias Corrected and Accelerated Interval, by default "percentile"
    seed : Optional[int], optional
        Seed that controls resampling. Set this to any integer to make results
        reproducible, by default None
    n_jobs: Optional[int], optional
        How many threads to run with. None means let the executor decide, and 1 means
        run sequentially, by default None
    chunksize: Optional[int], optional
        The chunksize for each thread. None means let the executor decide, by default
        None

    Raises
    ------
    ValueError
        If the method is not one of `standard`, `percentile`, `basic`, or `BCa`

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
        method: Literal["standard", "percentile", "basic", "BCa"] = "percentile",
        seed: Optional[int] = None,
        n_jobs: Optional[int] = None,
        chunksize: Optional[int] = None,
    ) -> None:
        if method not in ("standard", "percentile", "basic", "BCa"):
            raise ValueError(
                f"Invalid confidence interval method `{method}`, only `standard`, `percentile`, `basic`, and `BCa` are supported",
            )

        self.iterations = iterations
        self.confidence = confidence
        self.seed = seed
        self.alpha = (1 - confidence) / 2
        self.method = method
        self.n_jobs = n_jobs
        self.chunksize = chunksize

        self._params = {
            "iterations": self.iterations,
            "alpha": self.alpha,
            "method": self.method,
            "seed": self.seed,
            "n_jobs": self.n_jobs,
            "chunksize": self.chunksize,
        }

    def run(
        self, df: pl.DataFrame, stat_func: StatFunc, **kwargs
    ) -> ConfidenceInterval:
        """Run bootstrap for an arbitrary function that accepts a Polars DataFrame and
        returns a scalar real number.

        Parameters
        ----------
        df : pl.DataFrame
            The data to pass to `stat_func`
        stat_func : StatFunc
            A callable that takes a Polars DataFrame as its first argument and returns
            a scalar real number.

        Returns
        -------
        ConfidenceInterval
            A tuple of (lower, mean, higher)
        """
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

        if self.method == "standard":
            return _standard_interval(bootstrap_stats, self.alpha)
        elif self.method == "percentile":
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
        else:
            # We shouldn't hit this since we check method in __init__, but it makes the
            # type-checker happy
            raise ValueError("Invalid method")

    def confusion_matrix(
        self,
        y_true: ArrayLike,
        y_pred: ArrayLike,
    ) -> BootstrappedConfusionMatrix:
        """Bootstrap confusion matrix. See [rapidstats.confusion_matrix][] for
        more details.

        Parameters
        ----------
        y_true : ArrayLike
            Ground truth target
        y_pred : ArrayLike
            Predicted target

        Returns
        -------
        BootstrappedConfusionMatrix
            A dataclass of confusion matrix metrics as (lower, mean, upper). See
            [rapidstats._bootstrap.BootstrappedConfusionMatrix][] for more details.
        """
        df = _y_true_y_pred_to_df(y_true, y_pred)

        return BootstrappedConfusionMatrix(
            *_bootstrap_confusion_matrix(df, **self._params)
        )

    def confusion_matrix_at_thresholds(
        self,
        y_true: ArrayLike,
        y_score: ArrayLike,
        thresholds: Optional[list[float]] = None,
        metrics: Iterable[ConfusionMatrixMetric] = DefaultConfusionMatrixMetrics,
        strategy: LoopStrategy = "auto",
    ) -> pl.DataFrame:
        """Bootstrap confusion matrix at thresholds. See
        [rapidstats.confusion_matrix_at_thresholds][] for more details.

        Parameters
        ----------
        y_true : ArrayLike
            Ground truth target
        y_score : ArrayLike
            Predicted scores
        thresholds : Optional[list[float]], optional
            The thresholds to compute `y_pred` at, i.e. y_score >= t. If None,
            uses every score present in `y_score`, by default None
        metrics : Iterable[ConfusionMatrixMetric], optional
            The metrics to compute, by default DefaultConfusionMatrixMetrics
        strategy : LoopStrategy, optional
            Computation method, by default "auto"

        Returns
        -------
        pl.DataFrame
            A DataFrame of `threshold`, `metric`, `lower`, `mean`, and `upper`

        Raises
        ------
        NotImplementedError
            When `strategy` is `cum_sum` and `method` is `BCa`
        """
        df = _y_true_y_score_to_df(y_true, y_score).rename({"y_score": "threshold"})
        final_cols = ["threshold", "metric", "lower", "mean", "upper"]

        strategy = _set_loop_strategy(thresholds, strategy)

        if strategy == "loop":
            cms: list[pl.DataFrame] = []
            for t in tqdm(set(thresholds or y_score)):
                cm = (
                    self.confusion_matrix(df["y_true"], df["threshold"].ge(t))
                    .to_polars()
                    .with_columns(pl.lit(t).alias("threshold"))
                )
                cms.append(cm)

            return pl.concat(cms, how="vertical").with_columns(
                pl.col("lower", "mean", "upper").fill_nan(None)
            )
        elif strategy == "cum_sum":
            if thresholds is None:
                thresholds = df["threshold"].unique()

            def _cm_inner(pf: PolarsFrame) -> pl.LazyFrame:
                return (
                    pf.lazy()
                    .pipe(_base_confusion_matrix_at_thresholds)
                    .pipe(_full_confusion_matrix_from_base)
                    .unique("threshold")
                    .pipe(_map_to_thresholds, thresholds)
                    .drop("_threshold_actual")
                )

            def _cm(i: int) -> pl.LazyFrame:
                sample_df = df.sample(fraction=1, with_replacement=True, seed=i)

                return _cm_inner(sample_df)

            cms: list[pl.LazyFrame] = _run_concurrent(
                _cm,
                (
                    (self.seed + i for i in range(self.iterations))
                    if self.seed is not None
                    else (None for _ in range(self.iterations))
                ),
            )

            lf = (
                pl.concat(cms, how="vertical")
                .select("threshold", *metrics)
                .unpivot(index="threshold")
                .rename({"variable": "metric"})
                .group_by("threshold", "metric")
            )

            if self.method == "standard":
                return (
                    _standard_interval_polars(lf, self.alpha)
                    .select(final_cols)
                    .collect()
                )
            elif self.method == "percentile":
                return (
                    _percentile_interval_polars(lf, self.alpha)
                    .select(final_cols)
                    .collect()
                )
            elif self.method == "basic":
                original = (
                    _cm_inner(df)
                    .select("threshold", *metrics)
                    .pipe(_map_to_thresholds, thresholds)
                    .unpivot(index="threshold")
                    .rename({"variable": "metric", "value": "original"})
                )

                return (
                    _percentile_interval_polars(lf, self.alpha)
                    .join(
                        original,
                        on=["threshold", "metric"],
                        how="left",
                        validate="1:1",
                    )
                    .pipe(_basic_interval_polars)
                    .select(final_cols)
                    .collect()
                )
            elif self.method == "BCa":
                raise NotImplementedError(
                    "BCa is not yet implemented for strategy `cum_sum`."
                )

    def roc_auc(
        self,
        y_true: ArrayLike,
        y_score: ArrayLike,
    ) -> ConfidenceInterval:
        """Bootstrap ROC-AUC. See [rapidstats.roc_auc][] for more details.

        Parameters
        ----------
        y_true : ArrayLike
            Ground truth target
        y_score : ArrayLike
            Predicted scores

        Returns
        -------
        ConfidenceInterval
            A tuple of (lower, mean, upper)
        """
        df = _y_true_y_score_to_df(y_true, y_score).with_columns(
            pl.col("y_true").cast(pl.Float64)
        )

        return _bootstrap_roc_auc(df, **self._params)

    def max_ks(self, y_true: ArrayLike, y_score: ArrayLike) -> ConfidenceInterval:
        """Bootstrap Max-KS. See [rapidstats.max_ks][] for more details.

        Parameters
        ----------
        y_true : ArrayLike
            Ground truth target
        y_score : ArrayLike
            Predicted scores

        Returns
        -------
        ConfidenceInterval
            A tuple of (lower, mean, upper)
        """
        df = _y_true_y_score_to_df(y_true, y_score)

        return _bootstrap_max_ks(df, **self._params)

    def brier_loss(self, y_true: ArrayLike, y_score: ArrayLike) -> ConfidenceInterval:
        """Bootstrap Brier loss. See [rapidstats.brier_loss][] for more details.

        Parameters
        ----------
        y_true : ArrayLike
            Ground truth target
        y_score : ArrayLike
            Predicted scores

        Returns
        -------
        ConfidenceInterval
            A tuple of (lower, mean, upper)
        """
        df = _y_true_y_score_to_df(y_true, y_score)

        return _bootstrap_brier_loss(df, **self._params)

    def mean(self, y: ArrayLike) -> ConfidenceInterval:
        """Bootstrap mean.

        Parameters
        ----------
        y : ArrayLike
            A 1D-array

        Returns
        -------
        ConfidenceInterval
            A tuple of (lower, mean, upper)
        """
        df = pl.DataFrame({"y": y})

        return _bootstrap_mean(df, **self._params)

    def adverse_impact_ratio(
        self, y_pred: ArrayLike, protected: ArrayLike, control: ArrayLike
    ) -> ConfidenceInterval:
        """Bootstrap AIR. See [rapidstats.adverse_impact_ratio][] for more details.

        Parameters
        ----------
        y_pred : ArrayLike
            Predicted target
        protected : ArrayLike
            An array of booleans identifying the protected class
        control : ArrayLike
            An array of booleans identifying the control class

        Returns
        -------
        ConfidenceInterval
            A tuple of (lower, mean, upper)
        """
        df = pl.DataFrame(
            {"y_pred": y_pred, "protected": protected, "control": control}
        ).cast(pl.Boolean)

        return _bootstrap_adverse_impact_ratio(df, **self._params)

    def adverse_impact_ratio_at_thresholds(
        self,
        y_score: ArrayLike,
        protected: ArrayLike,
        control: ArrayLike,
        thresholds: Optional[list[float]] = None,
        strategy: LoopStrategy = "auto",
    ) -> pl.DataFrame:
        """Bootstrap AIR at thresholds. See
        [rapidstats.adverse_impact_ratio_at_thresholds][] for more details.

        Parameters
        ----------
        y_score : ArrayLike
            Predicted scores
        protected : ArrayLike
            An array of booleans identifying the protected class
        control : ArrayLike
            An array of booleans identifying the control class
        thresholds : Optional[list[float]], optional
            The thresholds to compute `is_predicted_negative` at, i.e. y_score < t.
            If None, uses every score present in `y_score`, by default None
        strategy : LoopStrategy, optional
            Computation method, by default "auto"

        Returns
        -------
        pl.DataFrame
            A DataFrame of `threshold`, `lower`, `mean`, and `upper`

        Raises
        ------
        NotImplementedError
            When `strategy` is `cum_sum` and `method` is `BCa`
        """
        df = pl.DataFrame(
            {"y_score": y_score, "protected": protected, "control": control}
        ).with_columns(pl.col("protected", "control").cast(pl.Boolean))

        strategy = _set_loop_strategy(thresholds, strategy)

        if strategy == "loop":
            airs: list[dict[str, float]] = []
            for t in tqdm(set(thresholds or y_score)):
                l, m, u = self.adverse_impact_ratio(
                    df["y_score"].lt(t), df["protected"], df["control"]
                )
                airs.append({"threshold": t, "lower": l, "mean": m, "upper": u})

            return pl.DataFrame(airs).fill_nan(None).pipe(_fill_infinite, None)

        elif strategy == "cum_sum":
            if thresholds is None:
                thresholds = df["y_score"]

            def _air(i: int) -> pl.LazyFrame:
                sample_df = df.sample(fraction=1, with_replacement=True, seed=i)

                return _air_at_thresholds_core(sample_df, thresholds)

            airs: list[pl.LazyFrame] = _run_concurrent(
                _air,
                (
                    (self.seed + i for i in range(self.iterations))
                    if self.seed is not None
                    else (None for _ in range(self.iterations))
                ),
            )
            lf = (
                pl.concat(airs, how="vertical")
                .rename({"air": "value"})
                .with_columns(
                    _expr_fill_infinite(pl.col("value").fill_nan(None)).alias("value")
                )
                .group_by("threshold")
            )

            final_cols = ["threshold", "lower", "mean", "upper"]

            if self.method == "standard":
                return (
                    _standard_interval_polars(lf, self.alpha)
                    .select(final_cols)
                    .collect()
                )
            elif self.method == "percentile":
                return (
                    _percentile_interval_polars(lf, self.alpha)
                    .select(final_cols)
                    .collect()
                )
            elif self.method == "basic":
                original = (
                    _air_at_thresholds_core(df)
                    .rename({"air": "original"})
                    .unique("threshold")
                )

                return (
                    _percentile_interval_polars(lf, self.alpha)
                    .join(original, on="threshold", how="left", validate="1:1")
                    .pipe(_basic_interval_polars)
                    .select(final_cols)
                    .collect()
                )
            elif self.method == "BCa":
                raise NotImplementedError(
                    "BCa not yet implemented for strategy `cum_sum`."
                )

    def mean_squared_error(
        self, y_true: ArrayLike, y_score: ArrayLike
    ) -> ConfidenceInterval:
        r"""Bootstrap MSE. See [rapidstats.mean_squared_error][] for more details.

        Parameters
        ----------
        y_true : ArrayLike
            Ground truth target
        y_score : ArrayLike
            Predicted scores

        Returns
        -------
        ConfidenceInterval
            A tuple of (lower, mean, upper)
        """
        return _bootstrap_mean_squared_error(
            _regression_to_df(y_true, y_score), **self._params
        )

    def root_mean_squared_error(
        self, y_true: ArrayLike, y_score: ArrayLike
    ) -> ConfidenceInterval:
        r"""Bootstrap RMSE. See [rapidstats.root_mean_squared_error][] for more details.

        Parameters
        ----------
        y_true : ArrayLike
            Ground truth target
        y_score : ArrayLike
            Predicted scores

        Returns
        -------
        ConfidenceInterval
            A tuple of (lower, mean, upper)
        """
        return _bootstrap_root_mean_squared_error(
            _regression_to_df(y_true, y_score), **self._params
        )
