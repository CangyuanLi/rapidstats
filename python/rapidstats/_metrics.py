import dataclasses
from typing import Union

import polars as pl
from polars.series.series import ArrayLike

from ._rustystats import (
    _adverse_impact_ratio,
    _brier_loss,
    _confusion_matrix,
    _max_ks,
    _mean,
    _mean_squared_error,
    _roc_auc,
    _root_mean_squared_error,
)
from ._utils import (
    _fill_infinite,
    _regression_to_df,
    _y_true_y_pred_to_df,
    _y_true_y_score_to_df,
)

PolarsFrame = Union[pl.DataFrame, pl.LazyFrame]


@dataclasses.dataclass
class ConfusionMatrix:
    r"""Result object returned by `rapidstats.confusion_matrix`

    Attributes
    ----------
    tn : float
        ↑Count of True Negatives; y_true == False and y_pred == False
    fp : float
        ↓Count of False Positives; y_true == False and y_pred == True
    fn : float
        ↓Count of False Negatives; y_true == True and y_pred == False
    tp : float
        ↑Count of True Positives; y_true == True, y_pred == True
    tpr : float
        ↑True Positive Rate, Recall, Sensitivity; Probability that an actual positive
        will be predicted positive; \( \frac{TP}{TP + FN} \)
    fpr : float
        ↓False Positive Rate, Type I Error; Probability that an actual negative will
        be predicted positive; \( \frac{FP}{FP + TN} \)
    fnr : float
        ↓False Negative Rate, Type II Error; Probability an actual positive will be
        predicted negative; \( \frac{FN}{TP + FN} \)
    tnr : float
        ↑True Negative Rate, Specificity; Probability an actual negative will be
        predicted negative; \( \frac{TN}{FP + TN} \)
    prevalence : float
        Prevalence; Proportion of positive classes; \( \frac{TP + FN}{TN + FP + FN + TP} \)
    prevalence_threshold : float
        Prevalence Threshold; \( \frac{\sqrt{TPR \times FPR} - FPR}{TPR - FPR} \)
    informedness : float
        ↑Informedness, Youden's J; \( TPR + TNR - 1 \)
    precision : float
        ↑Precision, Positive Predicted Value (PPV); Probability a predicted positive was
        actually correct; \( \frac{TP}{TP + FP} \)
    false_omission_rate : float
        ↓False Omission Rate (FOR); Proportion of predicted negatives that were wrong
        \( \frac{FN}{FN + TN} \)
    plr : float
        ↑Positive Likelihood Ratio, LR+; \( \frac{TPR}{FPR} \)
    nlr : float
        Negative Likelihood Ratio, LR-; \( \frac{FNR}{TNR} \)
    acc : float
        ↑Accuracy (ACC); Probability of a correct prediction; \( \frac{TP + TN}{TN + FP + FN + TP} \)
    balanced_accuracy : float
        ↑Balanced Accuracy (BA); \( \frac{TP + TN}{2} \)
    f1 : float
        ↑F1; Harmonic mean of Precision and Recall; \( \frac{2 \times PPV \times TPR}{PPV + TPR} \)
    folkes_mallows_index : float
        ↑Folkes Mallows Index (FM); \( \sqrt{PPV \times TPR} \)
    mcc : float
        ↑Matthew Correlation Coefficient (MCC), Yule Phi Coefficient; \( \sqrt{TPR \times TNR \times PPV \times NPV} - \sqrt{FNR \times FPR \times FOR \times FDR} \)
    threat_score : float
        ↑Threat Score (TS), Critical Success Index (CSI), Jaccard Index; \( \frac{TP}{TP + FN + FP} \)
    markedness : float
        Markedness (MP), deltaP; \( PPV + NPV - 1 \)
    fdr : float
        ↓False Discovery Rate, Proportion of predicted positives that are wrong; \( \frac{FP}{TP + FP} \)
    ↑npv : float
        Negative Predictive Value; Proportion of predicted negatives that are correct; \( \frac{TN}{FN + TN} \)
    dor : float
        Diagnostic Odds Ratio; \( \frac{LR+}{LR-} \)
    """

    tn: float
    fp: float
    fn: float
    tp: float
    tpr: float
    fpr: float
    fnr: float
    tnr: float
    prevalence: float
    prevalence_threshold: float
    informedness: float
    precision: float
    false_omission_rate: float
    plr: float
    nlr: float
    acc: float
    balanced_accuracy: float
    f1: float
    folkes_mallows_index: float
    mcc: float
    threat_score: float
    markedness: float
    fdr: float
    npv: float
    dor: float

    def to_polars(self) -> pl.DataFrame:
        """Convert the dataclass to a long Polars DataFrame with columns `metric` and
        `value`.

        Returns
        -------
        pl.DataFrame
            DataFrame with columns `metric` and `value`
        """
        dct = self.__dict__

        return pl.DataFrame({"metric": dct.keys(), "value": dct.values()})


def confusion_matrix(y_true: ArrayLike, y_pred: ArrayLike) -> ConfusionMatrix:
    """Computes the 25 confusion matrix metrics (TP, FP, TN, FN, TPR, F1, etc.). Please
    see https://en.wikipedia.org/wiki/Confusion_matrix for a list of all confusion
    matrix metrics and their formulas.

    Parameters
    ----------
    y_true : ArrayLike
        Ground truth target
    y_pred : ArrayLike
        Predicted target

    Returns
    -------
    ConfusionMatrix
        Dataclass of 25 confusion matrix metrics
    """
    df = _y_true_y_pred_to_df(y_true, y_pred)

    return ConfusionMatrix(*_confusion_matrix(df))


def roc_auc(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Computes Area Under the Receiver Operating Characteristic Curve.

    Parameters
    ----------
    y_true : ArrayLike
        Ground truth target
    y_score : ArrayLike
        Predicted scores

    Returns
    -------
    float
        ROC-AUC
    """
    df = _y_true_y_score_to_df(y_true, y_score).with_columns(
        pl.col("y_true").cast(pl.Float64)
    )

    return _roc_auc(df)


def max_ks(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Performs the two-sample Kolmogorov-Smirnov test on the predicted scores of the
    ground truth positive and ground truth negative classes. The KS test measures the
    highest distance between two CDFs, so the Max-KS metric measures how well the model
    separates two classes. In pseucode:

    ``` py
    df = Frame(y_true, y_score)
    class0 = df.filter(~y_true)["y_score"]
    class1 = df.filter(y_true)["y_score"]

    ks(class0, class1)
    ```

    Parameters
    ----------
    y_true : ArrayLike
        Ground truth target
    y_score : ArrayLike
        Predicted scores

    Returns
    -------
    float
        Max-KS
    """
    df = _y_true_y_score_to_df(y_true, y_score)

    return _max_ks(df)


def brier_loss(y_true: ArrayLike, y_score: ArrayLike) -> float:
    r"""Computes the Brier loss (smaller is better). The Brier loss measures the mean
    squared difference between the predicted scores and the ground truth target.
    Calculated as:

    \[ \frac{1}{N} \sum_{i=1}^N (yt_i - ys_i)^2 \]

    where \( yt \) is `y_true` and \( ys \) is `y_score`.

    Parameters
    ----------
    y_true : ArrayLike
        Ground truth target
    y_score : ArrayLike
        Predicted scores

    Returns
    -------
    float
        Brier loss
    """
    df = _y_true_y_score_to_df(y_true, y_score)

    return _brier_loss(df)


def mean(y: ArrayLike) -> float:
    """Computes the mean of the input array.

    Parameters
    ----------
    y : ArrayLike
        A 1D-array of numbers

    Returns
    -------
    float
        Mean
    """
    return _mean(pl.DataFrame({"y": y}))


def adverse_impact_ratio(
    y_pred: ArrayLike,
    protected: ArrayLike,
    control: ArrayLike,
) -> float:
    """Computes the ratio of positive predictions for the protected class and the
    control class. The ideal ratio is 1. For example, in an underwriting context, this
    means that the model is equally as likely to approve protected applicants as it is
    unprotected applicants.

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
    float
        Adverse Impact Ratio (AIR)
    """
    return _adverse_impact_ratio(
        pl.DataFrame(
            {"y_pred": y_pred, "protected": protected, "control": control}
        ).cast(pl.Boolean)
    )


def mean_squared_error(y_true: ArrayLike, y_score: ArrayLike) -> float:
    r"""Computes Mean Squared Error (MSE) as

    \[ \frac{1}{N} \sum_{i=1}^{N} (yt_i - ys_i)^2 \]

    where \( yt \) is `y_true` and \( ys \) is `y_score`.

    Parameters
    ----------
    y_true : ArrayLike
        Ground truth target
    y_score : ArrayLike
        Predicted scores

    Returns
    -------
    float
        Mean Squared Error (MSE)
    """
    return _mean_squared_error(_regression_to_df(y_true, y_score))


def root_mean_squared_error(y_true: ArrayLike, y_score: ArrayLike) -> float:
    r"""Computes Root Mean Squared Error (RMSE) as

    \[ \sqrt{\frac{1}{N} \sum_{i=1}^{N} (yt_i - ys_i)^2} \]

    where \( yt \) is `y_true` and \( ys \) is `y_score`.

    Parameters
    ----------
    y_true : ArrayLike
        Ground truth target
    y_score : ArrayLike
        Predicted scores

    Returns
    -------
    float
        Root Mean Squared Error (RMSE)
    """
    return _root_mean_squared_error(_regression_to_df(y_true, y_score))


def _base_confusion_matrix_at_thresholds(pf: PolarsFrame) -> PolarsFrame:
    return (
        pf.sort("threshold", descending=True)
        .with_columns(
            pl.col("y_true").cum_sum().alias("tp"),
            pl.col("y_true").not_().cum_sum().alias("fp"),
        )
        .with_columns(
            pl.col("tp").tail(1).first().alias("total_positives"),
        )
        .with_columns(pl.len().sub(pl.col("total_positives")).alias("total_negatives"))
        .with_columns(
            pl.col("total_positives").sub(pl.col("tp")).alias("fn"),
            pl.col("total_negatives").sub(pl.col("fp")).alias("tn"),
        )
        .with_columns(
            pl.col("tp").add(pl.col("fn")).alias("p"),
            pl.col("fp").add(pl.col("tn")).alias("n"),
        )
        .select("threshold", "tn", "fp", "fn", "tp")
    )


def _full_confusion_matrix_from_base(pf: PolarsFrame) -> PolarsFrame:
    return (
        pf.with_columns(
            pl.col("tp").add(pl.col("fn")).alias("p"),
            pl.col("fp").add(pl.col("tn")).alias("n"),
        )
        .with_columns(
            pl.col("tp").truediv("p").alias("tpr"),
            pl.col("fp").truediv("n").alias("fpr"),
            pl.col("tp").truediv(pl.col("tp").add(pl.col("fp"))).alias("precision"),
            pl.col("fn")
            .truediv(pl.col("fn").add(pl.col("tn")))
            .alias("false_omission_rate"),
            pl.col("p").truediv(pl.col("p").add(pl.col("n"))).alias("prevalence"),
        )
        .with_columns(
            pl.lit(1).sub(pl.col("fpr")).alias("tnr"),
            pl.lit(1).sub(pl.col("precision")).alias("fdr"),
            pl.col("tpr")
            .add(pl.lit(1).sub(pl.col("fpr")))
            .sub(1)
            .alias("informedness"),
            pl.col("precision").sub(pl.col("false_omission_rate")).alias("markedness"),
            pl.lit(2)
            .mul(pl.col("precision"))
            .mul(pl.col("tpr"))
            .truediv(pl.col("precision").add(pl.col("tpr")))
            .alias("f1"),
            (pl.col("precision").mul(pl.col("tpr")))
            .sqrt()
            .alias("folkes_mallows_index"),
            pl.col("tp")
            .add(pl.col("tn"))
            .truediv(pl.col("p").add(pl.col("n")))
            .alias("acc"),
            pl.col("tp")
            .truediv(pl.col("tp").add(pl.col("fn")).add(pl.col("fp")))
            .alias("threat_score"),
        )
        .with_columns(
            pl.lit(1).sub(pl.col("tpr")).alias("fnr"),
            pl.lit(1).sub(pl.col("false_omission_rate")).alias("npv"),
            pl.col("tpr").truediv(pl.col("fpr")).alias("plr"),
        )
        .with_columns(pl.col("fnr").truediv(pl.lit(1).sub(pl.col("fpr"))).alias("nlr"))
        .with_columns(
            pl.col("tpr")
            .mul(pl.col("fpr"))
            .sqrt()
            .sub(pl.col("fpr"))
            .truediv(pl.col("tpr").sub(pl.col("fpr")))
            .alias("prevalence_threshold"),
            pl.col("tpr")
            .mul(pl.lit(1).sub(pl.col("fpr")))
            .mul(pl.col("precision"))
            .mul(pl.col("npv"))
            .sqrt()
            .sub(
                pl.col("fnr")
                .mul(pl.col("fpr"))
                .mul(pl.col("false_omission_rate"))
                .mul(pl.col("fdr"))
                .sqrt()
            )
            .alias("mcc"),
            pl.col("tpr")
            .add(pl.lit(1).sub(pl.col("fpr")))
            .truediv(2)
            .alias("balanced_accuracy"),
            pl.col("plr").truediv(pl.col("nlr")).alias("dor"),
        )
        .drop("p", "n")
        .pipe(_fill_infinite, None)
        .fill_nan(None)
    )


def confusion_matrix_at_thresholds(
    y_true: ArrayLike, y_score: ArrayLike
) -> pl.DataFrame:
    """Compute the confusion matrix at each model score. Equivalent to

    ``` py
    for t in y_score:
        y_pred = y_score >= t
        confusion_matrix(y_true, y_pred)
    ```

    but does so using fast DataFrame operations.

    Parameters
    ----------
    y_true : ArrayLike
        Ground truth target
    y_score : ArrayLike
        Predicted scores

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame of `threshold` and the confusion matrix metrics.
    """
    return (
        pl.LazyFrame({"y_true": y_true, "threshold": y_score})
        .with_columns(pl.col("y_true").cast(pl.Boolean))
        .drop_nulls()
        .pipe(_base_confusion_matrix_at_thresholds)
        .pipe(_full_confusion_matrix_from_base)
        .unique("threshold")
        .collect()
    )
