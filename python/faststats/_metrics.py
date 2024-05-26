import dataclasses

import polars as pl
from polars.series.series import ArrayLike

from ._rustystats import _confusion_matrix


@dataclasses.dataclass
class ConfusionMatrix:
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


def confusion_matrix(y_true: ArrayLike, y_pred: ArrayLike) -> ConfusionMatrix:
    df = pl.DataFrame({"y_true": y_true, "y_pred": y_pred}).with_columns(
        pl.col("y_true", "y_pred").cast(pl.Boolean)
    )

    return ConfusionMatrix(*_confusion_matrix(df))
