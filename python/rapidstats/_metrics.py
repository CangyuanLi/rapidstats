import dataclasses

import polars as pl
from polars.series.series import ArrayLike

from ._rustystats import _brier_loss, _confusion_matrix, _max_ks, _roc_auc
from ._utils import _y_true_y_pred_to_df, _y_true_y_score_to_df


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

    def to_polars(self) -> pl.DataFrame:
        dct = self.__dict__

        return pl.DataFrame({"metric": dct.keys(), "value": dct.values()})


def confusion_matrix(y_true: ArrayLike, y_pred: ArrayLike) -> ConfusionMatrix:
    df = _y_true_y_pred_to_df(y_true, y_pred)

    return ConfusionMatrix(*_confusion_matrix(df))


def roc_auc(y_true: ArrayLike, y_score: ArrayLike) -> float:
    df = _y_true_y_score_to_df(y_true, y_score)

    return _roc_auc(df)


def max_ks(y_true: ArrayLike, y_score: ArrayLike) -> float:
    df = _y_true_y_score_to_df(y_true, y_score)

    return _max_ks(df)


def brier_loss(y_true: ArrayLike, y_score: ArrayLike) -> float:
    df = _y_true_y_score_to_df(y_true, y_score)

    return _brier_loss(df)
