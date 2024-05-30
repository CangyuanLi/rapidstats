import numpy as np
import pytest
import scipy.stats
import sklearn.metrics

import rapidstats
from rapidstats._metrics import ConfusionMatrix

SEED = 208
N_ROWS = 1_000
BOOTSTRAP_ITERATIONS = 100

np.random.seed(SEED)


Y_TRUE = np.random.choice([True, False], N_ROWS)
Y_SCORE = np.random.rand(N_ROWS)
Y_PRED = Y_SCORE > 0.5
Y_TRUE_SC = np.full(N_ROWS, True)
Y_SCORE_SC = np.ones(N_ROWS)
Y_PRED_SC = Y_SCORE_SC > 0.5

TRUE_PRED_COMBOS = [
    (Y_TRUE, Y_PRED),
    (Y_TRUE_SC, Y_PRED),
    (Y_TRUE, Y_PRED_SC),
    (Y_TRUE_SC, Y_PRED_SC),
]

TRUE_SCORE_COMBOS = [
    (Y_TRUE, Y_SCORE),
    (Y_TRUE, Y_SCORE_SC),
    (Y_TRUE_SC, Y_SCORE),
    (Y_TRUE_SC, Y_SCORE_SC),
]


def reference_confusion_matrix(y_true, y_pred):
    tn, fp, fn_, tp = sklearn.metrics.confusion_matrix(
        y_true, y_pred, labels=[False, True]
    ).ravel()

    p = tn + fn_
    n = fp + tn
    tpr = tp / p
    fnr = 1.0 - tpr
    fpr = fp / n
    tnr = 1.0 - fpr
    precision = tp / (tp + fp)
    false_omission_rate = fn_ / (fn_ + tn)
    plr = tpr / fpr
    nlr = fnr / tnr
    npv = 1.0 - false_omission_rate
    fdr = 1.0 - precision
    prevalence = p / (p + n)
    informedness = tpr + tnr - 1.0
    prevalence_threshold = (np.sqrt(tpr * fpr) - fpr) / (tpr - fpr)
    markedness = precision - false_omission_rate
    dor = plr / nlr
    balanced_accuracy = (tpr + tnr) / 2.0
    f1 = (2.0 * precision * tpr) / (precision + tpr)
    folkes_mallows_index = np.sqrt(precision * tpr)
    mcc = np.sqrt(tpr * tnr * precision * npv) - np.sqrt(
        fnr * fpr * false_omission_rate * fdr
    )
    acc = (tp + tn) / (p + n)
    threat_score = tp / (tp + fn_ + fp)

    return ConfusionMatrix(
        tn,
        fp,
        fn_,
        tp,
        tpr,
        fpr,
        fnr,
        tnr,
        prevalence,
        prevalence_threshold,
        informedness,
        precision,
        false_omission_rate,
        plr,
        nlr,
        acc,
        balanced_accuracy,
        f1,
        folkes_mallows_index,
        mcc,
        threat_score,
        markedness,
        fdr,
        npv,
        dor,
    )


@pytest.mark.parametrize("y_true,y_pred", TRUE_PRED_COMBOS)
def test_confusion_matrix(y_true, y_pred):
    ref = reference_confusion_matrix(y_true, y_pred).__dict__
    fs = rapidstats.confusion_matrix(y_true, y_pred).__dict__

    pytest.approx(list(fs.values())) == list(ref.values())


def reference_roc_auc(y_true, y_score):
    try:
        return sklearn.metrics.roc_auc_score(y_true, y_score)
    except ValueError:
        return float("nan")


@pytest.mark.parametrize("y_true,y_score", TRUE_SCORE_COMBOS)
def test_roc_auc(y_true, y_score):
    ref = reference_roc_auc(y_true, y_score)
    fs = rapidstats.roc_auc(y_true, y_score)

    pytest.approx(fs) == ref


def reference_max_ks(y_true, y_score):
    class0 = y_score[~y_true]
    class1 = y_score[y_true]

    try:
        return scipy.stats.ks_2samp(class0, class1)
    except ValueError:
        return float("nan")


@pytest.mark.parametrize("y_true,y_score", TRUE_SCORE_COMBOS)
def test_max_ks(y_true, y_score):
    ref = reference_max_ks(y_true, y_score)
    fs = rapidstats.max_ks(y_true, y_score)

    pytest.approx(fs) == ref
