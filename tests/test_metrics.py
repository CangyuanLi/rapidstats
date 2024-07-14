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

Y_TRUE_REG = np.random.rand(N_ROWS)
Y_SCORE_REG = np.random.rand(N_ROWS)


def reference_f1(y_true, y_pred):
    f1 = sklearn.metrics.f1_score(
        y_true, y_pred, labels=[False, True], zero_division=np.nan
    )

    return f1


def reference_confusion_matrix(y_true, y_pred):
    tn, fp, fn_, tp = sklearn.metrics.confusion_matrix(
        y_true, y_pred, labels=[False, True]
    ).ravel()

    p = tp + fn_
    n = fp + tn
    tpr = tp / p
    fnr = 1.0 - tpr
    fpr = fp / n
    tnr = 1.0 - fpr
    precision = sklearn.metrics.precision_score(
        y_true, y_pred, labels=[False, True], zero_division=np.nan
    )
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
    balanced_accuracy = sklearn.metrics.balanced_accuracy_score(y_true, y_pred)
    f1 = reference_f1(y_true, y_pred)
    folkes_mallows_index = sklearn.metrics.fowlkes_mallows_score(y_true, y_pred)
    mcc = sklearn.metrics.matthews_corrcoef(y_true, y_pred)
    acc = sklearn.metrics.accuracy_score(y_true, y_pred)
    threat_score = tp / (tp + fn_ + fp)

    return ConfusionMatrix(
        *[
            float("nan") if np.isinf(x) else x
            for x in [
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
            ]
        ]
    )


@pytest.mark.parametrize("y_true,y_pred", TRUE_PRED_COMBOS)
def test_confusion_matrix(y_true, y_pred):
    ref = reference_confusion_matrix(y_true, y_pred).__dict__
    fs = rapidstats.confusion_matrix(y_true, y_pred).__dict__

    pytest.approx(list(fs.values())) == list(ref.values())


@pytest.mark.parametrize("y_true,y_pred", [(Y_TRUE, Y_PRED)])
def test_bootstrap_f1(y_true, y_pred):
    # Unfortunately, it is difficult to test if I implemented the bootstrap exactly
    # correctly compared to scipy.stats.bootstrap. Even if I use the same seed, the
    # random number generators are different. The best I can do is 1) test if the
    # metrics are the same and 2) if the interval calculations are the same. However,
    # this does mean that there could be something wrong with how I implement the
    # resampling for getting the bootstrap runs. For some measure of confidence, lets
    # run both bootstraps 20 times and take the mean of the results, and see if this
    # mean is roughly similar.
    rs_res = []
    ref_res = []
    for i in range(20):
        rs = (
            rapidstats.Bootstrap(iterations=BOOTSTRAP_ITERATIONS, seed=SEED + i)
            .confusion_matrix(y_true, y_pred)
            .f1
        )
        rs_res.append((rs[0], rs[2]))

        ref = scipy.stats.bootstrap(
            (y_true, y_pred),
            reference_f1,
            n_resamples=BOOTSTRAP_ITERATIONS,
            method="percentile",
            random_state=SEED + i,
        ).confidence_interval
        ref_res.append((ref.low, ref.high))

    rs_lower = np.mean([rs[0] for rs in rs_res])
    rs_upper = np.mean([rs[1] for rs in rs_res])
    ref_lower = np.mean([ref[0] for ref in ref_res])
    ref_upper = np.mean([ref[1] for ref in ref_res])

    assert pytest.approx(rs_lower, rel=1e-2) == ref_lower
    assert pytest.approx(rs_upper, rel=1e-2) == ref_upper


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


@pytest.mark.parametrize("y_true,y_score", TRUE_SCORE_COMBOS)
def test_brier_loss(y_true, y_score):
    ref = sklearn.metrics.brier_score_loss(y_true, y_score)
    res = rapidstats.brier_loss(y_true, y_score)

    pytest.approx(res) == ref


@pytest.mark.parametrize("y_true", [Y_TRUE, Y_TRUE_SC])
def test_mean(y_true):
    ref = y_true.mean()
    res = rapidstats.mean(y_true)

    pytest.approx(res) == ref


def test_mean_squared_error():
    res = rapidstats.mean_squared_error(Y_TRUE_REG, Y_SCORE_REG)
    ref = sklearn.metrics.mean_squared_error(Y_TRUE_REG, Y_SCORE_REG)

    pytest.approx(res) == ref


def test_root_mean_squared_error():
    res = rapidstats.root_mean_squared_error(Y_TRUE_REG, Y_SCORE_REG)
    ref = sklearn.metrics.root_mean_squared_error(Y_TRUE_REG, Y_SCORE_REG)

    pytest.approx(res) == ref
