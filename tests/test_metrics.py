import math

import numpy as np
import polars as pl
import polars.testing
import pytest
import scipy.stats
import sklearn.metrics

import rapidstats as rs
from rapidstats.metrics import ConfusionMatrix

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
    (Y_TRUE_SC, Y_SCORE),
]

Y_TRUE_REG = np.random.rand(N_ROWS)
Y_SCORE_REG = np.random.rand(N_ROWS)

SAMPLE_WEIGHT = np.random.rand(N_ROWS)
SAMPLE_WEIGHT[:100] = 1.0

PROTECTED = np.random.choice([True, False], N_ROWS)
CONTROL = ~PROTECTED

THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def reference_f1(y_true, y_pred):
    f1 = sklearn.metrics.f1_score(
        y_true, y_pred, labels=[False, True], zero_division=np.nan
    )

    return f1


def reference_confusion_matrix(y_true, y_pred, beta: float = 1.0, sample_weight=None):
    tn, fp, fn_, tp = sklearn.metrics.confusion_matrix(
        y_true, y_pred, labels=[False, True], sample_weight=sample_weight
    ).ravel()

    p = tp + fn_
    n = fp + tn
    tpr = tp / p
    fnr = 1.0 - tpr
    fpr = fp / n
    tnr = 1.0 - fpr
    precision = sklearn.metrics.precision_score(
        y_true,
        y_pred,
        labels=[False, True],
        zero_division=np.nan,
        sample_weight=sample_weight,
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
    balanced_accuracy = (tpr + tnr) / 2
    fbeta = sklearn.metrics.fbeta_score(
        y_true,
        y_pred,
        beta=beta,
        labels=[False, True],
        zero_division=np.nan,
        sample_weight=sample_weight,
    )
    folkes_mallows_index = np.sqrt(precision * tpr)
    mcc = np.sqrt(tpr * tnr * precision * npv) - np.sqrt(
        fnr * fpr * false_omission_rate * fdr
    )
    acc = sklearn.metrics.accuracy_score(y_true, y_pred, sample_weight=sample_weight)
    threat_score = tp / (tp + fn_ + fp)
    ppr = (tp + fp) / (p + n)
    pnr = (tn + fn_) / (p + n)

    return ConfusionMatrix(
        *[
            float("nan") if np.isinf(x) else float(x)
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
                fbeta,
                folkes_mallows_index,
                mcc,
                threat_score,
                markedness,
                fdr,
                npv,
                dor,
                ppr,
                pnr,
            ]
        ]
    )


@pytest.mark.parametrize("y_true,y_pred", TRUE_PRED_COMBOS)
@pytest.mark.parametrize("sample_weight", [None, SAMPLE_WEIGHT])
def test_confusion_matrix(y_true, y_pred, sample_weight):
    ref = reference_confusion_matrix(
        y_true, y_pred, sample_weight=sample_weight
    ).__dict__
    fs = rs.metrics.confusion_matrix(
        y_true, y_pred, sample_weight=sample_weight
    ).__dict__

    assert pytest.approx(list(fs.values()), nan_ok=True) == list(ref.values())


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
        res = (
            rs.Bootstrap(
                iterations=BOOTSTRAP_ITERATIONS,
                method="percentile",
                sampling_method="multinomial",
                seed=SEED + i,
            )
            .confusion_matrix(y_true, y_pred)
            .fbeta
        )
        rs_res.append((res[0], res[2]))

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


@pytest.mark.parametrize("y_true,y_pred", TRUE_PRED_COMBOS)
def test_fbeta(y_true, y_pred):
    for beta in (0.1, 1, 8, 94):
        ref = sklearn.metrics.fbeta_score(
            y_true,
            y_pred,
            beta=beta,
            labels=[False, True],
            zero_division=np.nan,
        )
        res = rs.metrics.confusion_matrix(y_true, y_pred, beta=beta).fbeta

        assert pytest.approx(ref, nan_ok=True) == res


def reference_roc_auc(y_true, y_score, sample_weight):
    try:
        return sklearn.metrics.roc_auc_score(
            y_true, y_score, sample_weight=sample_weight
        )
    except ValueError:
        return float("nan")


@pytest.mark.parametrize("y_true,y_score", TRUE_SCORE_COMBOS)
@pytest.mark.parametrize("sample_weight", [None, SAMPLE_WEIGHT])
def test_roc_auc(y_true, y_score, sample_weight):
    ref = reference_roc_auc(y_true, y_score, sample_weight=sample_weight)
    fs = rs.metrics.roc_auc(y_true, y_score, sample_weight=sample_weight)

    assert pytest.approx(fs, nan_ok=True) == ref


@pytest.mark.parametrize("y_true,y_score", TRUE_SCORE_COMBOS)
def test_average_precision(y_true, y_score):
    ref = sklearn.metrics.average_precision_score(y_true, y_score)
    res = rs.metrics.average_precision(y_true, y_score)

    assert pytest.approx(ref) == res


def reference_max_ks(y_true, y_score):
    class0 = y_score[~y_true]
    class1 = y_score[y_true]

    try:
        return scipy.stats.ks_2samp(class0, class1).statistic
    except ValueError:
        return float("nan")


@pytest.mark.parametrize("y_true,y_score", TRUE_SCORE_COMBOS)
def test_max_ks(y_true, y_score):
    ref = reference_max_ks(y_true, y_score)
    fs = rs.metrics.max_ks(y_true, y_score)

    assert pytest.approx(fs, nan_ok=True) == ref


@pytest.mark.parametrize("y_true,y_score", TRUE_SCORE_COMBOS)
def test_brier_loss(y_true, y_score):
    ref = sklearn.metrics.brier_score_loss(y_true, y_score)
    res = rs.metrics.brier_loss(y_true, y_score)

    assert pytest.approx(res) == ref


@pytest.mark.parametrize("y_true", [Y_TRUE, Y_TRUE_SC])
def test_mean(y_true):
    ref = y_true.mean()
    res = rs.metrics.mean(y_true)

    assert pytest.approx(res) == ref


def test_mean_squared_error():
    res = rs.metrics.mean_squared_error(Y_TRUE_REG, Y_SCORE_REG)
    ref = sklearn.metrics.mean_squared_error(Y_TRUE_REG, Y_SCORE_REG)

    assert pytest.approx(res) == ref


def test_root_mean_squared_error():
    res = rs.metrics.root_mean_squared_error(Y_TRUE_REG, Y_SCORE_REG)
    ref = sklearn.metrics.root_mean_squared_error(Y_TRUE_REG, Y_SCORE_REG)

    assert pytest.approx(res) == ref


def reference_confusion_matrix_at_thresholds(
    y_true, y_score, beta, sample_weight, thresholds
) -> pl.DataFrame:
    if thresholds is None:
        thresholds = y_score
    cms = []
    for t in thresholds:
        cms.append(
            reference_confusion_matrix(y_true, y_score >= t, beta, sample_weight)
            .to_polars()
            .with_columns(pl.lit(t).alias("threshold"))
        )

    return pl.concat(cms, how="vertical_relaxed")


@pytest.mark.parametrize("beta", [0.1, 1, 11])
@pytest.mark.parametrize(
    "sample_weight",
    [
        None,
        SAMPLE_WEIGHT,
    ],
)
@pytest.mark.parametrize("loop_strategy", ["loop", "cum_sum"])
@pytest.mark.parametrize("thresholds", [None, THRESHOLDS])
def test_confusion_matrix_at_thresholds(beta, sample_weight, loop_strategy, thresholds):
    y_true = Y_TRUE
    y_score = Y_SCORE

    ref = (
        reference_confusion_matrix_at_thresholds(
            y_true,
            y_score,
            beta=beta,
            sample_weight=sample_weight,
            thresholds=thresholds,
        )
        .fill_nan(None)
        .sort(["threshold", "metric"])
    )

    res = rs.metrics.confusion_matrix_at_thresholds(
        y_true,
        y_score,
        beta=beta,
        sample_weight=sample_weight,
        strategy=loop_strategy,
        thresholds=thresholds,
    ).sort("threshold", "metric")

    polars.testing.assert_series_equal(ref["value"], res["value"])


def reference_adverse_impact_ratio(
    approved, protected, control, sample_weight
) -> float:
    df = pl.DataFrame(
        {
            "approved": approved,
            "protected": protected,
            "control": control,
            "sample_weight": 1.0 if sample_weight is None else sample_weight,
        }
    )

    p = df.filter(pl.col("protected"))
    appr_rate_protected = np.average(p["approved"], weights=p["sample_weight"])

    c = df.filter(pl.col("control"))
    appr_rate_control = np.average(c["approved"], weights=c["sample_weight"])

    res = float(appr_rate_protected / appr_rate_control)

    if math.isfinite(res):
        return res
    else:
        return None


def reference_adverse_impact_ratio_at_thresholds(
    y_score, protected, control, sample_weight, thresholds
):
    if thresholds is None:
        thresholds = y_score

    res = {"threshold": [], "air": []}
    for t in thresholds:
        air = reference_adverse_impact_ratio(
            y_score < t, protected, control, sample_weight
        )
        res["threshold"].append(t)
        res["air"].append(air)

    return pl.DataFrame(res)


def test_adverse_impact_ratio_all_approved():
    y_pred = [True] * 1_00
    protected = [True] * 50 + [False] * 50
    control = [False] * 50 + [True] * 50

    assert (
        pytest.approx(rs.metrics.adverse_impact_ratio(y_pred, protected, control)) == 1
    )


@pytest.mark.parametrize("sample_weight", [None, SAMPLE_WEIGHT])
def test_adverse_impact_ratio(sample_weight):
    t = 0.5
    ref = reference_adverse_impact_ratio(
        Y_SCORE < t,
        protected=PROTECTED,
        control=CONTROL,
        sample_weight=sample_weight,
    )
    res = rs.metrics.adverse_impact_ratio(
        Y_SCORE < t,
        protected=PROTECTED,
        control=CONTROL,
        sample_weight=sample_weight,
    )
    assert pytest.approx(ref) == res


@pytest.mark.parametrize("sample_weight", [None, SAMPLE_WEIGHT])
@pytest.mark.parametrize("thresholds", [None, THRESHOLDS])
@pytest.mark.parametrize("strategy", ["loop", "cum_sum"])
def test_adverse_impact_ratio_at_thresholds(sample_weight, thresholds, strategy):
    ref = reference_adverse_impact_ratio_at_thresholds(
        Y_SCORE,
        protected=PROTECTED,
        control=CONTROL,
        sample_weight=sample_weight,
        thresholds=thresholds,
    ).sort("threshold")

    res = rs.metrics.adverse_impact_ratio_at_thresholds(
        Y_SCORE,
        protected=PROTECTED,
        control=CONTROL,
        sample_weight=sample_weight,
        thresholds=thresholds,
        strategy=strategy,
    ).sort("threshold")

    polars.testing.assert_series_equal(ref["air"], res["air"])


def reference_predicted_positive_ratio_at_thresholds(
    y_score, sample_weight, thresholds
) -> pl.DataFrame:
    if thresholds is None:
        thresholds = y_score

    res = {"threshold": [], "ppr": []}
    for t in thresholds:
        ppr = float(np.average(y_score >= t, weights=sample_weight))

        if not math.isfinite(ppr):
            ppr = None

        res["ppr"].append(ppr)
        res["threshold"].append(t)

    return pl.DataFrame(res)


@pytest.mark.parametrize("sample_weight", [None, SAMPLE_WEIGHT])
@pytest.mark.parametrize("thresholds", [None, THRESHOLDS])
@pytest.mark.parametrize("strategy", ["loop", "cum_sum"])
def test_predicted_positive_ratio_at_thresholds(sample_weight, thresholds, strategy):
    ref = reference_predicted_positive_ratio_at_thresholds(
        Y_SCORE, sample_weight=sample_weight, thresholds=thresholds
    ).sort("threshold")
    res = rs.metrics.predicted_positive_ratio_at_thresholds(
        Y_SCORE, sample_weight=sample_weight, thresholds=thresholds, strategy=strategy
    ).sort("threshold")

    polars.testing.assert_series_equal(ref["ppr"], res["ppr"])


def test_r2():
    ref = sklearn.metrics.r2_score(Y_TRUE_REG, Y_SCORE_REG)
    res = rs.metrics.r2(Y_TRUE_REG, Y_SCORE_REG)

    assert pytest.approx(res) == ref
