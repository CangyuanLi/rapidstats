use core::f64;

use crate::bootstrap;
use polars::prelude::*;

pub type ConfusionMatrixArray = [f64; 27];

pub fn base_confusion_matrix(df: DataFrame) -> DataFrame {
    df.lazy()
        .select([(lit(2) * col("y_true") + col("y_pred")).alias("y")])
        .collect()
        .unwrap()
}

pub fn confusion_matrix(base_cm: DataFrame, beta: f64) -> ConfusionMatrixArray {
    let mut s = [0u32; 4];
    for i in base_cm["y"]
        .cast(&DataType::UInt64)
        .unwrap()
        .u64()
        .unwrap()
        .into_no_null_iter()
    {
        s[i as usize] += 1;
    }

    let tn = s[0] as f64;
    let fp = s[1] as f64;
    let fn_ = s[2] as f64;
    let tp = s[3] as f64;

    let p = tp + fn_;
    let n = fp + tn;
    let total = p + n;
    let tpr = tp / p;
    let fnr = 1.0 - tpr;
    let fpr = fp / n;
    let tnr = 1.0 - fpr;
    let precision = tp / (tp + fp);
    let false_omission_rate = fn_ / (fn_ + tn);
    let plr = tpr / fpr;
    let nlr = fnr / tnr;
    let npv = 1.0 - false_omission_rate;
    let fdr = 1.0 - precision;
    let prevalence = p / (p + n);
    let informedness = tpr + tnr - 1.0;
    let prevalence_threshold = ((tpr * fpr).sqrt() - fpr) / (tpr - fpr);
    let markedness = precision - false_omission_rate;
    let dor = plr / nlr;
    let balanced_accuracy = (tpr + tnr) / 2.0;
    let fbeta = ((1.0 + beta.powi(2)) * precision * tpr) / ((beta.powi(2) * precision) + tpr);
    let folkes_mallows_index = (precision * tpr).sqrt();
    let mcc = (tpr * tnr * precision * npv).sqrt() - (fnr * fpr * false_omission_rate * fdr).sqrt();
    let acc = (tp + tn) / total;
    let threat_score = tp / (tp + fn_ + fp);
    let ppr = (tp + fp) / total;
    let pnr = (tn + fn_) / total;

    [
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
    .map(|x| if x.is_infinite() { f64::NAN } else { x })
}

fn transpose_confusion_matrix_results(results: Vec<[f64; 27]>) -> [Vec<f64>; 27] {
    let mut transposed: [Vec<f64>; 27] = Default::default();
    for arr in results {
        for (i, v) in arr.into_iter().enumerate() {
            transposed[i].push(v);
        }
    }

    transposed
}

pub fn bootstrap_confusion_matrix(
    df: DataFrame,
    beta: f64,
    iterations: u64,
    alpha: f64,
    method: &str,
    seed: Option<u64>,
    n_jobs: Option<usize>,
    chunksize: Option<usize>,
) -> Vec<bootstrap::ConfidenceInterval> {
    let base_cm = base_confusion_matrix(df);

    let bootstrap_stats = bootstrap::run_bootstrap(
        base_cm.clone(),
        iterations,
        seed,
        |x| confusion_matrix(x, beta),
        n_jobs,
        chunksize,
    );
    let bs_transposed = transpose_confusion_matrix_results(bootstrap_stats);

    if method == "standard" {
        bs_transposed
            .into_iter()
            .map(|bs| bootstrap::standard_interval(bs, alpha))
            .collect::<Vec<bootstrap::ConfidenceInterval>>()
    } else if method == "percentile" {
        bs_transposed
            .into_iter()
            .map(|bs| bootstrap::percentile_interval(bs, alpha))
            .collect::<Vec<bootstrap::ConfidenceInterval>>()
    } else if method == "basic" {
        let original_stats = confusion_matrix(base_cm.clone(), beta);

        original_stats
            .into_iter()
            .zip(bs_transposed)
            .map(|(original_stat, bs)| bootstrap::basic_interval(original_stat, bs, alpha))
            .collect::<Vec<bootstrap::ConfidenceInterval>>()
    } else if method == "BCa" {
        let original_stats = confusion_matrix(base_cm.clone(), beta);
        let jacknife_stats = bootstrap::run_jacknife(base_cm, |x| confusion_matrix(x, beta));
        let js_transposed = transpose_confusion_matrix_results(jacknife_stats);

        original_stats
            .into_iter()
            .zip(bs_transposed)
            .zip(js_transposed)
            .map(|((original_stat, bs), js)| bootstrap::bca_interval(original_stat, bs, js, alpha))
            .collect::<Vec<bootstrap::ConfidenceInterval>>()
    } else {
        panic!("Invalid method");
    }
}

pub fn roc_auc(df: DataFrame) -> f64 {
    let df = df.sort(["y_score"], Default::default()).unwrap();
    let y_true = df["y_true"].f64().unwrap();

    let n = y_true.len() as f64;
    let (auc, nfalse) = y_true
        .into_no_null_iter()
        .fold((0.0, 0.0), |(auc, nfalse), y_i| {
            let new_nfalse = nfalse + (1.0 - y_i);
            let new_auc = auc + y_i * new_nfalse;
            (new_auc, new_nfalse)
        });

    auc / (nfalse * (n - nfalse))
}

// Max KS code taken largely from https://github.com/abstractqqq/polars_ds_extension/blob/main/src/stats/ks.rs

fn binary_search_right<T: PartialOrd>(arr: &[T], t: &T) -> Option<usize> {
    let mut left = 0;
    let mut right = arr.len();

    while left < right {
        let mid = left + ((right - left) >> 1);
        if let Some(c) = arr[mid].partial_cmp(t) {
            match c {
                std::cmp::Ordering::Greater => right = mid,
                _ => left = mid + 1,
            }
        } else {
            return None;
        }
    }
    Some(left)
}

fn ks_2samp(v1: &[f64], v2: &[f64]) -> f64 {
    // https://github.com/scipy/scipy/blob/v1.11.3/scipy/stats/_stats_py.py#L8644-L8875

    // v1 and v2 must be sorted
    let n1: f64 = v1.len() as f64;
    let n2: f64 = v2.len() as f64;

    if n1 == 0.0 || n2 == 0.0 {
        return f64::NAN;
    }

    let stats = v1
        .iter()
        .chain(v2.iter())
        .map(|x| {
            (
                (binary_search_right(v1, x).unwrap() as f64) / n1,
                (binary_search_right(v2, x).unwrap() as f64) / n2,
            )
        })
        .fold(f64::MIN, |acc, (x, y)| acc.max((x - y).abs()));

    if stats.is_infinite() {
        f64::NAN
    } else {
        stats
    }
}

pub fn max_ks(df: DataFrame) -> f64 {
    let y_score = df["y_score"].f64().unwrap();
    let y_true = df["y_true"].bool().unwrap();

    ks_2samp(
        y_score
            .filter(y_true)
            .unwrap()
            .sort(false)
            .cont_slice()
            .unwrap(),
        y_score
            .filter(&!y_true)
            .unwrap()
            .sort(false)
            .cont_slice()
            .unwrap(),
    )
}

pub fn brier_loss(df: DataFrame) -> f64 {
    df.lazy()
        .with_column((col("y_true") - col("y_score")).pow(2).alias("x"))
        .collect()
        .unwrap()
        .column("x")
        .unwrap()
        .f64()
        .unwrap()
        .mean()
        .unwrap_or(f64::NAN)
}

pub fn mean(df: DataFrame) -> f64 {
    df["y"].as_series().unwrap().mean().unwrap_or(f64::NAN)
}

pub fn adverse_impact_ratio(df: DataFrame) -> f64 {
    let is_protected = df["protected"].bool().unwrap();
    let is_control = df["control"].bool().unwrap();
    let y_pred = df["y_pred"].bool().unwrap();
    let protected = y_pred.filter(is_protected).unwrap();
    let control = y_pred.filter(is_control).unwrap();

    let res = protected.mean().unwrap_or(f64::NAN) / control.mean().unwrap_or(f64::NAN);

    if res.is_infinite() {
        f64::NAN
    } else {
        res
    }
}

pub fn mean_squared_error(df: DataFrame) -> f64 {
    let y_true = df["y_true"].f64().unwrap();
    let y_score = df["y_score"].f64().unwrap();

    let x = &(y_true - y_score);

    (x * x).mean().unwrap()
}

pub fn root_mean_squared_error(df: DataFrame) -> f64 {
    mean_squared_error(df).sqrt()
}

pub fn r2(df: DataFrame) -> f64 {
    let y_true = df["y_true"].f64().unwrap();
    let y_score = df["y_score"].f64().unwrap();

    let residual = &(y_true - y_score);
    let squared_residual = residual * residual;
    let mean = y_true.mean().unwrap();
    let error = &(y_true - mean);
    let squared_error = error * error;

    1.0 - (squared_residual.sum().unwrap() / squared_error.sum().unwrap())
}
