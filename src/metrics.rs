use crate::bootstrap;
use ndarray::{s, ArrayView1};
use polars::prelude::*;

pub type ConfusionMatrixArray = [f64; 25];

const BINARY_CM_VALUES: [i32; 4] = [0, 1, 2, 3];

pub fn base_confusion_matrix(df: DataFrame) -> DataFrame {
    df.lazy()
        .select([(lit(2) * col("y_true") + col("y_pred")).alias("y")])
        .collect()
        .unwrap()
}

pub fn confusion_matrix(base_cm: DataFrame) -> ConfusionMatrixArray {
    let value_counts = base_cm["y"].value_counts(false, false).unwrap();

    let value_counts = if value_counts.height() < 4 {
        let seen: Vec<i32> = value_counts["y"]
            .i32()
            .unwrap()
            .iter()
            .map(|x| x.unwrap())
            .collect();
        let not_seen: Vec<i32> = BINARY_CM_VALUES
            .into_iter()
            .filter(|x| !seen.contains(x))
            .collect();

        let zeros = vec![0u32; not_seen.len()];

        value_counts
            .vstack(&df!("y" => not_seen, "count" => zeros).unwrap())
            .unwrap()
    } else {
        value_counts
    };

    let s: Vec<f64> = value_counts
        .sort(["y"], Default::default())
        .unwrap()
        .column("count")
        .unwrap()
        .u32()
        .unwrap()
        .iter()
        .map(|x| x.unwrap() as f64)
        .collect();

    let tn = s[0];
    let fp = s[1];
    let fn_ = s[2];
    let tp = s[3];

    let p = tp + fn_;
    let n = fp + tn;
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
    let f1 = (2.0 * precision * tpr) / (precision + tpr);
    let folkes_mallows_index = (precision * tpr).sqrt();
    let mcc = (tpr * tnr * precision * npv).sqrt() - (fnr * fpr * false_omission_rate * fdr).sqrt();
    let acc = (tp + tn) / (p + n);
    let threat_score = tp / (tp + fn_ + fp);

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
        f1,
        folkes_mallows_index,
        mcc,
        threat_score,
        markedness,
        fdr,
        npv,
        dor,
    ]
    .map(|x| if x.is_infinite() { f64::NAN } else { x })
}

fn transpose_confusion_matrix_results(results: Vec<[f64; 25]>) -> [Vec<f64>; 25] {
    let mut transposed: [Vec<f64>; 25] = Default::default();
    for arr in results {
        for (i, v) in arr.into_iter().enumerate() {
            transposed[i].push(v);
        }
    }

    transposed
}

pub fn bootstrap_confusion_matrix(
    df: DataFrame,
    iterations: u64,
    alpha: f64,
    method: &str,
    seed: Option<u64>,
) -> Vec<bootstrap::ConfidenceInterval> {
    let base_cm = base_confusion_matrix(df);

    let bootstrap_stats =
        bootstrap::run_bootstrap(base_cm.clone(), iterations, seed, confusion_matrix);
    let bs_transposed = transpose_confusion_matrix_results(bootstrap_stats);

    if method == "percentile" {
        bs_transposed
            .into_iter()
            .map(|bs| bootstrap::percentile_interval(bs, alpha))
            .collect::<Vec<bootstrap::ConfidenceInterval>>()
    } else if method == "basic" {
        let original_stats = confusion_matrix(base_cm.clone());

        original_stats
            .into_iter()
            .zip(bs_transposed)
            .map(|(original_stat, bs)| bootstrap::basic_interval(original_stat, bs, alpha))
            .collect::<Vec<bootstrap::ConfidenceInterval>>()
    } else if method == "BCa" {
        let original_stats = confusion_matrix(base_cm.clone());
        let jacknife_stats = bootstrap::run_jacknife(base_cm, confusion_matrix);
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

// AUC code taken largely from https://github.com/abstractqqq/polars_ds_extension/blob/main/src/num/tp_fp.rs

fn trapz(y: ArrayView1<f64>, x: ArrayView1<f64>) -> f64 {
    let y_s = &y.slice(s![1..]) + &y.slice(s![..-1]);
    let x_d = &x.slice(s![1..]) - &x.slice(s![..-1]);

    0.5 * (x_d.into_iter().zip(y_s).fold(0., |acc, (x, y)| acc + x * y))
}

pub fn roc_auc(df: DataFrame) -> f64 {
    let positive_counts = df["y_true"].sum::<u32>().unwrap_or(0);
    if positive_counts == 0 {
        return f64::NAN;
    }
    let n = df.height() as u32;

    let mut binding = df
        .lazy()
        .group_by([col("y_score")])
        .agg([
            len().alias("cnt"),
            col("y_true").sum().alias("pos_cnt_at_threshold"),
        ])
        .sort(["y_score"], Default::default())
        .with_columns([
            (lit(n) - col("cnt").cum_sum(false) + col("cnt")).alias("predicted_positive"),
            (lit(positive_counts) - col("pos_cnt_at_threshold").cum_sum(false))
                .shift_and_fill(1, positive_counts)
                .alias("tp"),
        ])
        .with_column((col("predicted_positive") - col("tp")).alias("fp"))
        .with_columns([
            col("tp").cast(DataType::Float64),
            col("fp").cast(DataType::Float64),
        ])
        .select([
            (col("tp") / col("tp").first()).alias("tpr"),
            (col("fp") / col("fp").first()).alias("fpr"),
        ])
        .collect()
        .unwrap();

    let aligned = binding.as_single_chunk();

    -trapz(
        aligned["tpr"].f64().unwrap().to_ndarray().unwrap(),
        aligned["fpr"].f64().unwrap().to_ndarray().unwrap(),
    )
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
        .mean()
        .unwrap_or(f64::NAN)
}

pub fn mean(df: DataFrame) -> f64 {
    df["y"].mean().unwrap_or(f64::NAN)
}

pub fn adverse_impact_ratio(df: DataFrame) -> f64 {
    let is_protected = df["protected"].bool().unwrap();
    let is_control = df["control"].bool().unwrap();
    let y_pred = df["y_pred"].bool().unwrap();
    let protected = y_pred.filter(is_protected).unwrap();
    let control = y_pred.filter(is_control).unwrap();

    protected.mean().unwrap_or(f64::NAN) / control.mean().unwrap_or(f64::NAN)
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
