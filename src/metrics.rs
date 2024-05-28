use crate::bootstrap;
use ndarray::{s, ArrayView1};
use polars::prelude::*;

pub type ConfusionMatrixArray = [f64; 25];

pub fn base_confusion_matrix(df: DataFrame) -> DataFrame {
    df.lazy()
        .select([(lit(2) * col("y_true") + col("y_pred")).alias("y")])
        .collect()
        .unwrap()
}

pub fn confusion_matrix(base_cm: DataFrame) -> ConfusionMatrixArray {
    let s: Vec<f64> = base_cm["y"]
        .value_counts(false, false)
        .unwrap()
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

    let p = tn + fn_;
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
}

pub fn bootstrap_confusion_matrix(
    df: DataFrame,
    iterations: u64,
    z: f64,
    seed: Option<u64>,
) -> Vec<(f64, f64, f64)> {
    let base_cm = base_confusion_matrix(df);

    let runs = bootstrap::run_bootstrap(base_cm, iterations, seed, confusion_matrix);

    let bs_df = DataFrame::new(vec![
        Series::from_vec(
            "stat",
            (0..25)
                .cycle()
                .take((iterations * 25) as usize)
                .collect::<Vec<u64>>(),
        ),
        Series::from_vec("val", runs.concat()),
    ])
    .unwrap()
    .lazy()
    .fill_nan(lit(NULL))
    .group_by_stable(["stat"])
    .agg([
        col("val").mean().alias("mean"),
        col("val").std(0).alias("std"),
    ])
    .drop(["stat"])
    .with_column((lit(z) * (col("std") / lit((iterations as f64).sqrt()))).alias("val"))
    .with_columns([
        (col("mean") - col("val")).alias("lower"),
        (col("mean") + col("val")).alias("upper"),
    ])
    .collect()
    .unwrap();

    bs_df["lower"]
        .f64()
        .unwrap()
        .iter()
        .zip(bs_df["mean"].f64().unwrap().iter())
        .zip(bs_df["upper"].f64().unwrap().iter())
        .map(|((x, y), z)| (x.unwrap(), y.unwrap(), z.unwrap()))
        .collect()
}

fn trapz(y: ArrayView1<f64>, x: ArrayView1<f64>) -> f64 {
    let y_s = &y.slice(s![1..]) + &y.slice(s![..-1]);
    let x_d = &x.slice(s![1..]) - &x.slice(s![..-1]);

    0.5 * (x_d.into_iter().zip(y_s).fold(0., |acc, (x, y)| acc + x * y))
}

pub fn roc_auc(df: DataFrame) -> f64 {
    let positive_counts = df["y_true"].sum::<u32>().unwrap_or(0);
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

fn binary_search_right<T: PartialOrd>(arr: &[T], t: &T) -> Option<usize> {
    // Can likely get rid of the partial_cmp, because I have gauranteed the values to be finite
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

    // Follow SciPy's trick to compute the difference between two CDFs
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

    stats
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
