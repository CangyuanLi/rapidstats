use polars::prelude::*;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

pub type ConfusionMatrixArray = [f64; 25];

pub fn confusion_matrix(df: DataFrame) -> ConfusionMatrixArray {
    let y_true = df["y_true"].bool().unwrap();
    let y_pred = df["y_pred"].bool().unwrap();

    let tn = (!y_true & !y_pred).sum().unwrap() as f64;
    let fp = (&!y_true & y_pred).sum().unwrap() as f64;
    let fn_ = (y_true & &!y_pred).sum().unwrap() as f64;
    let tp = (y_true & y_pred).sum().unwrap() as f64;
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
    let df_height = df.height();
    // (1..iterations)
    //     .into_par_iter()
    //     .for_each(|i| df.sample_n_literal(df_height, true, false, Some(i)))

    // let mut runs: Vec<metrics::ConfusionMatrixArray> = Vec::with_capacity(iterations as usize);
    // for i in 0..iterations {
    //     let sampled = df.sample_n_literal(df_height, true, false, Some(i));
    //     let cm = metrics::confusion_matrix(sampled.unwrap());
    //     runs.push(cm);
    // }

    let runs: Vec<ConfusionMatrixArray> = (0..iterations)
        .into_par_iter()
        .map(|i| {
            confusion_matrix(
                df.sample_n_literal(df_height, true, false, seed.map(|seed| seed + i))
                    .unwrap(),
            )
        })
        .collect();

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
    .group_by_stable(["stat"])
    .agg([
        col("val").mean().alias("mean"),
        col("val").std(1).alias("std"),
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

fn roc_auc() {}
