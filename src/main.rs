use std::ops::Mul;

use ndarray::{array, s, Array1, ArrayView1};
use polars::prelude::*;
use rayon::vec;

fn trapz(y: ArrayView1<f64>, x: ArrayView1<f64>) -> f64 {
    let y_s = &y.slice(s![1..]) + &y.slice(s![..-1]);
    let x_d = &x.slice(s![1..]) - &x.slice(s![..-1]);

    0.5 * (x_d.into_iter().zip(y_s).fold(0., |acc, (x, y)| acc + x * y))
}

fn integrate_trapezoid(y: &[f64], x: &[f64]) -> f64 {
    // assuming `x.len() == y.len()`
    0.5 * (0..x.len() - 1)
        .map(|i| (x[i + 1] - x[i]) * (y[i + 1] + y[i]))
        .sum::<f64>()
}

// fn windows<T, const N: usize>(slice: &[T]) -> impl Iterator<Item = &[T; N]> {
//     slice.windows(N).map(|x| x.try_into().unwrap())
// }

// pub fn trap_v3(x: &[f64], y: &[f64]) -> f64 {
//     assert_eq!(x.len(), y.len());
//     windows(x)
//         .zip(windows(y))
//         .map(|([x0, x1], [y0, y1])| (x1 - x0) * (y1 + y0))
//         .sum::<f64>()
// }

fn main() {
    let arr1 = &[1.0, 2.0, 3.0, f64::NAN];
    let arr2 = &[4.0, 5.0, 6.0];

    let arr = array![-2., -1., 0., 1., 2., f64::NAN];

    // Clip lower, removing NANs. This can alternatively be written as
    // `v.max(-1.)`. For types implementing `Ord`, you could use `cmp::max` or
    // the `.max()` method.
    let clip_lower_remove = arr.mapv(|v| f64::max(v, -1.));
    dbg!(clip_lower_remove);

    // dbg!(ArrayView1::from(arr1)
    //     .iter()
    //     .filter(|x| x.is_nan())
    //     .collect::<Array1, <f64>>()
    //     .mean()
    //     .unwrap());
    dbg!(ChunkedArray::new("x", arr1).mean());

    // pub fn base_confusion_matrix(df: DataFrame) -> DataFrame {
    //     df.lazy()
    //         .select([(lit(2) * col("y_true") + col("y_pred")).alias("y")])
    //         .collect()
    //         .unwrap()
    // }

    // let df =
    //     df!("y_true" => vec![true, false, true], "y_pred" => vec![true, false, false]).unwrap();

    // dbg!(base_confusion_matrix(df)
    //     .column("y")
    //     .unwrap()
    //     .value_counts(false, false)
    //     .unwrap());

    // let x: ChunkedArray<Float64Type> = ChunkedArray::from_vec("x", arr1.to_vec());

    // dbg!(x
    //     .into_series()
    //     .value_counts(false, false)
    //     .unwrap()
    //     .sort(["x"], Default::default())
    //     .unwrap()
    //     .column("count")
    //     .unwrap()
    //     .u32()
    //     .unwrap()
    //     .to_vec());

    // let x = ArrayView1::from(arr1);
    // dbg!(x.slice(s![[1, 2]]));

    // let actual = [true, false, true];
    // let predicted = [0.7, 0.4, 0.9];

    // let n = predicted.len() as u32;
    // let df = df!(
    //     "threshold" => arr1,
    //     "actual" => arr2
    // )
    // .unwrap()
    // .lazy()
    // .select([col("*").sum()])
    // .collect()
    // .unwrap();

    // df["threshold"].f64().unwrap().get(0).unwrap();

    // dbg!(df);

    // let positive_counts = df["actual"].sum::<u32>().unwrap_or(0);

    // let temp = df
    //     .lazy()
    //     .group_by([col("threshold")])
    //     .agg([
    //         len().alias("cnt"),
    //         col("actual").sum().alias("pos_cnt_at_threshold"),
    //     ])
    //     .sort(["threshold"], Default::default())
    //     .with_columns([
    //         (lit(n) - col("cnt").cum_sum(false) + col("cnt")).alias("predicted_positive"),
    //         (lit(positive_counts) - col("pos_cnt_at_threshold").cum_sum(false))
    //             .shift_and_fill(1, positive_counts)
    //             .alias("tp"),
    //     ])
    //     .select([
    //         col("threshold"),
    //         col("tp"),
    //         (col("predicted_positive") - col("tp")).alias("fp"),
    //         (col("tp").cast(DataType::Float64) / col("predicted_positive").cast(DataType::Float64))
    //             .alias("precision"),
    //     ])
    //     .with_columns([col("tp").first().cast(DataType::Float64).alias("tpr")]);

    // dbg!(temp.collect().unwrap());
}
