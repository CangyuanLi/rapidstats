use polars::prelude::*;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

pub fn run_bootstrap<T: Send + Sync>(
    df: DataFrame,
    iterations: u64,
    seed: Option<u64>,
    func: fn(DataFrame) -> T,
) -> Vec<T> {
    let df_height = df.height();

    let runs: Vec<T> = (0..iterations)
        .into_par_iter()
        .map(|i| {
            func(
                df.sample_n_literal(df_height, true, false, seed.map(|seed| seed + i))
                    .unwrap(),
            )
        })
        .collect();

    runs
}

pub fn confidence_interval(runs: Vec<f64>, z: f64) -> (f64, f64, f64) {
    let iterations = runs.len() as f64;
    let s = ChunkedArray::new("x", runs);
    let s = s.filter(&s.is_not_nan()).unwrap();
    let std = s.std(0).unwrap_or(f64::NAN);
    let mean = s.mean().unwrap_or(f64::NAN);
    let x = z * std / iterations.sqrt();

    (mean - x, mean, mean + x)
}
