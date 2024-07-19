use polars::prelude::*;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon::prelude::*;

use crate::distributions;

pub type ConfidenceInterval = (f64, f64, f64);

// The resulting bootstrap vectors are small vectors, usually around 500-10_000 in
// length, so let's just operate on these vectors directly instead of converting into
// ndarray or ChunkedArray
trait VecUtils {
    fn mean(&self) -> f64;
    fn std(&self) -> f64;
    fn drop_nans(&self) -> Vec<f64>;
    fn percentile(&self, q: f64) -> f64;
}

impl VecUtils for Vec<f64> {
    #[allow(clippy::manual_range_contains)]
    fn percentile(&self, q: f64) -> f64 {
        if self.is_empty() {
            return f64::NAN;
        }

        if q < 0.0 || q > 100.0 {
            panic!("Percentile must be between 0 and 100");
        }

        let mut sorted_data = self.clone();
        sorted_data.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

        if q == 0.0 {
            return sorted_data[0];
        }
        if q == 100.0 {
            return sorted_data[sorted_data.len() - 1];
        }

        let rank = (q / 100.0) * (sorted_data.len() - 1) as f64;
        let lower_index = rank.floor() as usize;
        let upper_index = rank.ceil() as usize;

        if lower_index == upper_index {
            sorted_data[lower_index]
        } else {
            let lower_value = sorted_data[lower_index];
            let upper_value = sorted_data[upper_index];
            let fraction = rank - lower_index as f64;

            lower_value + (upper_value - lower_value) * fraction
        }
    }

    fn drop_nans(&self) -> Vec<f64> {
        // copied is a no-op for f64
        self.iter().copied().filter(|x| !x.is_nan()).collect()
    }

    fn mean(&self) -> f64 {
        if self.is_empty() {
            return f64::NAN;
        }

        self.iter().sum::<f64>() / self.len() as f64
    }

    fn std(&self) -> f64 {
        if self.len() < 2 {
            return f64::NAN;
        }
        let mean = self.mean();
        let variance =
            self.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (self.len() - 1) as f64;

        variance.sqrt()
    }
}

// fn _compare(
//     arr1: Vec<&str>,
//     arr2: Vec<&str>,
//     func_name: &str,
//     n_jobs: usize,
//     quiet: bool,
// ) -> PyResult<Vec<f64>> {
//     let func = func_dispatcher(func_name);

//     let arr1 = arr1.as_slice();
//     let arr2 = arr2.as_slice();

//     if n_jobs == 0 {
//         Ok(fuzzycompare(arr1, arr2, func, quiet))
//     } else if n_jobs == 1 {
//         Ok(fuzzycompare_sequential(arr1, arr2, func, quiet))
//     } else {
//         Ok(utils::create_rayon_pool(n_jobs)?.install(|| fuzzycompare(arr1, arr2, func, quiet)))
//     }
// }

fn create_rayon_pool(n_jobs: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n_jobs)
        .build()
        .unwrap()
}

fn bootstrap_core<T: Send + Sync>(
    df: DataFrame,
    iterations: u64,
    seed: Option<u64>,
    func: fn(DataFrame) -> T,
    chunksize: Option<usize>,
) -> Vec<T> {
    let df_height = df.height();

    let seeds: Vec<u64> = (0..iterations).collect();

    let res: Vec<T> = if chunksize.is_none() {
        seeds
            .par_iter()
            .map(|i| {
                func(
                    df.sample_n_literal(df_height, true, false, seed.map(|seed| seed + i))
                        .unwrap(),
                )
            })
            .collect()
    } else {
        let chunksize = chunksize.unwrap();
        seeds
            .par_chunks(chunksize)
            .flat_map(|chunk| {
                chunk
                    .iter()
                    .map(|&i| {
                        func(
                            df.sample_n_literal(df_height, true, false, seed.map(|seed| seed + i))
                                .unwrap(),
                        )
                    })
                    .collect::<Vec<T>>()
            })
            .collect()
    };

    res
}

pub fn run_bootstrap<T: Send + Sync>(
    df: DataFrame,
    iterations: u64,
    seed: Option<u64>,
    func: fn(DataFrame) -> T,
    n_jobs: Option<usize>,
    chunksize: Option<usize>,
) -> Vec<T> {
    let df_height = df.height();

    let bootstrap_stats: Vec<T> = if n_jobs == Some(1) {
        (0..iterations)
            .map(|i| {
                func(
                    df.sample_n_literal(df_height, true, false, seed.map(|seed| seed + i))
                        .unwrap(),
                )
            })
            .collect()
    } else if n_jobs.is_none() {
        bootstrap_core(df, iterations, seed, func, chunksize)
    } else {
        create_rayon_pool(n_jobs.unwrap())
            .install(|| bootstrap_core(df, iterations, seed, func, chunksize))
    };

    bootstrap_stats
}

pub fn run_jacknife<T: Send + Sync>(df: DataFrame, func: fn(DataFrame) -> T) -> Vec<T> {
    let df_height = df.height();
    let index = ChunkedArray::new("index", 0..df_height as u64);
    let jacknife_stats: Vec<T> = (0..df_height)
        .into_par_iter()
        .map(|i| func(df.filter(&index.not_equal(i)).unwrap()))
        .collect();

    jacknife_stats
}

pub fn standard_interval(bootstrap_stats: Vec<f64>, alpha: f64) -> ConfidenceInterval {
    let runs = bootstrap_stats.drop_nans();
    let mean = runs.mean();
    let stderr = runs.std();
    let z = distributions::norm_ppf(1.0 - alpha);
    let x = z * stderr;

    (mean - x, mean, mean + x)
}

pub fn percentile_interval(bootstrap_stats: Vec<f64>, alpha: f64) -> ConfidenceInterval {
    let runs = bootstrap_stats.drop_nans();
    let mean = runs.mean();

    (
        runs.percentile(alpha * 100.0),
        mean,
        runs.percentile((1.0 - alpha) * 100.0),
    )
}

pub fn basic_interval(
    original_stat: f64,
    bootstrap_stats: Vec<f64>,
    alpha: f64,
) -> ConfidenceInterval {
    let interval = percentile_interval(bootstrap_stats, alpha);
    let lower = interval.0;
    let mean = interval.1;
    let upper = interval.2;

    let x = 2.0 * original_stat;

    (x - upper, mean, x - lower)
}

fn percentile_of_score(arr: &[f64], score: f64) -> f64 {
    let a1 = arr.iter().filter(|x| x < &&score).count() as f64;
    let a2 = arr.iter().filter(|x| x <= &&score).count() as f64;

    (a1 + a2) / (2.0 * arr.len() as f64)
}

pub fn bca_interval(
    original_stat: f64,
    bootstrap_stats: Vec<f64>,
    jacknife_stats: Vec<f64>,
    alpha: f64,
) -> ConfidenceInterval {
    let bootstrap_stats = bootstrap_stats.drop_nans();
    let jacknife_stats = jacknife_stats.drop_nans();
    let z1 = distributions::norm_ppf(alpha);
    let z2 = -z1;

    let bias_correction_factor =
        distributions::norm_ppf(percentile_of_score(&bootstrap_stats, original_stat));

    let jacknife_mean = jacknife_stats.mean();
    let n = jacknife_stats.len() as f64;
    let n1 = n - 1.0;
    let diff: Vec<f64> = jacknife_stats
        .iter()
        .map(|x| n1 * (jacknife_mean - x))
        .collect();
    let numerator = diff.iter().map(|x| x.powi(3)).sum::<f64>() / n.powi(3);
    let denominator = diff.iter().map(|x| x.powi(2)).sum::<f64>() / n.powi(2);
    let acceleration_factor = numerator / (6.0 * denominator.powf(1.5));

    let lower_p = distributions::norm_cdf(
        bias_correction_factor
            + (bias_correction_factor + z1)
                / (1.0 - acceleration_factor * (bias_correction_factor + z1)),
    );
    let upper_p = distributions::norm_cdf(
        bias_correction_factor
            + (bias_correction_factor + z2)
                / (1.0 - acceleration_factor * (bias_correction_factor + z2)),
    );

    (
        bootstrap_stats.percentile(lower_p * 100.0),
        bootstrap_stats.mean(),
        bootstrap_stats.percentile(upper_p * 100.0),
    )
}
