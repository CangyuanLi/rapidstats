use bootstrap::ConfidenceInterval;
use paste::paste;
use polars::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

mod bootstrap;
mod distributions;
mod metrics;
mod utils;

macro_rules! generate_functions {
    ($func_name:ident, $metric_func:path) => {
        #[pyfunction]
        fn $func_name(df: PyDataFrame) -> PyResult<f64> {
            Ok($metric_func(df.into()))
        }

        paste! {
            #[pyfunction]
            fn [<_bootstrap $func_name>] (
                df: PyDataFrame,
                iterations: u64,
                alpha: f64,
                method: &str,
                seed: Option<u64>,
                n_jobs: Option<usize>,
                chunksize: Option<usize>,
            ) -> PyResult<bootstrap::ConfidenceInterval> {
                let df: DataFrame = df.into();
                let bootstrap_stats =
                    bootstrap::run_bootstrap(df.clone(), iterations, seed, $metric_func, n_jobs, chunksize);
                if method == "standard" {
                    Ok(bootstrap::standard_interval(bootstrap_stats, alpha))
                }
                else if method == "percentile" {
                    Ok(bootstrap::percentile_interval(bootstrap_stats, alpha))
                } else if method == "basic" {
                    let original_stat = $metric_func(df.clone());
                    Ok(bootstrap::basic_interval(original_stat, bootstrap_stats, alpha))
                } else if method == "BCa" {
                    let original_stat = $metric_func(df.clone());
                    let jacknife_stats = bootstrap::run_jacknife(df, $metric_func);
                    Ok(bootstrap::bca_interval(
                        original_stat,
                        bootstrap_stats,
                        jacknife_stats,
                        alpha,
                    ))
                } else {
                    Err(PyValueError::new_err(format!(
                        "Invalid confidence interval method `{}`, only `percentile`, `basic`, and `BCa` are supported",
                        method
                    )))
                }
            }
        }
    };
}

#[pyfunction]
fn _confusion_matrix(df: PyDataFrame) -> PyResult<metrics::ConfusionMatrixArray> {
    let df: DataFrame = df.into();
    let base_cm = metrics::base_confusion_matrix(df);

    Ok(metrics::confusion_matrix(base_cm))
}

#[pyfunction]
fn _bootstrap_confusion_matrix(
    df: PyDataFrame,
    iterations: u64,
    alpha: f64,
    method: &str,
    seed: Option<u64>,
    n_jobs: Option<usize>,
    chunksize: Option<usize>,
) -> PyResult<Vec<bootstrap::ConfidenceInterval>> {
    let df: DataFrame = df.into();

    Ok(metrics::bootstrap_confusion_matrix(
        df, iterations, alpha, method, seed, n_jobs, chunksize,
    ))
}

generate_functions!(_roc_auc, metrics::roc_auc);
generate_functions!(_max_ks, metrics::max_ks);
generate_functions!(_brier_loss, metrics::brier_loss);
generate_functions!(_mean, metrics::mean);
generate_functions!(_adverse_impact_ratio, metrics::adverse_impact_ratio);
generate_functions!(_mean_squared_error, metrics::mean_squared_error);
generate_functions!(_root_mean_squared_error, metrics::root_mean_squared_error);

#[pyfunction]
fn _standard_interval(bootstrap_stats: Vec<f64>, alpha: f64) -> PyResult<ConfidenceInterval> {
    Ok(bootstrap::standard_interval(bootstrap_stats, alpha))
}

#[pyfunction]
fn _percentile_interval(bootstrap_stats: Vec<f64>, alpha: f64) -> PyResult<ConfidenceInterval> {
    Ok(bootstrap::percentile_interval(bootstrap_stats, alpha))
}

#[pyfunction]
fn _basic_interval(
    original_stat: f64,
    bootstrap_stats: Vec<f64>,
    alpha: f64,
) -> PyResult<ConfidenceInterval> {
    Ok(bootstrap::basic_interval(
        original_stat,
        bootstrap_stats,
        alpha,
    ))
}

#[pyfunction]
fn _bca_interval(
    original_stat: f64,
    bootstrap_stats: Vec<f64>,
    jacknife_stats: Vec<f64>,
    alpha: f64,
) -> PyResult<ConfidenceInterval> {
    Ok(bootstrap::bca_interval(
        original_stat,
        bootstrap_stats,
        jacknife_stats,
        alpha,
    ))
}

#[pyfunction]
fn _norm_ppf(q: f64) -> PyResult<f64> {
    Ok(distributions::norm_ppf(q))
}

#[pyfunction]
fn _norm_cdf(x: f64) -> PyResult<f64> {
    Ok(distributions::norm_cdf(x))
}

/// A Python module implemented in Rust.
#[pymodule]
fn _rustystats(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_confusion_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(_bootstrap_confusion_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(_roc_auc, m)?)?;
    m.add_function(wrap_pyfunction!(_bootstrap_roc_auc, m)?)?;
    m.add_function(wrap_pyfunction!(_max_ks, m)?)?;
    m.add_function(wrap_pyfunction!(_bootstrap_max_ks, m)?)?;
    m.add_function(wrap_pyfunction!(_brier_loss, m)?)?;
    m.add_function(wrap_pyfunction!(_bootstrap_brier_loss, m)?)?;
    m.add_function(wrap_pyfunction!(_mean, m)?)?;
    m.add_function(wrap_pyfunction!(_bootstrap_mean, m)?)?;
    m.add_function(wrap_pyfunction!(_adverse_impact_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(_bootstrap_adverse_impact_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(_mean_squared_error, m)?)?;
    m.add_function(wrap_pyfunction!(_bootstrap_mean_squared_error, m)?)?;
    m.add_function(wrap_pyfunction!(_root_mean_squared_error, m)?)?;
    m.add_function(wrap_pyfunction!(_bootstrap_root_mean_squared_error, m)?)?;
    m.add_function(wrap_pyfunction!(_standard_interval, m)?)?;
    m.add_function(wrap_pyfunction!(_percentile_interval, m)?)?;
    m.add_function(wrap_pyfunction!(_basic_interval, m)?)?;
    m.add_function(wrap_pyfunction!(_bca_interval, m)?)?;
    m.add_function(wrap_pyfunction!(_norm_ppf, m)?)?;
    m.add_function(wrap_pyfunction!(_norm_cdf, m)?)?;

    Ok(())
}
