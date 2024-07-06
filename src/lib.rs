use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

mod bootstrap;
mod distributions;
mod metrics;

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
    z: f64,
    seed: Option<u64>,
) -> PyResult<Vec<(f64, f64, f64)>> {
    let df: DataFrame = df.into();

    Ok(metrics::bootstrap_confusion_matrix(df, iterations, z, seed))
}

#[pyfunction]
fn _roc_auc(df: PyDataFrame) -> PyResult<f64> {
    Ok(metrics::roc_auc(df.into()))
}

#[pyfunction]
fn _bootstrap_roc_auc(
    df: PyDataFrame,
    iterations: u64,
    z: f64,
    seed: Option<u64>,
) -> PyResult<(f64, f64, f64)> {
    Ok(bootstrap::confidence_interval(
        bootstrap::run_bootstrap(df.into(), iterations, seed, metrics::roc_auc),
        z,
    ))
}

#[pyfunction]
fn _max_ks(df: PyDataFrame) -> PyResult<f64> {
    Ok(metrics::max_ks(df.into()))
}

#[pyfunction]
fn _bootstrap_max_ks(
    df: PyDataFrame,
    iterations: u64,
    z: f64,
    seed: Option<u64>,
) -> PyResult<(f64, f64, f64)> {
    Ok(bootstrap::confidence_interval(
        bootstrap::run_bootstrap(df.into(), iterations, seed, metrics::max_ks),
        z,
    ))
}

#[pyfunction]
fn _brier_loss(df: PyDataFrame) -> PyResult<f64> {
    Ok(metrics::brier_loss(df.into()))
}

#[pyfunction]
fn _bootstrap_brier_loss(
    df: PyDataFrame,
    iterations: u64,
    z: f64,
    seed: Option<u64>,
) -> PyResult<(f64, f64, f64)> {
    Ok(bootstrap::confidence_interval(
        bootstrap::run_bootstrap(df.into(), iterations, seed, metrics::brier_loss),
        z,
    ))
}

#[pyfunction]
fn _mean(df: PyDataFrame) -> PyResult<f64> {
    Ok(metrics::mean(df.into()))
}

#[pyfunction]
fn _bootstrap_mean(
    df: PyDataFrame,
    iterations: u64,
    z: f64,
    seed: Option<u64>,
) -> PyResult<(f64, f64, f64)> {
    Ok(bootstrap::confidence_interval(
        bootstrap::run_bootstrap(df.into(), iterations, seed, metrics::mean),
        z,
    ))
}

#[pyfunction]
fn _adverse_impact_ratio(df: PyDataFrame) -> PyResult<f64> {
    Ok(metrics::adverse_impact_ratio(df.into()))
}

#[pyfunction]
fn _bootstrap_adverse_impact_ratio(
    df: PyDataFrame,
    iterations: u64,
    z: f64,
    seed: Option<u64>,
) -> PyResult<(f64, f64, f64)> {
    Ok(bootstrap::confidence_interval(
        bootstrap::run_bootstrap(df.into(), iterations, seed, metrics::adverse_impact_ratio),
        z,
    ))
}

#[pyfunction]
fn _norm_ppf(q: f64) -> PyResult<f64> {
    Ok(distributions::norm_ppf(q))
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
    m.add_function(wrap_pyfunction!(_norm_ppf, m)?)?;

    Ok(())
}
