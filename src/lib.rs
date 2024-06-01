use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

mod bootstrap;
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

    Ok(())
}
