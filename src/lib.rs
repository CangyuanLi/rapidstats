use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::{PyDataFrame, PySeries};

mod bootstrap;
mod metrics;

#[pyfunction]
fn _confusion_matrix(df: PyDataFrame) -> PyResult<metrics::ConfusionMatrixArray> {
    let df: DataFrame = df.into();

    Ok(metrics::confusion_matrix(df))
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

/// A Python module implemented in Rust.
#[pymodule]
fn _rustystats(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_confusion_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(_bootstrap_confusion_matrix, m)?)?;

    Ok(())
}
