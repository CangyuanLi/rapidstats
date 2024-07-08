use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

mod bootstrap;
mod distributions;
mod metrics;

macro_rules! generate_bootstrap_function {
    ($func_name:ident, $metric_func:path) => {
        #[pyfunction]
        fn $func_name(
            df: PyDataFrame,
            iterations: u64,
            z: (f64, f64),
            seed: Option<u64>,
        ) -> PyResult<bootstrap::ConfidenceInterval> {
            let df: DataFrame = df.into();
            let bootstrap_stats =
                bootstrap::run_bootstrap(df.clone(), iterations, seed, $metric_func);
            if z.1.is_nan() {
                Ok(bootstrap::confidence_interval(bootstrap_stats, z.0))
            } else {
                let original_stat = $metric_func(df.clone());
                let jacknife_stats = bootstrap::run_jacknife(df, $metric_func);
                Ok(bootstrap::bca_confidence_interval(
                    original_stat,
                    bootstrap_stats,
                    jacknife_stats,
                    z,
                ))
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
    z: (f64, f64),
    seed: Option<u64>,
) -> PyResult<Vec<bootstrap::ConfidenceInterval>> {
    let df: DataFrame = df.into();

    Ok(metrics::bootstrap_confusion_matrix(df, iterations, z, seed))
}

#[pyfunction]
fn _roc_auc(df: PyDataFrame) -> PyResult<f64> {
    Ok(metrics::roc_auc(df.into()))
}

generate_bootstrap_function!(_bootstrap_roc_auc, metrics::roc_auc);

#[pyfunction]
fn _max_ks(df: PyDataFrame) -> PyResult<f64> {
    Ok(metrics::max_ks(df.into()))
}

generate_bootstrap_function!(_bootstrap_max_ks, metrics::max_ks);

#[pyfunction]
fn _brier_loss(df: PyDataFrame) -> PyResult<f64> {
    Ok(metrics::brier_loss(df.into()))
}

generate_bootstrap_function!(_bootstrap_brier_loss, metrics::brier_loss);

#[pyfunction]
fn _mean(df: PyDataFrame) -> PyResult<f64> {
    Ok(metrics::mean(df.into()))
}

generate_bootstrap_function!(_bootstrap_mean, metrics::mean);

#[pyfunction]
fn _adverse_impact_ratio(df: PyDataFrame) -> PyResult<f64> {
    Ok(metrics::adverse_impact_ratio(df.into()))
}

generate_bootstrap_function!(
    _bootstrap_adverse_impact_ratio,
    metrics::adverse_impact_ratio
);

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
