use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

pub fn trapezoidal_auc(x: &[f64], y: &[f64]) -> f64 {
    x.windows(2)
        .zip(y.windows(2))
        .fold(0.0, |sum, (x_window, y_window)| {
            let dx = x_window[1] - x_window[0];
            let trapezoid_area = (y_window[0] + y_window[1]) * dx / 2.0;
            sum + trapezoid_area
        })
}

pub fn rectangular_auc(x: &[f64], y: &[f64]) -> f64 {
    x.windows(2)
        .zip(y.windows(2))
        .fold(0.0, |sum, (x_window, y_window)| {
            let dx = x_window[1] - x_window[0];
            let y_midpoint = (y_window[0] + y_window[1]) / 2.0;
            sum + dx * y_midpoint
        })
}

#[polars_expr(output_type=Float64)]
fn pl_auc(inputs: &[Series]) -> PolarsResult<Series> {
    let x = &inputs[0];
    let y = &inputs[1];

    let is_trapezoidal = inputs[2].bool()?.get(0).unwrap();

    let df = df!(
        "x" => x,
        "y" => y,
    )?
    .sort(["x"], Default::default())?;

    let x = df["x"].f64()?.cont_slice()?;
    let y = df["y"].f64()?.cont_slice()?;

    let res = if is_trapezoidal {
        trapezoidal_auc(x, y)
    } else {
        rectangular_auc(x, y)
    };

    Ok(Series::from_vec("auc".into(), vec![res]))
}
