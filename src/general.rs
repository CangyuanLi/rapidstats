use std::f64;

use polars::prelude::*;
use pyo3_polars::{
    derive::polars_expr,
    export::{
        polars_arrow::{
            array::BooleanArray, bitmap::MutableBitmap, datatypes::ArrowDataType::Boolean,
        },
        polars_core::utils::Container,
    },
};

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

    let mut df = df!(
        "x" => x,
        "y" => y,
    )?
    .sort(["x"], Default::default())?;

    df.rechunk_mut();

    let x = df["x"].f64()?.cont_slice()?;
    let y = df["y"].f64()?.cont_slice()?;

    let res = if is_trapezoidal {
        trapezoidal_auc(x, y)
    } else {
        rectangular_auc(x, y)
    };

    Ok(Series::from_vec("auc".into(), vec![res]))
}

#[polars_expr(output_type=Boolean)]
fn pl_pareto_2d(inputs: &[Series]) -> PolarsResult<Series> {
    let x = &inputs[0];
    let y = &inputs[1];

    let df = df!("x" => x, "y" => y)?
        .with_row_index("index".into(), Some(0))?
        .sort(
            ["x", "y"],
            SortMultipleOptions::default().with_order_descending_multi([true, true]),
        )?;

    let index = df["index"].u32()?;
    let x_sorted = df["x"].f64()?;
    let y_sorted = df["y"].f64()?;

    let mut res: Vec<bool> = vec![false; x.len()];
    let mut validity = MutableBitmap::with_capacity(x.len());
    validity.extend_constant(x.len(), true);

    let mut best_y = -f64::INFINITY;
    for ((i, x), y) in index.into_no_null_iter().zip(x_sorted).zip(y_sorted) {
        let i_u = i as usize;

        let (x, y) = match (x, y) {
            (Some(x), Some(y)) => (x, y),
            _ => {
                validity.set(i_u, false);
                continue;
            }
        };

        if x.is_nan() || y.is_nan() {
            validity.set(i_u, false);
            continue;
        }

        if y > best_y {
            res[i_u] = true;
            best_y = y;
        } else {
            res[i_u] = false;
        }
    }

    let arr = BooleanArray::new(Boolean, res.into(), validity.into());

    Ok(BooleanChunked::with_chunk(PlSmallStr::EMPTY, arr).into_series())
}
