use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

fn format_with_separator(s: &str, sep: char) -> String {
    let parts: Vec<_> = s.split('.').collect();

    // Format the integer part
    let int_part = parts[0];
    let mut result = String::new();
    let len = int_part.len();

    for (i, c) in int_part.chars().enumerate() {
        result.push(c);
        if (len - i - 1) % 3 == 0 && i < len - 1 {
            result.push(sep);
        }
    }

    // Add the decimal part back if it exists
    if parts.len() > 1 {
        result.push('.');
        result.push_str(parts[1]);
    }

    result
}

#[polars_expr(output_type=String)]
fn _pl_format_strnum_with_sep(inputs: &[Series]) -> PolarsResult<Series> {
    let sep = inputs[1].str()?.get(0).unwrap().chars().next().unwrap();

    Ok(inputs[0]
        .str()?
        .apply_into_string_amortized(|s, buf| *buf = format_with_separator(s, sep))
        .into_series())
}
