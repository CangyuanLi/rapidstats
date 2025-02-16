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
