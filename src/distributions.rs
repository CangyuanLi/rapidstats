use std::f64::consts::SQRT_2;

pub fn norm_ppf(q: f64) -> f64 {
    if q == 0.0 {
        return f64::NEG_INFINITY;
    }

    if q == 1.0 {
        return f64::INFINITY;
    }

    // Coefficients for the rational approximation
    const A1: f64 = -39.6968302866538;
    const A2: f64 = 220.946098424521;
    const A3: f64 = -275.928510446969;
    const A4: f64 = 138.357751867269;
    const A5: f64 = -30.6647980661472;
    const A6: f64 = 2.50662827745924;

    const B1: f64 = -54.4760987982241;
    const B2: f64 = 161.585836858041;
    const B3: f64 = -155.698979859887;
    const B4: f64 = 66.8013118877197;
    const B5: f64 = -13.2806815528857;

    const C1: f64 = -7.78489400243029E-03;
    const C2: f64 = -0.322396458041136;
    const C3: f64 = -2.40075827716184;
    const C4: f64 = -2.54973253934373;
    const C5: f64 = 4.37466414146497;
    const C6: f64 = 2.93816398269878;

    const D1: f64 = 7.78469570904146E-03;
    const D2: f64 = 0.32246712907004;
    const D3: f64 = 2.445134137143;
    const D4: f64 = 3.75440866190742;

    // Rational approximation for lower region
    if q < 0.02425 {
        let q = (-2.0 * q.ln()).sqrt();
        return (((((C1 * q + C2) * q + C3) * q + C4) * q + C5) * q + C6)
            / ((((D1 * q + D2) * q + D3) * q + D4) * q + 1.0);
    }

    // Rational approximation for central region
    if q < 0.97575 {
        let q = q - 0.5;
        let r = q * q;
        return (((((A1 * r + A2) * r + A3) * r + A4) * r + A5) * r + A6) * q
            / (((((B1 * r + B2) * r + B3) * r + B4) * r + B5) * r + 1.0);
    }

    // Rational approximation for upper region
    let q = (-2.0 * (1.0 - q).ln()).sqrt();

    -(((((C1 * q + C2) * q + C3) * q + C4) * q + C5) * q + C6)
        / ((((D1 * q + D2) * q + D3) * q + D4) * q + 1.0)
}

fn erf(x: f64) -> f64 {
    // Constants
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    // Save the sign of x
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    // A&S formula 7.1.26
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

pub fn norm_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / SQRT_2))
}
