//! Natural cubic splines over tabulated data (suspension kinematics, engine and tyre maps).

use serde::{Deserialize, Serialize};

/// Error for tables that cannot be interpolated.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("invalid table: {0}")]
pub struct TableError(pub String);

/// Natural cubic spline through `(x_k, y_k)`, continued linearly beyond the end knots (so the
/// value and first derivative are continuous everywhere and the second derivative is zero
/// outside the table).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CubicSpline {
    x: Vec<f64>,
    y: Vec<f64>,
    /// Second derivatives at the knots.
    m: Vec<f64>,
}

impl CubicSpline {
    /// A spline through at least two points with strictly increasing, finite `x`.
    pub fn new(x: Vec<f64>, y: Vec<f64>) -> Result<Self, TableError> {
        let n = x.len();
        if n < 2 || y.len() != n {
            return Err(TableError(format!("{n} abscissae and {} values; at least 2 of each needed", y.len())));
        }
        if x.iter().chain(&y).any(|v| !v.is_finite()) || x.windows(2).any(|w| w[1] <= w[0]) {
            return Err(TableError("abscissae must be finite and strictly increasing, values finite".into()));
        }
        // Tridiagonal system for the interior second derivatives (m₀ = m_{n−1} = 0), Thomas
        // algorithm.
        let mut m = vec![0.0; n];
        if n > 2 {
            let mut diag = vec![0.0; n];
            let mut rhs = vec![0.0; n];
            for i in 1..n - 1 {
                let (h0, h1) = (x[i] - x[i - 1], x[i + 1] - x[i]);
                diag[i] = 2.0 * (h0 + h1);
                rhs[i] = 6.0 * ((y[i + 1] - y[i]) / h1 - (y[i] - y[i - 1]) / h0);
            }
            for i in 2..n - 1 {
                let w = (x[i] - x[i - 1]) / diag[i - 1];
                diag[i] -= w * (x[i] - x[i - 1]);
                rhs[i] -= w * rhs[i - 1];
            }
            for i in (1..n - 1).rev() {
                let upper = if i + 1 < n - 1 { (x[i + 1] - x[i]) * m[i + 1] } else { 0.0 };
                m[i] = (rhs[i] - upper) / diag[i];
            }
        }
        Ok(Self { x, y, m })
    }

    /// A constant.
    pub fn constant(y: f64) -> Self {
        Self { x: vec![0.0, 1.0], y: vec![y, y], m: vec![0.0, 0.0] }
    }

    pub fn knots(&self) -> &[f64] {
        &self.x
    }

    pub fn values(&self) -> &[f64] {
        &self.y
    }

    /// Value, first and second derivative at `t`.
    #[inline]
    pub fn eval(&self, t: f64) -> (f64, f64, f64) {
        let (x, y, m) = (&self.x, &self.y, &self.m);
        let n = x.len();
        if t <= x[0] {
            let d = self.slope(0, 0.0);
            return (y[0] + d * (t - x[0]), d, 0.0);
        }
        if t >= x[n - 1] {
            let d = self.slope(n - 2, 1.0);
            return (y[n - 1] + d * (t - x[n - 1]), d, 0.0);
        }
        let i = x.partition_point(|&k| k <= t).clamp(1, n - 1) - 1;
        let h = x[i + 1] - x[i];
        let (a, b) = ((x[i + 1] - t) / h, (t - x[i]) / h);
        let value = a * y[i] + b * y[i + 1] + ((a * a * a - a) * m[i] + (b * b * b - b) * m[i + 1]) * h * h / 6.0;
        let d1 = (y[i + 1] - y[i]) / h + ((3.0 * b * b - 1.0) * m[i + 1] - (3.0 * a * a - 1.0) * m[i]) * h / 6.0;
        let d2 = a * m[i] + b * m[i + 1];
        (value, d1, d2)
    }

    /// First derivative within interval `i` at the fraction `b` (0: left knot, 1: right knot).
    fn slope(&self, i: usize, b: f64) -> f64 {
        let (x, y, m) = (&self.x, &self.y, &self.m);
        let h = x[i + 1] - x[i];
        let a = 1.0 - b;
        (y[i + 1] - y[i]) / h + ((3.0 * b * b - 1.0) * m[i + 1] - (3.0 * a * a - 1.0) * m[i]) * h / 6.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reproduces_cubics_with_zero_end_curvature_and_lines_exactly() {
        let line = CubicSpline::new(vec![-1.0, 0.5, 2.0, 3.0], vec![-2.0, 1.0, 4.0, 6.0]).unwrap();
        for t in [-3.0, -1.0, 0.0, 1.7, 3.0, 5.0] {
            let (v, d, dd) = line.eval(t);
            assert!((v - 2.0 * t).abs() < 1e-12 && (d - 2.0).abs() < 1e-12 && dd.abs() < 1e-12, "{t}");
        }
        // sin on a fine grid: close to the function and its derivatives inside.
        let x: Vec<f64> = (0..=60).map(|k| k as f64 * 0.05).collect();
        let y: Vec<f64> = x.iter().map(|t| t.sin()).collect();
        let s = CubicSpline::new(x, y).unwrap();
        for t in [0.4, 1.0, 1.73, 2.5] {
            let (v, d, dd) = s.eval(t);
            assert!((v - t.sin()).abs() < 1e-5 && (d - t.cos()).abs() < 1e-3 && (dd + t.sin()).abs() < 2e-2, "{t}");
        }
    }

    #[test]
    fn derivatives_are_consistent_and_continuous() {
        let s = CubicSpline::new(vec![0.0, 1.0, 1.5, 4.0], vec![1.0, -2.0, 0.5, 3.0]).unwrap();
        let h = 1e-6;
        for t in [-0.5, 0.0, 0.3, 1.0, 1.2, 1.5, 3.9, 4.0, 6.0] {
            let (_, d, dd) = s.eval(t);
            let fd = (s.eval(t + h).0 - s.eval(t - h).0) / (2.0 * h);
            let fdd = (s.eval(t + h).1 - s.eval(t - h).1) / (2.0 * h);
            assert!((fd - d).abs() < 1e-6, "{t}: {fd} vs {d}");
            // The second derivative is continuous inside; at the end knots it goes to 0.
            assert!((fdd - dd).abs() < 1e-4 || t == 0.0 || t == 4.0, "{t}: {fdd} vs {dd}");
        }
        assert!(CubicSpline::new(vec![0.0, 0.0], vec![1.0, 2.0]).is_err());
        assert!(CubicSpline::new(vec![0.0], vec![1.0]).is_err());
    }
}
