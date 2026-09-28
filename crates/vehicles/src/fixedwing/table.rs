//! Lookup tables and curves: linear interpolation held at the ends (as JSBSim's `<table>`), and
//! polynomials.

use serde::{Deserialize, Serialize};

/// Index `i` (1 ≤ i < n) and weight of `x` between `xs[i−1]` and `xs[i]`, clamped to the ends.
#[inline]
fn bracket(xs: &[f64], x: f64) -> (usize, f64) {
    let i = xs.partition_point(|&a| a <= x).clamp(1, xs.len() - 1);
    let (x0, x1) = (xs[i - 1], xs[i]);
    (i, ((x - x0) / (x1 - x0)).clamp(0.0, 1.0))
}

/// Linear interpolation in `(xs, ys)`, held at the ends; `xs` increasing, at least two points.
#[inline]
pub fn interp(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    if xs.len() == 1 {
        return ys[0];
    }
    let (i, s) = bracket(xs, x);
    ys[i - 1] + s * (ys[i] - ys[i - 1])
}

fn increasing(xs: &[f64]) -> bool {
    !xs.is_empty() && xs.iter().all(|x| x.is_finite()) && xs.windows(2).all(|w| w[1] > w[0])
}

/// A function of one variable: a polynomial `c₀ + c₁x + c₂x² + …`, or a table.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Curve {
    Poly { poly: Vec<f64> },
    Table { x: Vec<f64>, y: Vec<f64> },
}

impl Curve {
    pub fn eval(&self, x: f64) -> f64 {
        match self {
            Curve::Poly { poly } => poly.iter().rev().fold(0.0, |acc, c| acc * x + c),
            Curve::Table { x: xs, y } => interp(xs, y, x),
        }
    }

    pub fn scaled(&self, k: f64) -> Self {
        match self {
            Curve::Poly { poly } => Curve::Poly { poly: poly.iter().map(|c| c * k).collect() },
            Curve::Table { x, y } => Curve::Table { x: x.clone(), y: y.iter().map(|v| v * k).collect() },
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        match self {
            Curve::Poly { poly } if poly.is_empty() || poly.iter().any(|c| !c.is_finite()) => {
                Err("polynomial needs finite coefficients".into())
            }
            Curve::Table { x, y } if !increasing(x) || y.len() != x.len() || y.iter().any(|v| !v.is_finite()) => {
                Err("curve table needs increasing x and as many finite y".into())
            }
            _ => Ok(()),
        }
    }
}

/// Values of a table's lookup variables.
pub trait Lookup<V> {
    fn value(&self, var: V) -> f64;
}

/// A table of one or two variables (`row`, and `col` if given): `data` holds `rows.len()`
/// values, or `rows.len() × cols.len()` values row by row.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, bound(serialize = "V: Serialize", deserialize = "V: Deserialize<'de>"))]
pub struct Table<V> {
    pub row: V,
    pub rows: Vec<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub col: Option<V>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cols: Vec<f64>,
    pub data: Vec<f64>,
}

impl<V: Copy> Table<V> {
    pub fn eval(&self, vars: &impl Lookup<V>) -> f64 {
        let x = vars.value(self.row);
        let Some(col) = self.col else { return interp(&self.rows, &self.data, x) };
        let nc = self.cols.len();
        let at = |r: usize, c: usize| self.data[r * nc + c];
        let (c, t) = if nc == 1 { (1, 0.0) } else { bracket(&self.cols, vars.value(col)) };
        let row = |r: usize| if nc == 1 { at(r, 0) } else { at(r, c - 1) + t * (at(r, c) - at(r, c - 1)) };
        if self.rows.len() == 1 {
            return row(0);
        }
        let (r, s) = bracket(&self.rows, x);
        row(r - 1) + s * (row(r) - row(r - 1))
    }

    pub fn validate(&self) -> Result<(), String> {
        if !increasing(&self.rows) {
            return Err("table rows must be finite and increasing".into());
        }
        let n = match self.col {
            Some(_) if !increasing(&self.cols) => return Err("table columns must be finite and increasing".into()),
            Some(_) => self.rows.len() * self.cols.len(),
            None if !self.cols.is_empty() => return Err("table columns without a column variable".into()),
            None => self.rows.len(),
        };
        if self.data.len() != n || self.data.iter().any(|v| !v.is_finite()) {
            return Err(format!("table needs {n} finite values, has {}", self.data.len()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy)]
    enum V {
        A,
        B,
    }
    struct Vals(f64, f64);
    impl Lookup<V> for Vals {
        fn value(&self, v: V) -> f64 {
            match v {
                V::A => self.0,
                V::B => self.1,
            }
        }
    }

    #[test]
    fn curves_and_tables() {
        let p = Curve::Poly { poly: vec![1.0, 2.0, 3.0] };
        assert_eq!(p.eval(2.0), 1.0 + 4.0 + 12.0);
        let t = Curve::Table { x: vec![0.0, 1.0], y: vec![0.0, 10.0] };
        assert_eq!(t.eval(0.25), 2.5);
        assert_eq!(t.eval(-1.0), 0.0);
        assert_eq!(t.eval(3.0), 10.0);
        assert_eq!(t.scaled(0.5).eval(1.0), 5.0);

        let one = Table { row: V::A, rows: vec![0.0, 2.0], col: None, cols: vec![], data: vec![0.0, 4.0] };
        one.validate().unwrap();
        assert_eq!(one.eval(&Vals(1.0, 0.0)), 2.0);
        // Bilinear: f = a + 10·b on the grid.
        let two = Table {
            row: V::A,
            rows: vec![0.0, 1.0],
            col: Some(V::B),
            cols: vec![0.0, 1.0, 2.0],
            data: vec![0.0, 10.0, 20.0, 1.0, 11.0, 21.0],
        };
        two.validate().unwrap();
        assert!((two.eval(&Vals(0.5, 1.5)) - 15.5).abs() < 1e-12);
        assert!((two.eval(&Vals(9.0, -1.0)) - 1.0).abs() < 1e-12);
        let bad = Table { data: vec![0.0; 5], ..two };
        assert!(bad.validate().is_err());
    }
}
