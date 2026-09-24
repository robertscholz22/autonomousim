//! Small dense linear algebra: skew matrices, fixed ≤6×6 SPD solves and a dynamic dense
//! matrix used for joint-space quantities (mass matrix, CRBA).

use glam::{DMat3, DVec3};

/// Skew-symmetric cross-product matrix: `skew(a) * b == a.cross(b)`.
#[inline]
pub fn skew(a: DVec3) -> DMat3 {
    DMat3::from_cols(DVec3::new(0.0, a.z, -a.y), DVec3::new(-a.z, 0.0, a.x), DVec3::new(a.y, -a.x, 0.0))
}

/// Largest absolute element of a 3×3 matrix.
#[inline]
pub fn mat3_max_abs(m: DMat3) -> f64 {
    m.x_axis.abs().max_element().max(m.y_axis.abs().max_element()).max(m.z_axis.abs().max_element())
}

/// Row-major 6×6 matrix used for small joint-space blocks (`k ≤ 6`).
pub type Mat6 = [[f64; 6]; 6];

/// Error returned when a matrix is not (numerically) symmetric positive definite.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("matrix is not symmetric positive definite")]
pub struct NotPositiveDefinite;

/// In-place Cholesky factorisation `A = L Lᵀ` of the leading `k×k` block (lower triangle
/// holds `L`). Only the lower triangle of `a` is read.
pub fn cholesky6(a: &mut Mat6, k: usize) -> Result<(), NotPositiveDefinite> {
    for j in 0..k {
        let mut d = a[j][j];
        for p in 0..j {
            d -= a[j][p] * a[j][p];
        }
        if d <= 0.0 || !d.is_finite() {
            return Err(NotPositiveDefinite);
        }
        let ljj = d.sqrt();
        a[j][j] = ljj;
        for i in (j + 1)..k {
            let mut s = a[i][j];
            for p in 0..j {
                s -= a[i][p] * a[j][p];
            }
            a[i][j] = s / ljj;
        }
    }
    Ok(())
}

/// Solve `L Lᵀ x = b` in place for the leading `k` entries, given a factor from [`cholesky6`].
pub fn cholesky6_solve(l: &Mat6, k: usize, b: &mut [f64]) {
    for i in 0..k {
        let mut s = b[i];
        for p in 0..i {
            s -= l[i][p] * b[p];
        }
        b[i] = s / l[i][i];
    }
    for i in (0..k).rev() {
        let mut s = b[i];
        for p in (i + 1)..k {
            s -= l[p][i] * b[p];
        }
        b[i] = s / l[i][i];
    }
}

/// Error returned when a general matrix is (numerically) singular.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("matrix is singular")]
pub struct SingularMatrix;

/// Row pivots of an LU factorisation from [`lu_factor6`] (`piv[k]` = row swapped with row `k`
/// at elimination step `k`).
pub type Pivots6 = [usize; 6];

/// In-place LU factorisation with partial pivoting, `P A = L U` (unit-diagonal `L` below the
/// diagonal, `U` on and above it).
pub fn lu_factor6(a: &mut Mat6) -> Result<Pivots6, SingularMatrix> {
    let mut piv = [0; 6];
    for col in 0..6 {
        let mut p = col;
        for row in (col + 1)..6 {
            if a[row][col].abs() > a[p][col].abs() {
                p = row;
            }
        }
        if !a[p][col].is_finite() || a[p][col].abs() <= 1e-300 {
            return Err(SingularMatrix);
        }
        piv[col] = p;
        a.swap(col, p);
        let inv = 1.0 / a[col][col];
        for row in (col + 1)..6 {
            let f = a[row][col] * inv;
            a[row][col] = f;
            for c in (col + 1)..6 {
                a[row][c] -= f * a[col][c];
            }
        }
    }
    Ok(piv)
}

/// Solve `A x = b` in place given the factorisation from [`lu_factor6`].
pub fn lu_solve_factored6(lu: &Mat6, piv: &Pivots6, b: &mut [f64; 6]) {
    for (k, &p) in piv.iter().enumerate() {
        b.swap(k, p);
    }
    for row in 1..6 {
        for c in 0..row {
            b[row] -= lu[row][c] * b[c];
        }
    }
    for row in (0..6).rev() {
        for c in (row + 1)..6 {
            b[row] -= lu[row][c] * b[c];
        }
        b[row] /= lu[row][row];
    }
}

/// Solve a general 6×6 system `A x = b` in place (`a` is overwritten by its LU factors).
pub fn lu_solve6(a: &mut Mat6, b: &mut [f64; 6]) -> Result<(), SingularMatrix> {
    let piv = lu_factor6(a)?;
    lu_solve_factored6(a, &piv, b);
    Ok(())
}

/// Inverse of an SPD `k×k` block via Cholesky (result is full and symmetric).
pub fn spd_inverse6(a: &Mat6, k: usize) -> Result<Mat6, NotPositiveDefinite> {
    let mut l = *a;
    cholesky6(&mut l, k)?;
    let mut inv = [[0.0; 6]; 6];
    for c in 0..k {
        let mut e = [0.0; 6];
        e[c] = 1.0;
        cholesky6_solve(&l, k, &mut e);
        for r in 0..k {
            inv[r][c] = e[r];
        }
    }
    Ok(inv)
}

/// Row-major dynamically sized dense matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct DenseMatrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl DenseMatrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self { rows, cols, data: vec![0.0; rows * cols] }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m[(i, i)] = 1.0;
        }
        m
    }

    pub fn mul_vec(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(x.len(), self.cols);
        (0..self.rows)
            .map(|r| self.data[r * self.cols..(r + 1) * self.cols].iter().zip(x).map(|(a, b)| a * b).sum())
            .collect()
    }

    /// Solve `A x = b` for SPD `A` by Cholesky (does not modify `self`).
    pub fn solve_spd(&self, b: &[f64]) -> Result<Vec<f64>, NotPositiveDefinite> {
        assert_eq!(self.rows, self.cols);
        let n = self.rows;
        let mut l = self.clone();
        for j in 0..n {
            let mut d = l[(j, j)];
            for p in 0..j {
                d -= l[(j, p)] * l[(j, p)];
            }
            if d <= 0.0 || !d.is_finite() {
                return Err(NotPositiveDefinite);
            }
            let ljj = d.sqrt();
            l[(j, j)] = ljj;
            for i in (j + 1)..n {
                let mut s = l[(i, j)];
                for p in 0..j {
                    s -= l[(i, p)] * l[(j, p)];
                }
                l[(i, j)] = s / ljj;
            }
        }
        let mut x = b.to_vec();
        for i in 0..n {
            let mut s = x[i];
            for p in 0..i {
                s -= l[(i, p)] * x[p];
            }
            x[i] = s / l[(i, i)];
        }
        for i in (0..n).rev() {
            let mut s = x[i];
            for p in (i + 1)..n {
                s -= l[(p, i)] * x[p];
            }
            x[i] = s / l[(i, i)];
        }
        Ok(x)
    }

    pub fn max_abs_diff(&self, o: &DenseMatrix) -> f64 {
        self.data.iter().zip(&o.data).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max)
    }
}

impl std::ops::Index<(usize, usize)> for DenseMatrix {
    type Output = f64;
    #[inline]
    fn index(&self, (r, c): (usize, usize)) -> &f64 {
        &self.data[r * self.cols + c]
    }
}

impl std::ops::IndexMut<(usize, usize)> for DenseMatrix {
    #[inline]
    fn index_mut(&mut self, (r, c): (usize, usize)) -> &mut f64 {
        &mut self.data[r * self.cols + c]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn spd6(k: usize, seed: &[f64]) -> Mat6 {
        // A = B Bᵀ + k·1 is SPD.
        let mut b = [[0.0; 6]; 6];
        for r in 0..k {
            for c in 0..k {
                b[r][c] = seed[(r * 6 + c) % seed.len()];
            }
        }
        let mut a = [[0.0; 6]; 6];
        for r in 0..k {
            for c in 0..k {
                a[r][c] = (0..k).map(|p| b[r][p] * b[c][p]).sum::<f64>() + if r == c { k as f64 } else { 0.0 };
            }
        }
        a
    }

    proptest! {
        #[test]
        fn skew_matches_cross(a in prop::array::uniform3(-10.0..10.0f64), b in prop::array::uniform3(-10.0..10.0f64)) {
            let (a, b) = (DVec3::from_array(a), DVec3::from_array(b));
            prop_assert!((skew(a) * b - a.cross(b)).length() < 1e-12);
        }

        #[test]
        fn cholesky6_solves(k in 1usize..=6, seed in prop::collection::vec(-2.0..2.0f64, 36), x in prop::array::uniform6(-5.0..5.0f64)) {
            let a = spd6(k, &seed);
            let mut b = [0.0; 6];
            for r in 0..k { b[r] = (0..k).map(|c| a[r][c] * x[c]).sum(); }
            let mut l = a;
            cholesky6(&mut l, k).unwrap();
            cholesky6_solve(&l, k, &mut b);
            for r in 0..k { prop_assert!((b[r] - x[r]).abs() < 1e-8); }
            let inv = spd_inverse6(&a, k).unwrap();
            for r in 0..k { for c in 0..k {
                let e: f64 = (0..k).map(|p| a[r][p] * inv[p][c]).sum();
                let expected = if r == c { 1.0 } else { 0.0 };
                prop_assert!((e - expected).abs() < 1e-9);
            }}
        }

        #[test]
        fn dense_solve_spd(n in 1usize..12, seed in prop::collection::vec(-2.0..2.0f64, 144)) {
            let mut b = DenseMatrix::zeros(n, n);
            for r in 0..n { for c in 0..n { b[(r, c)] = seed[r * 12 + c]; } }
            let mut a = DenseMatrix::identity(n);
            for r in 0..n { for c in 0..n { a[(r, c)] += (0..n).map(|p| b[(r, p)] * b[(c, p)]).sum::<f64>(); } }
            let x: Vec<f64> = (0..n).map(|i| i as f64 - 3.0).collect();
            let rhs = a.mul_vec(&x);
            let sol = a.solve_spd(&rhs).unwrap();
            for i in 0..n { prop_assert!((sol[i] - x[i]).abs() < 1e-8); }
        }
    }

    #[test]
    fn lu_solve6_general() {
        let mut a = [[0.0; 6]; 6];
        for r in 0..6 {
            for c in 0..6 {
                a[r][c] = ((r * 7 + c * 3) % 11) as f64 - 5.0 + if r == c { 12.0 } else { 0.0 };
            }
        }
        let x = [1.0, -2.0, 3.0, 0.5, -0.25, 4.0];
        let mut b = [0.0; 6];
        for r in 0..6 {
            b[r] = (0..6).map(|c| a[r][c] * x[c]).sum();
        }
        let mut lu = a;
        lu_solve6(&mut lu, &mut b).unwrap();
        for r in 0..6 {
            assert!((b[r] - x[r]).abs() < 1e-12);
        }
    }

    #[test]
    fn rejects_indefinite() {
        let mut a = [[0.0; 6]; 6];
        a[0][0] = 1.0;
        a[1][1] = -1.0;
        assert_eq!(cholesky6(&mut a, 2), Err(NotPositiveDefinite));
    }
}
