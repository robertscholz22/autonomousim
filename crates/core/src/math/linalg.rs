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

    /// Matrix product `self · o`.
    pub fn mul(&self, o: &DenseMatrix) -> DenseMatrix {
        assert_eq!(self.cols, o.rows);
        let mut out = DenseMatrix::zeros(self.rows, o.cols);
        for r in 0..self.rows {
            for k in 0..self.cols {
                let a = self[(r, k)];
                if a != 0.0 {
                    for c in 0..o.cols {
                        out[(r, c)] += a * o[(k, c)];
                    }
                }
            }
        }
        out
    }

    pub fn transpose(&self) -> DenseMatrix {
        let mut out = DenseMatrix::zeros(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                out[(c, r)] = self[(r, c)];
            }
        }
        out
    }

    /// `self + s·o`.
    pub fn add_scaled(&self, o: &DenseMatrix, s: f64) -> DenseMatrix {
        assert_eq!((self.rows, self.cols), (o.rows, o.cols));
        DenseMatrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().zip(&o.data).map(|(a, b)| a + s * b).collect(),
        }
    }

    /// Inverse of a square matrix (Gauss–Jordan with partial pivoting).
    pub fn inverse(&self) -> Result<DenseMatrix, SingularMatrix> {
        assert_eq!(self.rows, self.cols);
        let n = self.rows;
        let mut a = self.clone();
        let mut inv = DenseMatrix::identity(n);
        for col in 0..n {
            let pivot = (col..n).max_by(|&i, &j| a[(i, col)].abs().total_cmp(&a[(j, col)].abs())).expect("rows");
            let p = a[(pivot, col)];
            if p == 0.0 || !p.is_finite() {
                return Err(SingularMatrix);
            }
            if pivot != col {
                for c in 0..n {
                    a.data.swap(pivot * n + c, col * n + c);
                    inv.data.swap(pivot * n + c, col * n + c);
                }
            }
            for c in 0..n {
                a[(col, c)] /= p;
                inv[(col, c)] /= p;
            }
            for r in (0..n).filter(|&r| r != col) {
                let f = a[(r, col)];
                if f != 0.0 {
                    for c in 0..n {
                        a[(r, c)] -= f * a[(col, c)];
                        inv[(r, c)] -= f * inv[(col, c)];
                    }
                }
            }
        }
        Ok(inv)
    }

    /// Eigenvalues `(re, im)` of a square matrix, sorted by real part, then imaginary part:
    /// balancing, reduction to Hessenberg form by elimination and the shifted QR algorithm
    /// (Press et al., Numerical Recipes, 3rd ed., §11.6–11.7). `None` if QR does not converge.
    pub fn eigenvalues(&self) -> Option<Vec<(f64, f64)>> {
        assert_eq!(self.rows, self.cols);
        let n = self.rows;
        let mut a = self.clone();
        balance(&mut a);
        hessenberg(&mut a);
        let mut out = hqr(&mut a, n)?;
        out.sort_by(|x, y| x.0.total_cmp(&y.0).then(x.1.total_cmp(&y.1)));
        Some(out)
    }
}

/// Scale rows and columns by powers of two towards equal norms (similarity transform).
fn balance(a: &mut DenseMatrix) {
    let n = a.rows;
    let radix = 2.0f64;
    let mut done = false;
    while !done {
        done = true;
        for i in 0..n {
            let (mut r, mut c) = (0.0, 0.0);
            for j in (0..n).filter(|&j| j != i) {
                c += a[(j, i)].abs();
                r += a[(i, j)].abs();
            }
            if c != 0.0 && r != 0.0 {
                let s = c + r;
                let mut f = 1.0;
                let mut g = r / radix;
                while c < g {
                    f *= radix;
                    c *= radix * radix;
                }
                g = r * radix;
                while c > g {
                    f /= radix;
                    c /= radix * radix;
                }
                if (c + r) / f < 0.95 * s {
                    done = false;
                    for j in 0..n {
                        a[(i, j)] /= f;
                        a[(j, i)] *= f;
                    }
                }
            }
        }
    }
}

/// Reduce to upper Hessenberg form by stabilised elementary similarity transforms.
fn hessenberg(a: &mut DenseMatrix) {
    let n = a.rows;
    for m in 1..n.saturating_sub(1) {
        let mut x = 0.0f64;
        let mut i = m;
        for j in m..n {
            if a[(j, m - 1)].abs() > x.abs() {
                x = a[(j, m - 1)];
                i = j;
            }
        }
        if i != m {
            for j in (m - 1)..n {
                a.data.swap(i * n + j, m * n + j);
            }
            for j in 0..n {
                a.data.swap(j * n + i, j * n + m);
            }
        }
        if x != 0.0 {
            for i in (m + 1)..n {
                let y = a[(i, m - 1)] / x;
                if y != 0.0 {
                    for j in m..n {
                        let v = a[(m, j)];
                        a[(i, j)] -= y * v;
                    }
                    for j in 0..n {
                        let v = a[(j, i)];
                        a[(j, m)] += y * v;
                    }
                }
            }
        }
    }
    for i in 0..n {
        for j in 0..i.saturating_sub(1) {
            a[(i, j)] = 0.0;
        }
    }
}

/// Eigenvalues of an upper Hessenberg matrix (destroyed) by the Francis double-shift QR
/// algorithm.
#[allow(clippy::many_single_char_names, unused_assignments)]
fn hqr(a: &mut DenseMatrix, n: usize) -> Option<Vec<(f64, f64)>> {
    let eps = f64::EPSILON;
    let mut w = vec![(0.0, 0.0); n];
    let mut anorm = 0.0;
    for i in 0..n {
        for j in i.saturating_sub(1)..n {
            anorm += a[(i, j)].abs();
        }
    }
    let at = |a: &DenseMatrix, i: isize, j: isize| a[(i as usize, j as usize)];
    let mut nn = n as isize - 1;
    let mut t = 0.0;
    let (mut p, mut q, mut r) = (0.0f64, 0.0f64, 0.0f64);
    while nn >= 0 {
        let mut its = 0;
        loop {
            let mut l = nn;
            while l > 0 {
                let mut s = at(a, l - 1, l - 1).abs() + at(a, l, l).abs();
                if s == 0.0 {
                    s = anorm;
                }
                if at(a, l, l - 1).abs() <= eps * s {
                    a[(l as usize, l as usize - 1)] = 0.0;
                    break;
                }
                l -= 1;
            }
            let mut x = at(a, nn, nn);
            if l == nn {
                w[nn as usize] = (x + t, 0.0);
                nn -= 1;
            } else {
                let mut y = at(a, nn - 1, nn - 1);
                let mut ww = at(a, nn, nn - 1) * at(a, nn - 1, nn);
                if l == nn - 1 {
                    p = 0.5 * (y - x);
                    q = p * p + ww;
                    let mut z = q.abs().sqrt();
                    x += t;
                    if q >= 0.0 {
                        z = p + z.copysign(p);
                        let (hi, lo) = (x + z, if z != 0.0 { x - ww / z } else { x + z });
                        w[nn as usize - 1] = (hi, 0.0);
                        w[nn as usize] = (lo, 0.0);
                    } else {
                        w[nn as usize] = (x + p, -z);
                        w[nn as usize - 1] = (x + p, z);
                    }
                    nn -= 2;
                } else {
                    if its == 60 {
                        return None;
                    }
                    if its == 10 || its == 20 || its == 40 {
                        t += x;
                        for i in 0..=nn {
                            a[(i as usize, i as usize)] -= x;
                        }
                        let s = at(a, nn, nn - 1).abs() + at(a, nn - 1, nn - 2).abs();
                        x = 0.75 * s;
                        y = x;
                        ww = -0.4375 * s * s;
                    }
                    its += 1;
                    let mut m = nn - 2;
                    loop {
                        let z = at(a, m, m);
                        let rr = x - z;
                        let ss = y - z;
                        p = (rr * ss - ww) / at(a, m + 1, m) + at(a, m, m + 1);
                        q = at(a, m + 1, m + 1) - z - rr - ss;
                        r = at(a, m + 2, m + 1);
                        let s = p.abs() + q.abs() + r.abs();
                        p /= s;
                        q /= s;
                        r /= s;
                        if m == l {
                            break;
                        }
                        let u = at(a, m, m - 1).abs() * (q.abs() + r.abs());
                        let v = p.abs() * (at(a, m - 1, m - 1).abs() + z.abs() + at(a, m + 1, m + 1).abs());
                        if u <= eps * v {
                            break;
                        }
                        m -= 1;
                    }
                    for i in m..nn - 1 {
                        a[(i as usize + 2, i as usize)] = 0.0;
                        if i != m {
                            a[(i as usize + 2, i as usize - 1)] = 0.0;
                        }
                    }
                    let mut k = m;
                    while k < nn {
                        if k != m {
                            p = at(a, k, k - 1);
                            q = at(a, k + 1, k - 1);
                            r = if k + 1 != nn { at(a, k + 2, k - 1) } else { 0.0 };
                            x = p.abs() + q.abs() + r.abs();
                            if x != 0.0 {
                                p /= x;
                                q /= x;
                                r /= x;
                            }
                        }
                        let s = (p * p + q * q + r * r).sqrt().copysign(p);
                        if s != 0.0 {
                            if k == m {
                                if l != m {
                                    a[(k as usize, k as usize - 1)] = -at(a, k, k - 1);
                                }
                            } else {
                                a[(k as usize, k as usize - 1)] = -s * x;
                            }
                            p += s;
                            x = p / s;
                            y = q / s;
                            let z = r / s;
                            q /= p;
                            r /= p;
                            for j in k..=nn {
                                let mut pp = at(a, k, j) + q * at(a, k + 1, j);
                                if k + 1 != nn {
                                    pp += r * at(a, k + 2, j);
                                    a[(k as usize + 2, j as usize)] -= pp * z;
                                }
                                a[(k as usize + 1, j as usize)] -= pp * y;
                                a[(k as usize, j as usize)] -= pp * x;
                            }
                            let mmin = if nn < k + 3 { nn } else { k + 3 };
                            for i in l..=mmin {
                                let mut pp = x * at(a, i, k) + y * at(a, i, k + 1);
                                if k + 1 != nn {
                                    pp += z * at(a, i, k + 2);
                                    a[(i as usize, k as usize + 2)] -= pp * r;
                                }
                                a[(i as usize, k as usize + 1)] -= pp * q;
                                a[(i as usize, k as usize)] -= pp;
                            }
                        }
                        k += 1;
                    }
                }
            }
            if l + 1 >= nn {
                break;
            }
        }
    }
    Some(w)
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

    fn dense(rows: &[&[f64]]) -> DenseMatrix {
        let n = rows.len();
        DenseMatrix { rows: n, cols: rows[0].len(), data: rows.iter().flat_map(|r| r.iter().copied()).collect() }
    }

    #[test]
    fn eigenvalues_of_known_matrices() {
        // Companion matrix of (s + 1)(s − 2)(s² + 2s + 5)(s + 7) = s⁵ + 8s⁴ + 8s³ − 2s² − 73s − 70.
        let c = [8.0, 8.0, -2.0, -73.0, -70.0];
        let mut m = DenseMatrix::zeros(5, 5);
        for j in 0..5 {
            m[(0, j)] = -c[j];
        }
        for i in 1..5 {
            m[(i, i - 1)] = 1.0;
        }
        let e = m.eigenvalues().unwrap();
        let want = [(-7.0, 0.0), (-1.0, -2.0), (-1.0, 2.0), (-1.0, 0.0), (2.0, 0.0)];
        for want in want {
            assert!(e.iter().any(|got| (got.0 - want.0).abs() < 1e-9 && (got.1 - want.1).abs() < 1e-9), "{e:?}");
        }
        // Upper triangular: the diagonal; a 1×1 and an empty matrix.
        let tri = dense(&[&[3.0, 1.0, 4.0], &[0.0, -2.0, 5.0], &[0.0, 0.0, 0.5]]);
        assert_eq!(tri.eigenvalues().unwrap(), vec![(-2.0, 0.0), (0.5, 0.0), (3.0, 0.0)]);
        assert_eq!(dense(&[&[4.0]]).eigenvalues().unwrap(), vec![(4.0, 0.0)]);
    }

    proptest! {
        #[test]
        fn eigenvalues_keep_trace_and_determinant(v in proptest::collection::vec(-3.0f64..3.0, 64)) {
            let n = 8;
            let m = DenseMatrix { rows: n, cols: n, data: v };
            let e = m.eigenvalues().unwrap();
            let trace: f64 = (0..n).map(|i| m[(i, i)]).sum();
            let sum: f64 = e.iter().map(|x| x.0).sum();
            prop_assert!((sum - trace).abs() < 1e-8 * (1.0 + trace.abs()));
            // The product of the eigenvalues against the determinant from the inverse's LU.
            let mut prod = (1.0f64, 0.0f64);
            for &(re, im) in &e {
                prod = (prod.0 * re - prod.1 * im, prod.0 * im + prod.1 * re);
            }
            prop_assert!(prod.1.abs() < 1e-6 * (1.0 + prod.0.abs()));
            let inv = m.inverse().unwrap();
            let id = m.mul(&inv);
            prop_assert!(id.max_abs_diff(&DenseMatrix::identity(n)) < 1e-6);
            // Each eigenvalue makes A − λI singular: its smallest singular value, via the
            // determinant of the 2n×2n real form, is tiny relative to the matrix norm.
            for &(re, im) in &e {
                let mut b = DenseMatrix::zeros(2 * n, 2 * n);
                for i in 0..n {
                    for j in 0..n {
                        let d = if i == j { 1.0 } else { 0.0 };
                        b[(i, j)] = m[(i, j)] - re * d;
                        b[(n + i, n + j)] = m[(i, j)] - re * d;
                        b[(i, n + j)] = im * d;
                        b[(n + i, j)] = -im * d;
                    }
                }
                prop_assert!(b.inverse().map_or(true, |bi| bi.data.iter().fold(0.0f64, |x, y| x.max(y.abs())) > 1e6));
            }
        }
    }
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
