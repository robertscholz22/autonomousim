//! Weighted least-effort control allocation with effector bounds and objective priorities.

/// Effectors at most: one thrust per rotor, differential tilt, aileron, elevator, rudder.
pub(crate) const EFFECTORS: usize = 8;
/// Objectives: roll, pitch and yaw moment, total thrust.
pub(crate) const OBJECTIVES: usize = 4;

/// Increments `u` of the effectors with the least effort `Σ w_j·u_j²` that meet
/// `Σ_j b[i][j]·u_j = v[i]` for each objective, within `lo ≤ u ≤ hi`. Effectors that would
/// leave their bounds are fixed at them one at a time (the worst first) and the others take up
/// the rest. When the free effectors can no longer meet every objective, objectives are
/// dropped in the order of `drop` (least important first).
pub(crate) fn allocate(
    b: &[[f64; EFFECTORS]; OBJECTIVES],
    w: &[f64; EFFECTORS],
    v: &[f64; OBJECTIVES],
    lo: &[f64; EFFECTORS],
    hi: &[f64; EFFECTORS],
    drop: &[usize],
) -> [f64; EFFECTORS] {
    let mut free = [true; EFFECTORS];
    let mut rows = [true; OBJECTIVES];
    let mut u = [0.0; EFFECTORS];
    let mut dropped = 0;
    for _ in 0..=EFFECTORS {
        // What the fixed effectors leave to the free ones.
        let mut rest = *v;
        for (i, r) in rest.iter_mut().enumerate() {
            *r -= (0..EFFECTORS).filter(|&j| !free[j]).map(|j| b[i][j] * u[j]).sum::<f64>();
        }
        let solved = loop {
            match min_norm(b, w, &rest, &free, &rows) {
                Some(x) => break Some(x),
                None if dropped < drop.len() => {
                    rows[drop[dropped]] = false;
                    dropped += 1;
                }
                None => break None,
            }
        };
        let Some(x) = solved else { break };
        for j in (0..EFFECTORS).filter(|&j| free[j]) {
            u[j] = x[j];
        }
        // The effector furthest outside its bounds (relative to its range) is fixed there.
        let worst = (0..EFFECTORS)
            .filter(|&j| free[j])
            .map(|j| {
                let range = (hi[j] - lo[j]).max(1e-12);
                (j, ((lo[j] - u[j]).max(u[j] - hi[j])) / range)
            })
            .filter(|&(_, over)| over > 1e-12)
            .max_by(|a, b| a.1.total_cmp(&b.1));
        match worst {
            Some((j, _)) => {
                u[j] = u[j].clamp(lo[j], hi[j]);
                free[j] = false;
            }
            None => return u,
        }
    }
    std::array::from_fn(|j| u[j].clamp(lo[j], hi[j]))
}

/// `u = W⁻¹Bᵀ(BW⁻¹Bᵀ)⁻¹v` over the free effectors and the kept objectives; `None` when the
/// free effectors cannot meet them (a singular system).
fn min_norm(
    b: &[[f64; EFFECTORS]; OBJECTIVES],
    w: &[f64; EFFECTORS],
    v: &[f64; OBJECTIVES],
    free: &[bool; EFFECTORS],
    rows: &[bool; OBJECTIVES],
) -> Option<[f64; EFFECTORS]> {
    let idx: Vec<usize> = (0..OBJECTIVES).filter(|&i| rows[i]).collect();
    let n = idx.len();
    if n == 0 {
        return Some([0.0; EFFECTORS]);
    }
    // A = B W⁻¹ Bᵀ on the kept rows, augmented with v.
    let mut a = [[0.0; OBJECTIVES + 1]; OBJECTIVES];
    for (r, &i) in idx.iter().enumerate() {
        for (c, &k) in idx.iter().enumerate() {
            a[r][c] = (0..EFFECTORS).filter(|&j| free[j]).map(|j| b[i][j] * b[k][j] / w[j]).sum();
        }
        a[r][n] = v[i];
    }
    let scale = (0..n).map(|r| a[r][r].abs()).fold(0.0, f64::max);
    if scale.is_nan() || scale <= 0.0 {
        return None;
    }
    // Gaussian elimination with partial pivoting.
    for col in 0..n {
        let p = (col..n).max_by(|&x, &y| a[x][col].abs().total_cmp(&a[y][col].abs()))?;
        if a[p][col].abs() < 1e-9 * scale {
            return None;
        }
        a.swap(col, p);
        for r in 0..n {
            if r != col {
                let f = a[r][col] / a[col][col];
                for c in col..=n {
                    a[r][c] -= f * a[col][c];
                }
            }
        }
    }
    let lambda: Vec<f64> = (0..n).map(|r| a[r][n] / a[r][r]).collect();
    let mut u = [0.0; EFFECTORS];
    for j in (0..EFFECTORS).filter(|&j| free[j]) {
        u[j] = idx.iter().zip(&lambda).map(|(&i, l)| b[i][j] * l).sum::<f64>() / w[j];
    }
    u.iter().all(|x| x.is_finite()).then_some(u)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn b() -> [[f64; EFFECTORS]; OBJECTIVES] {
        // Four rotors in an X (roll, pitch, yaw by torque, thrust) and one surface on roll.
        let mut b = [[0.0; EFFECTORS]; OBJECTIVES];
        b[0][..5].copy_from_slice(&[1.0, -1.0, 1.0, -1.0, 2.0]);
        b[1][..4].copy_from_slice(&[-1.0, -1.0, 1.0, 1.0]);
        b[2][..4].copy_from_slice(&[0.1, -0.1, -0.1, 0.1]);
        b[3][..4].copy_from_slice(&[1.0; 4]);
        b
    }

    #[test]
    fn meets_the_objectives_with_least_effort() {
        let (b, w) = (b(), [1.0; EFFECTORS]);
        let v = [0.5, 0.2, 0.01, 1.0];
        let wide = [1e9; EFFECTORS];
        let u = allocate(&b, &w, &v, &wide.map(|x| -x), &wide, &[2, 3]);
        for i in 0..OBJECTIVES {
            let got: f64 = (0..EFFECTORS).map(|j| b[i][j] * u[j]).sum();
            assert!((got - v[i]).abs() < 1e-12, "{i}: {got}");
        }
        // The surface, twice as effective on roll, moves twice as far as each rotor's roll
        // share (least effort: in proportion to effectiveness over weight).
        let rotor_roll = (u[0] - u[1] + u[2] - u[3]) / 4.0;
        assert!((u[4] - 2.0 * rotor_roll).abs() < 1e-12 && u[5..].iter().all(|x| *x == 0.0), "{u:?}");
    }

    #[test]
    fn saturated_effectors_hand_over_and_objectives_drop_in_order() {
        let (b, w) = (b(), [1.0; EFFECTORS]);
        let v = [0.5, 0.2, 0.01, 1.0];
        let mut hi = [1e9; EFFECTORS];
        hi[4] = 0.05;
        let lo = hi.map(|_| -1e9);
        let u = allocate(&b, &w, &v, &lo, &hi, &[2, 3]);
        assert_eq!(u[4], 0.05);
        for i in 0..OBJECTIVES {
            let got: f64 = (0..EFFECTORS).map(|j| b[i][j] * u[j]).sum();
            assert!((got - v[i]).abs() < 1e-12, "{i}: {got}");
        }
        // With every rotor pinned, no rotor can move: the objectives go, yaw first; roll
        // is left to the surface alone.
        let lo = [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0];
        let hi = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let u = allocate(&b, &w, &v, &lo, &hi, &[2, 3, 1]);
        assert!((u[4] - 0.25).abs() < 1e-12, "{u:?}");
    }
}
