//! Time integration of multibody states.

use super::{MbState, MultibodyModel};
use crate::math::linalg::{lu_factor6, lu_solve_factored6};
use crate::math::{RigidInertia, SpatialForce, SpatialMotion, skew};
use glam::{DMat3, DVec3};
use serde::{Deserialize, Serialize};

/// Integration scheme.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Integrator {
    /// Symplectic (semi-implicit) Euler: `v ← v + h q̈(q, v)`, then `q ← q ⊕ h v`. One force
    /// evaluation per step; the default for the simulator.
    #[default]
    SemiImplicitEuler,
    /// Classical 4th-order Runge–Kutta on the raw coordinates (quaternions renormalised).
    /// Four force evaluations per step; used for validation and optionally at run time.
    Rk4,
}

/// Semi-implicit Euler update given accelerations `qdd` already evaluated at `(q, v)`.
///
/// For a model that is a single free-floating body (every aerial vehicle) the velocity-product
/// (gyroscopic) term is re-evaluated with the implicit midpoint rule instead of explicitly.
/// Explicit treatment makes free rotation gain energy (≈1 %/s at 6 rad/s and 2 ms); the
/// midpoint rule conserves the kinetic energy of torque-free motion exactly. Articulated
/// models use the plain scheme.
pub fn semi_implicit_euler(model: &MultibodyModel, state: &mut MbState, qdd: &[f64], dt: f64) {
    semi_implicit_euler_with_momentum(model, state, qdd, dt, DVec3::ZERO);
}

/// [`semi_implicit_euler`] for a single free body carrying internal angular momentum `h_int`
/// (body frame, e.g. spinning rotors), whose gyroscopic torque `−ω × h_int` is *not* part of
/// `qdd` and is instead included implicitly in the midpoint update. Treated explicitly, this
/// skew-symmetric coupling amplifies the rates by `1 + (h·|h_int|/I)²` per step.
///
/// # Panics
/// In debug builds, if `h_int ≠ 0` and the model is not a single free body.
pub fn semi_implicit_euler_with_momentum(
    model: &MultibodyModel,
    state: &mut MbState,
    qdd: &[f64],
    dt: f64,
    h_int: DVec3,
) {
    if let Some(inertia) = model.single_free_body_inertia() {
        let v1 = SpatialMotion::from_array(state.v[..6].try_into().unwrap());
        let a = SpatialMotion::from_array(qdd[..6].try_into().unwrap());
        let v2 = midpoint_gyroscopic_update(inertia, v1, a, dt, h_int);
        state.v[..6].copy_from_slice(&v2.to_array());
    } else {
        debug_assert!(h_int == DVec3::ZERO, "internal momentum needs a single free body");
        for (v, a) in state.v.iter_mut().zip(qdd) {
            *v += a * dt;
        }
    }
    for (i, link) in model.links().iter().enumerate() {
        let (qo, vo) = (model.q_offset(i), model.v_offset(i));
        let (nq, nv) = (link.joint.nq(), link.joint.nv());
        link.joint.integrate(&mut state.q[qo..qo + nq], &state.v[vo..vo + nv], dt);
    }
}

/// Velocity update of a free rigid body where the explicit acceleration `a_exp` (evaluated at
/// `v1`, containing `-I⁻¹ (v1 ×* I v1)`) has its velocity-product term replaced by the midpoint
/// value: solves `I (v2 - v1) + h (vm ×* (I vm + h_int)) = h f` with `vm = (v1 + v2)/2`, where
/// `h_int` is internal angular momentum (body frame) that `a_exp` does not account for.
///
/// Simplified Newton: the Jacobian `I + h/2 (δ ×* (I vm + h_int) + vm ×* I δ)` is factored once
/// at the explicit predictor. The predictor is `O(h²)` from the solution without internal
/// momentum, so the iteration contracts by roughly `h² |ω|²` per step and reaches round-off in
/// two or three back-substitutions.
pub fn midpoint_gyroscopic_update(
    inertia: &RigidInertia,
    v1: SpatialMotion,
    a_exp: SpatialMotion,
    h: f64,
    h_int: DVec3,
) -> SpatialMotion {
    if inertia.com == DVec3::ZERO {
        midpoint_centered(inertia.mass, &inertia.i_com, v1, a_exp, h, h_int)
    } else {
        midpoint_general(inertia, v1, a_exp, h, h_int)
    }
}

/// [`midpoint_gyroscopic_update`] for a body frame at the centre of mass, where the spatial
/// inertia is block-diagonal: a 3×3 Newton solve for the angular velocity, then the linear
/// velocity `(1 + [k]×) u₂ = (1 − [k]×) u₁ + h·f/m` with `k = h·ω_m/2` in closed form.
fn midpoint_centered(
    m: f64,
    ic: &DMat3,
    v1: SpatialMotion,
    a_exp: SpatialMotion,
    h: f64,
    h_int: DVec3,
) -> SpatialMotion {
    let (w1, u1) = (v1.ang, v1.lin);
    let hf_ang = (*ic * a_exp.ang + w1.cross(*ic * w1)) * h;
    let hf_lin = (a_exp.lin + w1.cross(u1)) * h; // per unit mass
    debug_assert!(m > 0.0);

    let mut w2 = w1 + a_exp.ang * h;
    let wm = 0.5 * (w1 + w2);
    // Jacobian of I(w2 − w1) + h·wm × (I·wm + h_int): I + h/2 ([wm]× I − [I·wm + h_int]×).
    let jac = *ic + (skew(wm) * *ic - skew(*ic * wm + h_int)) * (0.5 * h);
    let jinv = jac.inverse();
    for _ in 0..6 {
        let wm = 0.5 * (w1 + w2);
        let r = *ic * (w2 - w1) + wm.cross(*ic * wm + h_int) * h - hf_ang;
        let d = jinv * r;
        w2 -= d;
        if d.length_squared() <= 1e-30 * (1.0 + w2.length_squared()) {
            break;
        }
    }
    let k = (w1 + w2) * (0.25 * h);
    let rhs = u1 - k.cross(u1) + hf_lin;
    // (1 + [k]×)⁻¹ = (1 − [k]× + k kᵀ) / (1 + |k|²)
    let u2 = (rhs - k.cross(rhs) + k * k.dot(rhs)) / (1.0 + k.length_squared());
    SpatialMotion::new(w2, u2)
}

fn midpoint_general(
    inertia: &RigidInertia,
    v1: SpatialMotion,
    a_exp: SpatialMotion,
    h: f64,
    h_int: DVec3,
) -> SpatialMotion {
    let iv1 = inertia.mul_motion(v1);
    let hint = SpatialForce::new(h_int, DVec3::ZERO);
    // h f = h (I a_exp + v1 ×* I v1)
    let hf = (inertia.mul_motion(a_exp) + v1.cross_force(iv1)) * h;
    let mut v2 = v1 + a_exp * h;

    let vm = (v1 + v2) * 0.5;
    let ivm = inertia.mul_motion(vm) + hint;
    let mut jac = inertia.to_articulated().to_mat6();
    for k in 0..6 {
        let mut e = [0.0; 6];
        e[k] = 1.0;
        let e = SpatialMotion::from_array(e);
        let ie =
            SpatialForce::new(DVec3::new(jac[0][k], jac[1][k], jac[2][k]), DVec3::new(jac[3][k], jac[4][k], jac[5][k]));
        let col = ((e.cross_force(ivm) + vm.cross_force(ie)) * (0.5 * h)).to_array();
        for row in 0..6 {
            jac[row][k] += col[row];
        }
    }
    let Ok(piv) = lu_factor6(&mut jac) else { return v2 };

    for _ in 0..6 {
        let vm = (v1 + v2) * 0.5;
        let r = inertia.mul_motion(v2 - v1) + vm.cross_force(inertia.mul_motion(vm) + hint) * h - hf;
        let mut delta = r.to_array();
        lu_solve_factored6(&jac, &piv, &mut delta);
        let delta = SpatialMotion::from_array(delta);
        v2 -= delta;
        if delta.dot6(delta) <= 1e-30 * (1.0 + v2.dot6(v2)) {
            break;
        }
    }
    v2
}

/// Reusable buffers for [`rk4`].
#[derive(Clone, Debug, Default)]
pub struct Rk4Workspace {
    q0: Vec<f64>,
    v0: Vec<f64>,
    kq: [Vec<f64>; 4],
    kv: [Vec<f64>; 4],
}

impl Rk4Workspace {
    pub fn new(model: &MultibodyModel) -> Self {
        let (nq, nv) = (model.nq(), model.nv());
        Self {
            q0: vec![0.0; nq],
            v0: vec![0.0; nv],
            kq: std::array::from_fn(|_| vec![0.0; nq]),
            kv: std::array::from_fn(|_| vec![0.0; nv]),
        }
    }
}

/// One classical RK4 step. `accel(q, v, qdd)` must write the joint accelerations at `(q, v)`
/// (e.g. by calling the ABA). Quaternion coordinates are renormalised at the end.
pub fn rk4<E>(
    model: &MultibodyModel,
    state: &mut MbState,
    dt: f64,
    ws: &mut Rk4Workspace,
    mut accel: impl FnMut(&[f64], &[f64], &mut [f64]) -> Result<(), E>,
) -> Result<(), E> {
    ws.q0.copy_from_slice(&state.q);
    ws.v0.copy_from_slice(&state.v);
    let stage_dt = [0.0, 0.5 * dt, 0.5 * dt, dt];
    for s in 0..4 {
        if s > 0 {
            for (j, q) in state.q.iter_mut().enumerate() {
                *q = ws.q0[j] + stage_dt[s] * ws.kq[s - 1][j];
            }
            for (j, v) in state.v.iter_mut().enumerate() {
                *v = ws.v0[j] + stage_dt[s] * ws.kv[s - 1][j];
            }
        }
        qdot(model, &state.q, &state.v, &mut ws.kq[s]);
        accel(&state.q, &state.v, &mut ws.kv[s])?;
    }
    for (j, q) in state.q.iter_mut().enumerate() {
        *q = ws.q0[j] + dt / 6.0 * (ws.kq[0][j] + 2.0 * ws.kq[1][j] + 2.0 * ws.kq[2][j] + ws.kq[3][j]);
    }
    for (j, v) in state.v.iter_mut().enumerate() {
        *v = ws.v0[j] + dt / 6.0 * (ws.kv[0][j] + 2.0 * ws.kv[1][j] + 2.0 * ws.kv[2][j] + ws.kv[3][j]);
    }
    normalize(model, state);
    Ok(())
}

/// Time derivative of the raw position coordinates.
pub fn qdot(model: &MultibodyModel, q: &[f64], v: &[f64], out: &mut [f64]) {
    for (i, link) in model.links().iter().enumerate() {
        let (qo, vo) = (model.q_offset(i), model.v_offset(i));
        let (nq, nv) = (link.joint.nq(), link.joint.nv());
        link.joint.qdot(&q[qo..qo + nq], &v[vo..vo + nv], &mut out[qo..qo + nq]);
    }
}

/// Renormalise all quaternion coordinates.
pub fn normalize(model: &MultibodyModel, state: &mut MbState) {
    for (i, link) in model.links().iter().enumerate() {
        let qo = model.q_offset(i);
        link.joint.normalize(&mut state.q[qo..qo + link.joint.nq()]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Seed;

    #[test]
    fn centered_midpoint_matches_general_solver() {
        let mut rng = Seed::from_u64(11).rng();
        let mut v3 = |s: f64| DVec3::new(rng.range(-s, s), rng.range(-s, s), rng.range(-s, s));
        for _ in 0..500 {
            // Random SPD inertia about the COM.
            let (a, b) = (v3(1.0), v3(1.0));
            let i_com = DMat3::from_diagonal(DVec3::new(0.02, 0.03, 0.05))
                + 0.01 * crate::math::outer(a, a)
                + 0.01 * crate::math::outer(b, b);
            let inertia = RigidInertia::new(1.3, DVec3::ZERO, i_com);
            let v1 = SpatialMotion::new(v3(8.0), v3(10.0));
            let acc = SpatialMotion::new(v3(50.0), v3(20.0));
            let h_int = v3(0.05);
            for h in [0.0005, 0.002, 0.01] {
                let fast = midpoint_centered(inertia.mass, &inertia.i_com, v1, acc, h, h_int);
                let slow = midpoint_general(&inertia, v1, acc, h, h_int);
                let d = fast - slow;
                assert!(d.dot6(d).sqrt() < 1e-11 * (1.0 + slow.dot6(slow).sqrt()), "{fast:?} vs {slow:?}");
            }
        }
    }
}
