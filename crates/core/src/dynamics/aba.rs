//! Articulated-Body Algorithm (Featherstone, *Rigid Body Dynamics Algorithms*, Table 7.1),
//! with support for prescribed joints (hybrid dynamics, §9.2).

use super::{JointType, KinCache, MultibodyModel, forward_kinematics};
use crate::math::linalg::{Mat6, cholesky6, cholesky6_solve, spd_inverse6};
use crate::math::{ArticulatedInertia, NotPositiveDefinite, SpatialForce, SpatialMotion, outer};
use glam::DVec3;

/// Reusable buffers for forward dynamics of one model (allocation-free after construction).
#[derive(Clone, Debug)]
pub struct AbaWorkspace {
    pub kin: KinCache,
    ia: Vec<ArticulatedInertia>,
    pa: Vec<SpatialForce>,
    /// `U = I^A S` columns per link (up to 6).
    u_cols: Vec<[SpatialForce; 6]>,
    /// `D⁻¹` per link (leading k×k block).
    d_inv: Vec<Mat6>,
    /// `u = τ - Sᵀ p^A` per link.
    u: Vec<[f64; 6]>,
    /// Link spatial accelerations in link coordinates (**including** the fictitious base
    /// acceleration `-g`, i.e. proper accelerations).
    pub acc: Vec<SpatialMotion>,
    /// Joint accelerations (output). For prescribed joints this is an input.
    pub qdd: Vec<f64>,
    /// Joint forces required by prescribed joints (output; zero for free joints).
    pub tau_prescribed: Vec<f64>,
}

impl AbaWorkspace {
    pub fn new(model: &MultibodyModel) -> Self {
        let n = model.num_links();
        Self {
            kin: KinCache::new(model),
            ia: vec![ArticulatedInertia::ZERO; n],
            pa: vec![SpatialForce::ZERO; n],
            u_cols: vec![[SpatialForce::ZERO; 6]; n],
            d_inv: vec![[[0.0; 6]; 6]; n],
            u: vec![[0.0; 6]; n],
            acc: vec![SpatialMotion::ZERO; n],
            qdd: vec![0.0; model.nv()],
            tau_prescribed: vec![0.0; model.nv()],
        }
    }
}

/// Error from the forward-dynamics solver.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum DynamicsError {
    #[error("singular joint-space inertia at link {0} (massless subtree or bad parameters)")]
    Singular(usize),
}

impl From<(usize, NotPositiveDefinite)> for DynamicsError {
    fn from((i, _): (usize, NotPositiveDefinite)) -> Self {
        DynamicsError::Singular(i)
    }
}

/// Forward dynamics: computes `ws.qdd` (and link accelerations `ws.acc`) for state `(q, v)`,
/// generalised forces `tau` (length `nv`) and external link forces `f_ext` (one per link, in
/// link coordinates about the link origin; empty slice = none). `gravity` is the world gravity
/// vector. Runs forward kinematics into `ws.kin` first.
pub fn aba(
    model: &MultibodyModel,
    q: &[f64],
    v: &[f64],
    tau: &[f64],
    f_ext: &[SpatialForce],
    gravity: DVec3,
    ws: &mut AbaWorkspace,
) -> Result<(), DynamicsError> {
    forward_kinematics(model, q, v, &mut ws.kin);
    aba_with_kinematics(model, tau, f_ext, gravity, ws)
}

/// Like [`aba`] but reuses the kinematics already stored in `ws.kin`.
pub fn aba_with_kinematics(
    model: &MultibodyModel,
    tau: &[f64],
    f_ext: &[SpatialForce],
    gravity: DVec3,
    ws: &mut AbaWorkspace,
) -> Result<(), DynamicsError> {
    let links = model.links();
    let n = links.len();
    debug_assert_eq!(tau.len(), model.nv());
    debug_assert!(f_ext.is_empty() || f_ext.len() == n);

    // Pass 1: bias forces (velocities come from the kinematics cache).
    for i in 0..n {
        let inertia = &links[i].inertia;
        let vel = ws.kin.vel[i];
        ws.ia[i] = inertia.to_articulated();
        let mut p = vel.cross_force(inertia.mul_motion(vel));
        if !f_ext.is_empty() {
            p -= f_ext[i];
        }
        ws.pa[i] = p;
    }

    // Pass 2: articulated inertias and bias forces, leaves to root.
    for i in (0..n).rev() {
        let link = &links[i];
        let k = link.joint.nv();
        let vo = model.v_offset(i);
        let ia = ws.ia[i];
        let c = ws.kin.c[i];
        if is_free_root(link) {
            // Solved directly in pass 3 (S = I₆, so D = I^A and U = I^A).
            continue;
        }
        let (ia_up, pa_up) = if k == 0 {
            (ia, ws.pa[i] + ia.mul_motion(c))
        } else if link.prescribed {
            let sq = ws.kin.joint_motion(link, i, &ws.qdd[vo..vo + k]);
            (ia, ws.pa[i] + ia.mul_motion(c + sq))
        } else {
            let mut d = [[0.0; 6]; 6];
            for col in 0..k {
                ws.u_cols[i][col] = ia.mul_motion(ws.kin.joint_col(link, i, col));
            }
            for r in 0..k {
                let s_r = ws.kin.joint_col(link, i, r);
                for col in 0..k {
                    d[r][col] = s_r.dot(ws.u_cols[i][col]);
                }
                ws.u[i][r] = tau[vo + r] - s_r.dot(ws.pa[i]);
            }
            if k == 1 {
                if d[0][0] <= 0.0 || !d[0][0].is_finite() {
                    return Err(DynamicsError::Singular(i));
                }
                ws.d_inv[i][0][0] = 1.0 / d[0][0];
            } else {
                ws.d_inv[i] = spd_inverse6(&d, k).map_err(|e| (i, e))?;
            }
            if link.parent.is_none() {
                continue;
            }
            // Ia = IA - U D⁻¹ Uᵀ ;  pa = pA + Ia c + U D⁻¹ u
            let mut ia_a = ia;
            let mut du = [0.0; 6];
            for r in 0..k {
                for col in 0..k {
                    let dinv = ws.d_inv[i][r][col];
                    du[r] += dinv * ws.u[i][col];
                    if dinv != 0.0 {
                        sub_scaled_outer(&mut ia_a, ws.u_cols[i][r], ws.u_cols[i][col], dinv);
                    }
                }
            }
            let mut pa = ws.pa[i] + ia_a.mul_motion(c);
            for r in 0..k {
                pa += ws.u_cols[i][r] * du[r];
            }
            (ia_a, pa)
        };
        if let Some(p) = link.parent {
            let x = ws.kin.x_up[i];
            ws.ia[p] += ia_up.transform_to_parent(&x);
            let f = x.inv_apply_force(pa_up);
            ws.pa[p] += f;
        }
    }

    // Pass 3: accelerations, root to leaves.
    let a0 = SpatialMotion::new(DVec3::ZERO, -gravity);
    for i in 0..n {
        let link = &links[i];
        let k = link.joint.nv();
        let vo = model.v_offset(i);
        let a_parent = match link.parent {
            Some(p) => ws.acc[p],
            None => ws.kin.x_world[i].apply_motion(a0),
        };
        let a_prime = match link.parent {
            Some(_) => ws.kin.x_up[i].apply_motion(a_parent),
            None => a_parent,
        } + ws.kin.c[i];
        let acc = if is_free_root(link) {
            // D q̈ = u - Uᵀ a'  ⇒  I^A (a' + q̈) = τ - p^A.
            let mut chol = ws.ia[i].to_mat6();
            cholesky6(&mut chol, 6).map_err(|e| (i, e))?;
            let mut rhs = [0.0; 6];
            let pa = ws.pa[i].to_array();
            for r in 0..6 {
                rhs[r] = tau[vo + r] - pa[r];
            }
            cholesky6_solve(&chol, 6, &mut rhs);
            let a = SpatialMotion::from_array(rhs);
            let qdd = a - a_prime;
            ws.qdd[vo..vo + 6].copy_from_slice(&qdd.to_array());
            a
        } else if k == 0 {
            a_prime
        } else if link.prescribed {
            let a = a_prime + ws.kin.joint_motion(link, i, &ws.qdd[vo..vo + k]);
            // Required joint force: τ = Sᵀ (I^A a + p^A).
            let f = ws.ia[i].mul_motion(a) + ws.pa[i];
            for r in 0..k {
                ws.tau_prescribed[vo + r] = ws.kin.joint_col(link, i, r).dot(f);
            }
            a
        } else {
            let mut rhs = [0.0; 6];
            for r in 0..k {
                rhs[r] = ws.u[i][r] - a_prime.dot(ws.u_cols[i][r]);
            }
            let mut qdd = [0.0; 6];
            for r in 0..k {
                qdd[r] = (0..k).map(|col| ws.d_inv[i][r][col] * rhs[col]).sum();
            }
            ws.qdd[vo..vo + k].copy_from_slice(&qdd[..k]);
            a_prime + ws.kin.joint_motion(link, i, &qdd[..k])
        };
        ws.acc[i] = acc;
    }
    Ok(())
}

/// Unprescribed free joint at the root: handled by a single 6×6 solve.
#[inline]
fn is_free_root(link: &super::Link) -> bool {
    link.parent.is_none() && !link.prescribed && link.joint == JointType::Free
}

/// `ia -= s · a bᵀ` for spatial force columns `a`, `b` (symmetric use: called for (r,c) and (c,r)).
#[inline]
fn sub_scaled_outer(ia: &mut ArticulatedInertia, a: SpatialForce, b: SpatialForce, s: f64) {
    ia.i -= outer(a.ang * s, b.ang);
    ia.h -= outer(a.ang * s, b.lin);
    ia.m -= outer(a.lin * s, b.lin);
}

/// Solve a free-floating single rigid body directly (used as a cross-check in tests).
#[doc(hidden)]
pub fn single_body_acceleration(
    ia: &ArticulatedInertia,
    rhs: SpatialForce,
) -> Result<SpatialMotion, NotPositiveDefinite> {
    let mut m = ia.to_mat6();
    cholesky6(&mut m, 6)?;
    let mut b = rhs.to_array();
    cholesky6_solve(&m, 6, &mut b);
    Ok(SpatialMotion::from_array(b))
}
