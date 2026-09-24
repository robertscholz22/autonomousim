//! Recursive Newton–Euler inverse dynamics and Composite-Rigid-Body mass matrix.

use super::{KinCache, MultibodyModel, forward_kinematics};
use crate::math::{DenseMatrix, SpatialForce, SpatialMotion};
use glam::DVec3;

/// Inverse dynamics `τ = M(q) q̈ + C(q, v) - Jᵀ f_ext` including gravity
/// (Featherstone Table 5.1). `f_ext` may be empty.
pub fn rnea(
    model: &MultibodyModel,
    q: &[f64],
    v: &[f64],
    qdd: &[f64],
    f_ext: &[SpatialForce],
    gravity: DVec3,
    kin: &mut KinCache,
) -> Vec<f64> {
    forward_kinematics(model, q, v, kin);
    let links = model.links();
    let n = links.len();
    let mut acc = vec![SpatialMotion::ZERO; n];
    let mut f = vec![SpatialForce::ZERO; n];
    let a0 = SpatialMotion::new(DVec3::ZERO, -gravity);
    for i in 0..n {
        let link = &links[i];
        let a_parent = match link.parent {
            Some(p) => kin.x_up[i].apply_motion(acc[p]),
            None => kin.x_up[i].apply_motion(a0),
        };
        acc[i] = a_parent + kin.joint_motion(link, i, model.v_slice(i, qdd)) + kin.c[i];
        let inertia = &link.inertia;
        f[i] = inertia.mul_motion(acc[i]) + kin.vel[i].cross_force(inertia.mul_motion(kin.vel[i]));
        if !f_ext.is_empty() {
            f[i] -= f_ext[i];
        }
    }
    let mut tau = vec![0.0; model.nv()];
    for i in (0..n).rev() {
        let link = &links[i];
        let vo = model.v_offset(i);
        for r in 0..link.joint.nv() {
            tau[vo + r] = kin.joint_col(link, i, r).dot(f[i]);
        }
        if let Some(p) = link.parent {
            let fp = kin.x_up[i].inv_apply_force(f[i]);
            f[p] += fp;
        }
    }
    tau
}

/// Joint-space inertia matrix `M(q)` by the Composite-Rigid-Body Algorithm (Table 6.2).
/// Uses the transforms in `kin`, which must be up to date for `q`.
pub fn crba(model: &MultibodyModel, kin: &KinCache) -> DenseMatrix {
    let links = model.links();
    let n = links.len();
    let nv = model.nv();
    let mut ic: Vec<_> = links.iter().map(|l| l.inertia.to_articulated()).collect();
    let mut m = DenseMatrix::zeros(nv, nv);
    for i in (0..n).rev() {
        if let Some(p) = links[i].parent {
            let up = ic[i].transform_to_parent(&kin.x_up[i]);
            ic[p] += up;
        }
        let ki = links[i].joint.nv();
        let vi = model.v_offset(i);
        for a in 0..ki {
            // F = I^c_i S_i[:, a], carried up the tree.
            let mut fcol = ic[i].mul_motion(kin.joint_col(&links[i], i, a));
            for b in 0..ki {
                m[(vi + a, vi + b)] = kin.joint_col(&links[i], i, b).dot(fcol);
            }
            let mut j = i;
            while let Some(p) = links[j].parent {
                fcol = kin.x_up[j].inv_apply_force(fcol);
                j = p;
                let kj = links[j].joint.nv();
                let vj = model.v_offset(j);
                for b in 0..kj {
                    let val = kin.joint_col(&links[j], j, b).dot(fcol);
                    m[(vi + a, vj + b)] = val;
                    m[(vj + b, vi + a)] = val;
                }
            }
        }
    }
    m
}
