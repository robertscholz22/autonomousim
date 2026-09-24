//! Joint models.
//!
//! A joint connects a predecessor frame (fixed in the parent body, placed by the tree
//! transform `X_T`) to the successor frame (the child body frame). `X_J(q)` maps
//! predecessor to successor coordinates and the motion subspace `S` is expressed in child
//! coordinates. For all joint types but [`JointType::KcTravel`] `S` is constant in child
//! coordinates, so the joint bias acceleration `c_J = Ṡ q̇` is zero; for `KcTravel`, forward
//! kinematics stores `S(q)` and `c_J` in the [`KinCache`](super::KinCache), and the dynamics
//! algorithms take the subspace from there ([`KinCache::joint_col`](super::KinCache::joint_col)).

use super::kc::KcTable;
use crate::math::{SpatialMotion, Xform, quat};
use glam::{DQuat, DVec3};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Kinematic joint type.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum JointType {
    /// 6-DoF floating joint. `q = [px, py, pz, qx, qy, qz, qw]` (position in the parent frame,
    /// orientation child→parent); `v = [ωx, ωy, ωz, vx, vy, vz]` in child coordinates.
    Free,
    /// Rotation about a unit `axis` (predecessor = child coordinates along the axis).
    Revolute { axis: DVec3 },
    /// Translation along a unit `axis`.
    Prismatic { axis: DVec3 },
    /// 3-DoF ball joint. `q = [qx, qy, qz, qw]`, `v = ω` in child coordinates.
    Spherical,
    /// Rigid attachment (no DoF).
    Fixed,
    /// Suspension travel: one DoF moving the child along the tabulated curves of a
    /// [`KcTable`] (`S` depends on `q`).
    KcTravel(Arc<KcTable>),
}

impl JointType {
    pub fn revolute(axis: DVec3) -> Self {
        JointType::Revolute { axis: axis.normalize() }
    }

    pub fn prismatic(axis: DVec3) -> Self {
        JointType::Prismatic { axis: axis.normalize() }
    }

    /// Number of position coordinates.
    #[inline]
    pub fn nq(&self) -> usize {
        match self {
            JointType::Free => 7,
            JointType::Revolute { .. } | JointType::Prismatic { .. } | JointType::KcTravel(_) => 1,
            JointType::Spherical => 4,
            JointType::Fixed => 0,
        }
    }

    /// Number of velocity coordinates (degrees of freedom).
    #[inline]
    pub fn nv(&self) -> usize {
        match self {
            JointType::Free => 6,
            JointType::Revolute { .. } | JointType::Prismatic { .. } | JointType::KcTravel(_) => 1,
            JointType::Spherical => 3,
            JointType::Fixed => 0,
        }
    }

    /// Neutral configuration (zero displacement, identity rotation).
    pub fn neutral_q(&self, q: &mut [f64]) {
        match self {
            JointType::Free => q.copy_from_slice(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            JointType::Spherical => q.copy_from_slice(&[0.0, 0.0, 0.0, 1.0]),
            JointType::Revolute { .. } | JointType::Prismatic { .. } | JointType::KcTravel(_) => q[0] = 0.0,
            JointType::Fixed => {}
        }
    }

    /// Joint transform `X_J(q)` (predecessor → successor coordinates).
    #[inline]
    pub fn transform(&self, q: &[f64]) -> Xform {
        match self {
            JointType::Free => Xform::from_pose(DVec3::new(q[0], q[1], q[2]), quat_at(q, 3)),
            JointType::Revolute { axis } => Xform::rotation(DQuat::from_axis_angle(*axis, q[0])),
            JointType::Prismatic { axis } => Xform::translation(*axis * q[0]),
            JointType::Spherical => Xform::rotation(quat_at(q, 0)),
            JointType::Fixed => Xform::IDENTITY,
            JointType::KcTravel(t) => t.transform(q[0]),
        }
    }

    /// Whether `S` depends on the configuration (then use [`subspace_at`](Self::subspace_at)).
    #[inline]
    pub fn configuration_dependent(&self) -> bool {
        matches!(self, JointType::KcTravel(_))
    }

    /// Column `k` of the motion subspace `S` (child coordinates) of a joint whose subspace is
    /// constant.
    ///
    /// # Panics
    /// For configuration-dependent joints.
    #[inline]
    pub fn s_col(&self, k: usize) -> SpatialMotion {
        match self {
            JointType::Free => {
                let mut a = [0.0; 6];
                a[k] = 1.0;
                SpatialMotion::from_array(a)
            }
            JointType::Revolute { axis } => SpatialMotion::new(*axis, DVec3::ZERO),
            JointType::Prismatic { axis } => SpatialMotion::new(DVec3::ZERO, *axis),
            JointType::Spherical => {
                let mut a = DVec3::ZERO;
                a[k] = 1.0;
                SpatialMotion::new(a, DVec3::ZERO)
            }
            JointType::Fixed => unreachable!("fixed joints have no motion subspace"),
            JointType::KcTravel(_) => panic!("the subspace of a KcTravel joint depends on q"),
        }
    }

    /// The single subspace column of a one-DoF joint at `q`, and its derivative with respect to
    /// `q` (zero for constant subspaces).
    #[inline]
    pub fn subspace_at(&self, q: &[f64]) -> (SpatialMotion, SpatialMotion) {
        match self {
            JointType::KcTravel(t) => {
                let p = t.eval(q[0]);
                (p.s, p.ds)
            }
            j => {
                debug_assert_eq!(j.nv(), 1, "subspace_at is for one-DoF joints");
                (j.s_col(0), SpatialMotion::ZERO)
            }
        }
    }

    /// Joint velocity `S q̇` (child coordinates) of a joint with a constant subspace.
    #[inline]
    pub fn motion(&self, v: &[f64]) -> SpatialMotion {
        match self {
            JointType::Free => SpatialMotion::new(DVec3::new(v[0], v[1], v[2]), DVec3::new(v[3], v[4], v[5])),
            JointType::Revolute { axis } => SpatialMotion::new(*axis * v[0], DVec3::ZERO),
            JointType::Prismatic { axis } => SpatialMotion::new(DVec3::ZERO, *axis * v[0]),
            JointType::Spherical => SpatialMotion::new(DVec3::new(v[0], v[1], v[2]), DVec3::ZERO),
            JointType::Fixed => SpatialMotion::ZERO,
            JointType::KcTravel(_) => panic!("the subspace of a KcTravel joint depends on q"),
        }
    }

    /// Joint velocity `S(q) q̇` for any joint.
    #[inline]
    pub fn motion_at(&self, q: &[f64], v: &[f64]) -> SpatialMotion {
        match self {
            JointType::KcTravel(t) => t.eval(q[0]).s * v[0],
            j => j.motion(v),
        }
    }

    /// Semi-implicit position update `q ← q ⊕ v·dt` on the joint's configuration manifold.
    pub fn integrate(&self, q: &mut [f64], v: &[f64], dt: f64) {
        match self {
            JointType::Free => {
                let rot = quat_at(q, 3);
                let p = DVec3::new(q[0], q[1], q[2]) + rot * DVec3::new(v[3], v[4], v[5]) * dt;
                let rot = quat::integrate_body_rate(rot, DVec3::new(v[0], v[1], v[2]), dt);
                q[..3].copy_from_slice(&p.to_array());
                set_quat(q, 3, rot);
            }
            JointType::Revolute { .. } | JointType::Prismatic { .. } | JointType::KcTravel(_) => q[0] += v[0] * dt,
            JointType::Spherical => {
                let rot = quat::integrate_body_rate(quat_at(q, 0), DVec3::new(v[0], v[1], v[2]), dt);
                set_quat(q, 0, rot);
            }
            JointType::Fixed => {}
        }
    }

    /// Time derivative of the raw position coordinates, `dq/dt` (length `nq`), used by RK4.
    pub fn qdot(&self, q: &[f64], v: &[f64], out: &mut [f64]) {
        match self {
            JointType::Free => {
                let rot = quat_at(q, 3);
                let pd = rot * DVec3::new(v[3], v[4], v[5]);
                out[..3].copy_from_slice(&pd.to_array());
                let qd = quat_derivative(rot, DVec3::new(v[0], v[1], v[2]));
                out[3..7].copy_from_slice(&qd);
            }
            JointType::Revolute { .. } | JointType::Prismatic { .. } | JointType::KcTravel(_) => out[0] = v[0],
            JointType::Spherical => {
                let qd = quat_derivative(quat_at(q, 0), DVec3::new(v[0], v[1], v[2]));
                out[..4].copy_from_slice(&qd);
            }
            JointType::Fixed => {}
        }
    }

    /// Re-normalise quaternion coordinates after an unconstrained update.
    pub fn normalize(&self, q: &mut [f64]) {
        match self {
            JointType::Free => set_quat(q, 3, quat_at(q, 3).normalize()),
            JointType::Spherical => set_quat(q, 0, quat_at(q, 0).normalize()),
            _ => {}
        }
    }
}

#[inline]
fn quat_at(q: &[f64], i: usize) -> DQuat {
    DQuat::from_xyzw(q[i], q[i + 1], q[i + 2], q[i + 3])
}

#[inline]
fn set_quat(q: &mut [f64], i: usize, r: DQuat) {
    q[i..i + 4].copy_from_slice(&[r.x, r.y, r.z, r.w]);
}

/// `dq/dt = ½ q ⊗ (ω, 0)` for a body-frame angular velocity, as `[x, y, z, w]`.
#[inline]
fn quat_derivative(q: DQuat, w: DVec3) -> [f64; 4] {
    let d = q * DQuat::from_xyzw(w.x, w.y, w.z, 0.0);
    [0.5 * d.x, 0.5 * d.y, 0.5 * d.z, 0.5 * d.w]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimensions() {
        assert_eq!((JointType::Free.nq(), JointType::Free.nv()), (7, 6));
        assert_eq!((JointType::Spherical.nq(), JointType::Spherical.nv()), (4, 3));
        assert_eq!((JointType::Fixed.nq(), JointType::Fixed.nv()), (0, 0));
    }

    /// The motion subspace must be consistent with the derivative of the joint transform:
    /// integrating the joint for a small step moves the child frame by `S q̇ dt`.
    #[test]
    fn subspace_matches_transform_derivative() {
        let joints = [
            JointType::revolute(DVec3::new(0.3, -0.5, 0.8)),
            JointType::prismatic(DVec3::new(-0.2, 0.1, 0.9)),
            JointType::Spherical,
            JointType::Free,
            JointType::KcTravel(Arc::new(crate::dynamics::kc::tests::curved())),
        ];
        for j in joints {
            let mut q = vec![0.0; j.nq()];
            j.neutral_q(&mut q);
            // Start from a non-trivial configuration.
            let v0: Vec<f64> = (0..j.nv()).map(|k| 0.3 + 0.1 * k as f64).collect();
            j.integrate(&mut q, &v0, 1.0);
            let v: Vec<f64> = (0..j.nv()).map(|k| 0.7 - 0.2 * k as f64).collect();
            let x0 = j.transform(&q);
            let h = 1e-6;
            let mut q1 = q.clone();
            j.integrate(&mut q1, &v, h);
            let x1 = j.transform(&q1);
            // rel = C1_X_C0: E ≈ 1 - h[ω]×, and its `pos` is C1's origin in C0 coordinates ≈ h v_lin.
            let rel = x1 * x0.inverse();
            let rot_err = rel.rot.transpose() - glam::DMat3::IDENTITY; // ≈ h [ω]×
            let omega = DVec3::new(rot_err.y_axis.z, rot_err.z_axis.x, rot_err.x_axis.y) / h;
            let lin_succ = rel.pos / h;
            let sv = j.motion_at(&q, &v);
            assert!((omega - sv.ang).length() < 1e-5, "{j:?}: {omega:?} vs {:?}", sv.ang);
            assert!((lin_succ - sv.lin).length() < 1e-5, "{j:?}: {lin_succ:?} vs {:?}", sv.lin);
        }
    }
}
