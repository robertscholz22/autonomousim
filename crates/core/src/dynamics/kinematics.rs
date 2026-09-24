//! Forward kinematics (positions and velocities of all links).

use super::{JointType, Link, MultibodyModel};
use crate::math::{Pose, SpatialMotion, Xform};

/// Per-link kinematic quantities for one configuration, reused across steps (no allocation
/// after construction).
#[derive(Clone, Debug, Default)]
pub struct KinCache {
    /// `iX_λ(i)`: parent link → link coordinates.
    pub x_up: Vec<Xform>,
    /// `iX_0`: world → link coordinates.
    pub x_world: Vec<Xform>,
    /// Link frame pose in the world.
    pub pose: Vec<Pose>,
    /// Link spatial velocity in link coordinates.
    pub vel: Vec<SpatialMotion>,
    /// Joint velocity `S q̇` in link coordinates.
    pub vj: Vec<SpatialMotion>,
    /// Velocity-product acceleration `c_i = c_J + v_i × S q̇`.
    pub c: Vec<SpatialMotion>,
    /// Subspace column of configuration-dependent joints at the current `q` (zero for others).
    pub s: Vec<SpatialMotion>,
}

impl KinCache {
    pub fn new(model: &MultibodyModel) -> Self {
        let n = model.num_links();
        Self {
            x_up: vec![Xform::IDENTITY; n],
            x_world: vec![Xform::IDENTITY; n],
            pose: vec![Pose::IDENTITY; n],
            vel: vec![SpatialMotion::ZERO; n],
            vj: vec![SpatialMotion::ZERO; n],
            c: vec![SpatialMotion::ZERO; n],
            s: vec![SpatialMotion::ZERO; n],
        }
    }

    /// Column `k` of the motion subspace of `link` (index `i`) at the current configuration.
    #[inline]
    pub fn joint_col(&self, link: &Link, i: usize, k: usize) -> SpatialMotion {
        match link.joint {
            JointType::KcTravel(_) => self.s[i],
            ref j => j.s_col(k),
        }
    }

    /// `S x` for joint coordinates `x` of `link` (index `i`) at the current configuration.
    #[inline]
    pub fn joint_motion(&self, link: &Link, i: usize, x: &[f64]) -> SpatialMotion {
        match link.joint {
            JointType::KcTravel(_) => self.s[i] * x[0],
            ref j => j.motion(x),
        }
    }

    /// Linear velocity of a point fixed in `link` (point given in link coordinates),
    /// expressed in world coordinates.
    #[inline]
    pub fn point_velocity_world(&self, link: usize, p_link: glam::DVec3) -> glam::DVec3 {
        self.pose[link].transform_vector(self.vel[link].point_velocity(p_link))
    }

    /// Angular velocity of `link` in world coordinates.
    #[inline]
    pub fn angular_velocity_world(&self, link: usize) -> glam::DVec3 {
        self.pose[link].transform_vector(self.vel[link].ang)
    }
}

/// Compute link transforms, poses and velocities for `(q, v)`.
pub fn forward_kinematics(model: &MultibodyModel, q: &[f64], v: &[f64], kin: &mut KinCache) {
    debug_assert_eq!(q.len(), model.nq());
    debug_assert_eq!(v.len(), model.nv());
    for (i, link) in model.links().iter().enumerate() {
        let (qi, vi) = (model.q_slice(i, q), model.v_slice(i, v));
        let (xj, vj, cj) = match &link.joint {
            JointType::KcTravel(t) => {
                let p = t.eval(qi[0]);
                kin.s[i] = p.s;
                (Xform::from_pose(p.position, p.rotation), p.s * vi[0], Some(p.ds * (vi[0] * vi[0])))
            }
            j => (j.transform(qi), j.motion(vi), None),
        };
        let x_up = xj * link.x_tree;
        let (x_world, vel) = match link.parent {
            Some(p) => (x_up * kin.x_world[p], x_up.apply_motion(kin.vel[p]) + vj),
            None => (x_up, vj),
        };
        kin.x_up[i] = x_up;
        kin.x_world[i] = x_world;
        let (pos, rot) = x_world.to_pose();
        kin.pose[i] = Pose::new(pos, rot);
        kin.vel[i] = vel;
        kin.vj[i] = vj;
        kin.c[i] = match cj {
            Some(cj) => vel.cross_motion(vj) + cj,
            None => vel.cross_motion(vj),
        };
    }
}
