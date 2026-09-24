//! Forward kinematics (positions and velocities of all links).

use super::MultibodyModel;
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
        let xj = link.joint.transform(model.q_slice(i, q));
        let x_up = xj * link.x_tree;
        let vj = link.joint.motion(model.v_slice(i, v));
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
        kin.c[i] = vel.cross_motion(vj);
    }
}
