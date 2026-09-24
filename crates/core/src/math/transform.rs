//! Plücker coordinate transforms and rigid poses.

use super::{SpatialForce, SpatialMotion};
use glam::{DMat3, DQuat, DVec3};
use serde::{Deserialize, Serialize};
use std::ops::Mul;

/// Plücker transform `B_X_A` from frame A coordinates to frame B coordinates.
///
/// Stored as Featherstone's `(E, r)`: `E` rotates A coordinates into B coordinates and `r` is the
/// position of B's origin expressed in A coordinates. As a 6×6 motion transform it is
/// `[E, 0; -E r×, E]`; it is never materialised as a matrix.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Xform {
    pub rot: DMat3,
    pub pos: DVec3,
}

impl Default for Xform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Xform {
    pub const IDENTITY: Self = Self { rot: DMat3::IDENTITY, pos: DVec3::ZERO };

    #[inline]
    pub const fn new(rot: DMat3, pos: DVec3) -> Self {
        Self { rot, pos }
    }

    /// Pure translation: B's origin sits at `r` in A, axes parallel.
    #[inline]
    pub fn translation(r: DVec3) -> Self {
        Self { rot: DMat3::IDENTITY, pos: r }
    }

    /// Pure rotation where B's axes are A's axes rotated by `q` (active rotation), so that
    /// coordinates transform with `E = R(q)ᵀ`.
    #[inline]
    pub fn rotation(q: DQuat) -> Self {
        Self { rot: DMat3::from_quat(q).transpose(), pos: DVec3::ZERO }
    }

    /// Transform from a parent frame A to a child frame B given B's pose in A
    /// (origin `p` and orientation `q`, both expressed in A).
    #[inline]
    pub fn from_pose(p: DVec3, q: DQuat) -> Self {
        Self { rot: DMat3::from_quat(q).transpose(), pos: p }
    }

    /// Pose of B in A as `(origin, orientation)`; inverse of [`Xform::from_pose`].
    #[inline]
    pub fn to_pose(&self) -> (DVec3, DQuat) {
        (self.pos, DQuat::from_mat3(&self.rot.transpose()).normalize())
    }

    /// `A_X_B`.
    #[inline]
    pub fn inverse(&self) -> Self {
        Self { rot: self.rot.transpose(), pos: -(self.rot * self.pos) }
    }

    /// Transform a motion vector from A to B coordinates.
    #[inline]
    pub fn apply_motion(&self, m: SpatialMotion) -> SpatialMotion {
        SpatialMotion { ang: self.rot * m.ang, lin: self.rot * (m.lin - self.pos.cross(m.ang)) }
    }

    /// Transform a motion vector from B back to A coordinates.
    #[inline]
    pub fn inv_apply_motion(&self, m: SpatialMotion) -> SpatialMotion {
        let ang = self.rot.transpose() * m.ang;
        SpatialMotion { ang, lin: self.rot.transpose() * m.lin + self.pos.cross(ang) }
    }

    /// Transform a force vector from A to B coordinates (`X*`).
    #[inline]
    pub fn apply_force(&self, f: SpatialForce) -> SpatialForce {
        SpatialForce { ang: self.rot * (f.ang - self.pos.cross(f.lin)), lin: self.rot * f.lin }
    }

    /// Transform a force vector from B back to A coordinates (`Xᵀ`, i.e. `A_X*_B`).
    #[inline]
    pub fn inv_apply_force(&self, f: SpatialForce) -> SpatialForce {
        let lin = self.rot.transpose() * f.lin;
        SpatialForce { ang: self.rot.transpose() * f.ang + self.pos.cross(lin), lin }
    }

    /// Map a point given in A coordinates to B coordinates.
    #[inline]
    pub fn apply_point(&self, p: DVec3) -> DVec3 {
        self.rot * (p - self.pos)
    }

    /// Map a point given in B coordinates to A coordinates.
    #[inline]
    pub fn inv_apply_point(&self, p: DVec3) -> DVec3 {
        self.rot.transpose() * p + self.pos
    }

    #[inline]
    pub fn is_finite(&self) -> bool {
        self.rot.is_finite() && self.pos.is_finite()
    }
}

/// Composition: `(C_X_B) * (B_X_A) = C_X_A`.
impl Mul for Xform {
    type Output = Xform;
    #[inline]
    fn mul(self, rhs: Xform) -> Xform {
        Xform { rot: self.rot * rhs.rot, pos: rhs.pos + rhs.rot.transpose() * self.pos }
    }
}

/// A rigid pose (position + orientation), e.g. of a body in the world frame.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Pose {
    pub pos: DVec3,
    /// Rotation from the local frame to the parent (world) frame.
    pub rot: DQuat,
}

impl Default for Pose {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Pose {
    pub const IDENTITY: Self = Self { pos: DVec3::ZERO, rot: DQuat::IDENTITY };

    #[inline]
    pub const fn new(pos: DVec3, rot: DQuat) -> Self {
        Self { pos, rot }
    }

    #[inline]
    pub fn from_translation(pos: DVec3) -> Self {
        Self { pos, rot: DQuat::IDENTITY }
    }

    /// Local point → parent coordinates.
    #[inline]
    pub fn transform_point(&self, p: DVec3) -> DVec3 {
        self.pos + self.rot * p
    }

    /// Local vector → parent coordinates.
    #[inline]
    pub fn transform_vector(&self, v: DVec3) -> DVec3 {
        self.rot * v
    }

    /// Parent point → local coordinates.
    #[inline]
    pub fn inverse_transform_point(&self, p: DVec3) -> DVec3 {
        self.rot.inverse() * (p - self.pos)
    }

    /// Parent vector → local coordinates.
    #[inline]
    pub fn inverse_transform_vector(&self, v: DVec3) -> DVec3 {
        self.rot.inverse() * v
    }

    #[inline]
    pub fn inverse(&self) -> Self {
        let inv = self.rot.inverse();
        Self { pos: -(inv * self.pos), rot: inv }
    }

    /// Plücker transform from the parent frame to this local frame.
    #[inline]
    pub fn to_xform(&self) -> Xform {
        Xform::from_pose(self.pos, self.rot)
    }
}

/// `a * b`: pose `b` (given in `a`'s local frame) expressed in `a`'s parent frame.
impl Mul for Pose {
    type Output = Pose;
    #[inline]
    fn mul(self, b: Pose) -> Pose {
        Pose { pos: self.transform_point(b.pos), rot: (self.rot * b.rot).normalize() }
    }
}

impl Pose {
    /// The same pose as a parry isometry.
    #[inline]
    pub fn to_parry(&self) -> parry3d_f64::math::Pose {
        parry3d_f64::math::Pose::from_parts(self.pos, self.rot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::testutil::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn inverse_roundtrip(x in xform(), m in motion(), f in force()) {
            let xi = x.inverse();
            assert_motion_close(xi.apply_motion(x.apply_motion(m)), m, 1e-9);
            assert_motion_close(x.inv_apply_motion(x.apply_motion(m)), m, 1e-9);
            assert_force_close(xi.apply_force(x.apply_force(f)), f, 1e-9);
            assert_force_close(x.inv_apply_force(x.apply_force(f)), f, 1e-9);
            assert_xform_close(x * xi, Xform::IDENTITY, 1e-9);
        }

        /// Power is frame invariant: (X m)·(X* f) = m·f.
        #[test]
        fn power_invariance(x in xform(), m in motion(), f in force()) {
            let lhs = x.apply_motion(m).dot(x.apply_force(f));
            let rhs = m.dot(f);
            prop_assert!((lhs - rhs).abs() <= 1e-8 * (1.0 + rhs.abs()));
        }

        /// Composition matches sequential application.
        #[test]
        fn composition(x1 in xform(), x2 in xform(), m in motion(), f in force(), p in vec3()) {
            let c = x2 * x1;
            assert_motion_close(c.apply_motion(m), x2.apply_motion(x1.apply_motion(m)), 1e-8);
            assert_force_close(c.apply_force(f), x2.apply_force(x1.apply_force(f)), 1e-8);
            assert_vec_close(c.apply_point(p), x2.apply_point(x1.apply_point(p)), 1e-8);
        }

        /// X (m1 × m2) = (X m1) × (X m2) and likewise for ×*.
        #[test]
        fn cross_products_are_covariant(x in xform(), m1 in motion(), m2 in motion(), f in force()) {
            assert_motion_close(x.apply_motion(m1.cross_motion(m2)), x.apply_motion(m1).cross_motion(x.apply_motion(m2)), 1e-7);
            assert_force_close(x.apply_force(m1.cross_force(f)), x.apply_motion(m1).cross_force(x.apply_force(f)), 1e-7);
        }

        /// A rigid body's point velocities agree after a change of frame.
        #[test]
        fn point_velocity_consistency(pose in pose(), m in motion(), p in vec3()) {
            // `m` is expressed in world (A); B is the local frame at `pose`.
            let x = pose.to_xform();
            let p_local = pose.inverse_transform_point(p);
            let v_world = m.point_velocity(p);
            let v_local = x.apply_motion(m).point_velocity(p_local);
            assert_vec_close(pose.transform_vector(v_local), v_world, 1e-8);
        }

        #[test]
        fn pose_xform_roundtrip(pose in pose(), p in vec3()) {
            let x = pose.to_xform();
            assert_vec_close(x.apply_point(p), pose.inverse_transform_point(p), 1e-9);
            let (pos, rot) = x.to_pose();
            assert_vec_close(pos, pose.pos, 1e-12);
            prop_assert!(rot.dot(pose.rot).abs() > 1.0 - 1e-12);
            let pi = pose.inverse();
            assert_vec_close((pose * pi).pos, DVec3::ZERO, 1e-9);
        }
    }
}
