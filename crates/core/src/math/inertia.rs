//! Rigid-body and articulated-body spatial inertias.

use super::{SpatialForce, SpatialMotion, Xform, skew};
use glam::{DMat3, DVec3};
use serde::{Deserialize, Serialize};
use std::ops::{Add, AddAssign, Sub};

/// Rigid-body inertia: mass, centre of mass and rotational inertia about the COM, all in the
/// body frame.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RigidInertia {
    pub mass: f64,
    pub com: DVec3,
    /// Rotational inertia about the COM (symmetric positive definite for a physical body).
    pub i_com: DMat3,
}

impl Default for RigidInertia {
    fn default() -> Self {
        Self::ZERO
    }
}

impl RigidInertia {
    pub const ZERO: Self = Self { mass: 0.0, com: DVec3::ZERO, i_com: DMat3::ZERO };

    pub fn new(mass: f64, com: DVec3, i_com: DMat3) -> Self {
        Self { mass, com, i_com }
    }

    /// Point mass at `com`.
    pub fn point(mass: f64, com: DVec3) -> Self {
        Self { mass, com, i_com: DMat3::ZERO }
    }

    /// Solid cuboid with full side lengths `size`, centred at the origin.
    pub fn cuboid(mass: f64, size: DVec3) -> Self {
        let s2 = size * size;
        let k = mass / 12.0;
        Self::diag(mass, DVec3::new(k * (s2.y + s2.z), k * (s2.x + s2.z), k * (s2.x + s2.y)))
    }

    /// Solid sphere of radius `r` centred at the origin.
    pub fn sphere(mass: f64, r: f64) -> Self {
        Self::diag(mass, DVec3::splat(0.4 * mass * r * r))
    }

    /// Solid cylinder of radius `r` and length `h` along the local `axis` (0 = x, 1 = y, 2 = z).
    pub fn cylinder(mass: f64, r: f64, h: f64, axis: usize) -> Self {
        let axial = 0.5 * mass * r * r;
        let trans = mass * (3.0 * r * r + h * h) / 12.0;
        let mut d = DVec3::splat(trans);
        d[axis] = axial;
        Self::diag(mass, d)
    }

    /// Principal moments about the COM at the origin.
    pub fn diag(mass: f64, moments: DVec3) -> Self {
        Self { mass, com: DVec3::ZERO, i_com: DMat3::from_diagonal(moments) }
    }

    /// Same body with its COM moved to `com`.
    pub fn with_com(mut self, com: DVec3) -> Self {
        self.com = com;
        self
    }

    /// Rotational inertia about the frame origin: `I_c + m (cᵀc 1 - c cᵀ)`.
    pub fn i_origin(&self) -> DMat3 {
        let c = skew(self.com);
        self.i_com - self.mass * c * c
    }

    /// `I v` for a spatial motion `v` (momentum, or `I a` for an acceleration).
    #[inline]
    pub fn mul_motion(&self, v: SpatialMotion) -> SpatialForce {
        // Velocity of the COM point.
        let v_c = v.lin - self.com.cross(v.ang);
        let lin = self.mass * v_c;
        SpatialForce { ang: self.i_com * v.ang + self.com.cross(lin), lin }
    }

    /// Kinetic energy `½ vᵀ I v`.
    #[inline]
    pub fn kinetic_energy(&self, v: SpatialMotion) -> f64 {
        0.5 * v.dot(self.mul_motion(v))
    }

    /// Express this inertia (given in frame B) in frame A, where `x = B_X_A`.
    pub fn transform_to_parent(&self, x: &Xform) -> RigidInertia {
        let et = x.rot.transpose();
        RigidInertia { mass: self.mass, com: x.pos + et * self.com, i_com: et * self.i_com * x.rot }
    }

    /// Combined inertia of two bodies expressed in the same frame.
    pub fn combine(&self, o: &RigidInertia) -> RigidInertia {
        let m = self.mass + o.mass;
        if m <= 0.0 {
            return RigidInertia::ZERO;
        }
        let com = (self.mass * self.com + o.mass * o.com) / m;
        let shift = |b: &RigidInertia| {
            let d = skew(b.com - com);
            b.i_com - b.mass * d * d
        };
        RigidInertia { mass: m, com, i_com: shift(self) + shift(o) }
    }

    pub fn to_articulated(&self) -> ArticulatedInertia {
        let c = skew(self.com);
        let m = self.mass;
        ArticulatedInertia { i: self.i_com - m * c * c, h: m * c, m: DMat3::from_diagonal(DVec3::splat(m)) }
    }
}

/// General symmetric spatial inertia `[I, H; Hᵀ, M]` (articulated-body inertia).
///
/// `I` and `M` are symmetric. Multiplying a motion `(ω, v)` gives the force
/// `(I ω + H v, Hᵀ ω + M v)`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ArticulatedInertia {
    pub i: DMat3,
    pub h: DMat3,
    pub m: DMat3,
}

impl Default for ArticulatedInertia {
    fn default() -> Self {
        Self::ZERO
    }
}

impl ArticulatedInertia {
    pub const ZERO: Self = Self { i: DMat3::ZERO, h: DMat3::ZERO, m: DMat3::ZERO };

    #[inline]
    pub fn mul_motion(&self, v: SpatialMotion) -> SpatialForce {
        SpatialForce { ang: self.i * v.ang + self.h * v.lin, lin: self.h.transpose() * v.ang + self.m * v.lin }
    }

    /// Rank-one downdate `self - u uᵀ / d` used by the ABA for single-DoF joints.
    #[inline]
    pub fn sub_outer(&mut self, u: SpatialForce, inv_d: f64) {
        let ua = u.ang * inv_d;
        let ul = u.lin * inv_d;
        self.i -= outer(ua, u.ang);
        self.h -= outer(ua, u.lin);
        self.m -= outer(ul, u.lin);
    }

    /// Express this inertia (given in frame B) in frame A, where `x = B_X_A`: `Xᵀ I X`.
    pub fn transform_to_parent(&self, x: &Xform) -> ArticulatedInertia {
        let e = x.rot;
        let et = e.transpose();
        let i = et * self.i * e;
        let h = et * self.h * e;
        let m = et * self.m * e;
        let r = skew(x.pos);
        let rm = r * m;
        ArticulatedInertia { i: i - h * r + r * h.transpose() - rm * r, h: h + rm, m }
    }

    /// Dense row-major 6×6 matrix.
    pub fn to_mat6(&self) -> [[f64; 6]; 6] {
        let mut a = [[0.0; 6]; 6];
        let ht = self.h.transpose();
        for r in 0..3 {
            for c in 0..3 {
                a[r][c] = self.i.col(c)[r];
                a[r][c + 3] = self.h.col(c)[r];
                a[r + 3][c] = ht.col(c)[r];
                a[r + 3][c + 3] = self.m.col(c)[r];
            }
        }
        a
    }

    /// From a dense row-major 6×6 matrix (assumed symmetric).
    pub fn from_mat6(a: &[[f64; 6]; 6]) -> Self {
        let blk = |r0: usize, c0: usize| {
            DMat3::from_cols(
                DVec3::new(a[r0][c0], a[r0 + 1][c0], a[r0 + 2][c0]),
                DVec3::new(a[r0][c0 + 1], a[r0 + 1][c0 + 1], a[r0 + 2][c0 + 1]),
                DVec3::new(a[r0][c0 + 2], a[r0 + 1][c0 + 2], a[r0 + 2][c0 + 2]),
            )
        };
        Self { i: blk(0, 0), h: blk(0, 3), m: blk(3, 3) }
    }

    pub fn is_finite(&self) -> bool {
        self.i.is_finite() && self.h.is_finite() && self.m.is_finite()
    }
}

impl From<RigidInertia> for ArticulatedInertia {
    fn from(r: RigidInertia) -> Self {
        r.to_articulated()
    }
}

impl Add for ArticulatedInertia {
    type Output = Self;
    #[inline]
    fn add(self, o: Self) -> Self {
        Self { i: self.i + o.i, h: self.h + o.h, m: self.m + o.m }
    }
}

impl AddAssign for ArticulatedInertia {
    #[inline]
    fn add_assign(&mut self, o: Self) {
        self.i += o.i;
        self.h += o.h;
        self.m += o.m;
    }
}

impl Sub for ArticulatedInertia {
    type Output = Self;
    #[inline]
    fn sub(self, o: Self) -> Self {
        Self { i: self.i - o.i, h: self.h - o.h, m: self.m - o.m }
    }
}

/// Outer product `a bᵀ`.
#[inline]
pub fn outer(a: DVec3, b: DVec3) -> DMat3 {
    DMat3::from_cols(a * b.x, a * b.y, a * b.z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::testutil::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn rigid_and_articulated_agree(ri in rigid_inertia(), v in motion()) {
            let a = ri.to_articulated();
            assert_force_close(a.mul_motion(v), ri.mul_motion(v), 1e-9);
        }

        /// Transforming an inertia is consistent with transforming motion and force:
        /// I_A v_A = Xᵀ (I_B (X v_A)).
        #[test]
        fn transform_consistency(ri in rigid_inertia(), x in xform(), v in motion()) {
            let a = ri.to_articulated();
            let expected = x.inv_apply_force(a.mul_motion(x.apply_motion(v)));
            assert_force_close(a.transform_to_parent(&x).mul_motion(v), expected, 1e-8);
            assert_force_close(ri.transform_to_parent(&x).mul_motion(v), expected, 1e-8);
        }

        /// Kinetic energy is frame invariant.
        #[test]
        fn kinetic_energy_invariance(ri in rigid_inertia(), x in xform(), v in motion()) {
            let e_b = ri.kinetic_energy(x.apply_motion(v));
            let e_a = ri.transform_to_parent(&x).kinetic_energy(v);
            prop_assert!((e_a - e_b).abs() <= 1e-8 * (1.0 + e_a.abs()));
        }

        #[test]
        fn mat6_roundtrip_is_symmetric(ri in rigid_inertia(), x in xform()) {
            let a = ri.to_articulated().transform_to_parent(&x);
            let m = a.to_mat6();
            for r in 0..6 { for c in 0..6 {
                prop_assert!((m[r][c] - m[c][r]).abs() <= 1e-9 * (1.0 + m[r][c].abs()));
            }}
            let b = ArticulatedInertia::from_mat6(&m);
            prop_assert!(crate::math::linalg::mat3_max_abs(b.i - a.i) < 1e-12);
            prop_assert!(crate::math::linalg::mat3_max_abs(b.h - a.h) < 1e-12);
            prop_assert!(crate::math::linalg::mat3_max_abs(b.m - a.m) < 1e-12);
        }

        #[test]
        fn combine_matches_sum(r1 in rigid_inertia(), r2 in rigid_inertia(), v in motion()) {
            let c = r1.combine(&r2);
            let expected = r1.mul_motion(v) + r2.mul_motion(v);
            assert_force_close(c.mul_motion(v), expected, 1e-8);
        }
    }

    #[test]
    fn primitive_inertias() {
        let c = RigidInertia::cuboid(12.0, DVec3::new(1.0, 2.0, 3.0));
        assert!((c.i_com.x_axis.x - 13.0).abs() < 1e-12);
        assert!((c.i_com.y_axis.y - 10.0).abs() < 1e-12);
        assert!((c.i_com.z_axis.z - 5.0).abs() < 1e-12);
        let cyl = RigidInertia::cylinder(2.0, 0.5, 1.0, 2);
        assert!((cyl.i_com.z_axis.z - 0.25).abs() < 1e-12);
        // Parallel-axis theorem via i_origin.
        let p = RigidInertia::point(2.0, DVec3::new(0.0, 0.0, 3.0));
        assert!((p.i_origin().x_axis.x - 18.0).abs() < 1e-12);
        assert!(p.i_origin().z_axis.z.abs() < 1e-12);
    }
}
