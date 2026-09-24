//! Spatial (6D) motion and force vectors in Featherstone's Plücker notation.
//!
//! Motion vectors are `(angular, linear)` = `(ω, v)` where `v` is the velocity of the
//! point at the frame origin. Force vectors are `(moment, force)` = `(n, f)` with the
//! moment taken about the frame origin. Motion and force vectors are dual: their dot
//! product is power.

use glam::DVec3;
use serde::{Deserialize, Serialize};
use std::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

/// Spatial motion vector (velocity, acceleration, joint motion subspace column).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SpatialMotion {
    pub ang: DVec3,
    pub lin: DVec3,
}

/// Spatial force vector (wrench, momentum, `I·S` columns).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SpatialForce {
    pub ang: DVec3,
    pub lin: DVec3,
}

macro_rules! impl_vec6 {
    ($t:ident) => {
        impl $t {
            pub const ZERO: Self = Self { ang: DVec3::ZERO, lin: DVec3::ZERO };

            #[inline]
            pub const fn new(ang: DVec3, lin: DVec3) -> Self {
                Self { ang, lin }
            }

            #[inline]
            pub fn from_array(a: [f64; 6]) -> Self {
                Self { ang: DVec3::new(a[0], a[1], a[2]), lin: DVec3::new(a[3], a[4], a[5]) }
            }

            #[inline]
            pub fn to_array(self) -> [f64; 6] {
                [self.ang.x, self.ang.y, self.ang.z, self.lin.x, self.lin.y, self.lin.z]
            }

            /// Component `k` in `(ang.x, ang.y, ang.z, lin.x, lin.y, lin.z)` order.
            #[inline]
            pub fn get(&self, k: usize) -> f64 {
                if k < 3 { self.ang[k] } else { self.lin[k - 3] }
            }

            /// Euclidean dot product of the raw 6 components (same-kind vectors).
            #[inline]
            pub fn dot6(self, o: Self) -> f64 {
                self.ang.dot(o.ang) + self.lin.dot(o.lin)
            }

            #[inline]
            pub fn is_finite(self) -> bool {
                self.ang.is_finite() && self.lin.is_finite()
            }
        }

        impl Add for $t {
            type Output = Self;
            #[inline]
            fn add(self, o: Self) -> Self {
                Self { ang: self.ang + o.ang, lin: self.lin + o.lin }
            }
        }
        impl AddAssign for $t {
            #[inline]
            fn add_assign(&mut self, o: Self) {
                self.ang += o.ang;
                self.lin += o.lin;
            }
        }
        impl Sub for $t {
            type Output = Self;
            #[inline]
            fn sub(self, o: Self) -> Self {
                Self { ang: self.ang - o.ang, lin: self.lin - o.lin }
            }
        }
        impl SubAssign for $t {
            #[inline]
            fn sub_assign(&mut self, o: Self) {
                self.ang -= o.ang;
                self.lin -= o.lin;
            }
        }
        impl Neg for $t {
            type Output = Self;
            #[inline]
            fn neg(self) -> Self {
                Self { ang: -self.ang, lin: -self.lin }
            }
        }
        impl Mul<f64> for $t {
            type Output = Self;
            #[inline]
            fn mul(self, s: f64) -> Self {
                Self { ang: self.ang * s, lin: self.lin * s }
            }
        }
        impl Mul<$t> for f64 {
            type Output = $t;
            #[inline]
            fn mul(self, v: $t) -> $t {
                v * self
            }
        }
    };
}

impl_vec6!(SpatialMotion);
impl_vec6!(SpatialForce);

impl SpatialMotion {
    /// Power `mᵀ f` (motion · force).
    #[inline]
    pub fn dot(self, f: SpatialForce) -> f64 {
        self.ang.dot(f.ang) + self.lin.dot(f.lin)
    }

    /// Motion cross product `self × m` (Featherstone `crm(self) m`).
    #[inline]
    pub fn cross_motion(self, m: SpatialMotion) -> SpatialMotion {
        SpatialMotion { ang: self.ang.cross(m.ang), lin: self.ang.cross(m.lin) + self.lin.cross(m.ang) }
    }

    /// Force cross product `self ×* f` (Featherstone `crf(self) f`).
    #[inline]
    pub fn cross_force(self, f: SpatialForce) -> SpatialForce {
        SpatialForce { ang: self.ang.cross(f.ang) + self.lin.cross(f.lin), lin: self.ang.cross(f.lin) }
    }

    /// Velocity of a point at offset `r` from the frame origin (same coordinates).
    #[inline]
    pub fn point_velocity(self, r: DVec3) -> DVec3 {
        self.lin + self.ang.cross(r)
    }
}

impl SpatialForce {
    /// Wrench of a pure force `f` applied at point `p` (both in this frame's coordinates).
    #[inline]
    pub fn from_force_at_point(f: DVec3, p: DVec3) -> Self {
        Self { ang: p.cross(f), lin: f }
    }

    /// Pure moment (couple).
    #[inline]
    pub fn from_moment(n: DVec3) -> Self {
        Self { ang: n, lin: DVec3::ZERO }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::testutil::*;
    use proptest::prelude::*;

    proptest! {
        /// (m1 × m2) · f = -m2 · (m1 ×* f): crf = -crmᵀ.
        #[test]
        fn cross_force_is_negative_transpose_of_cross_motion(m1 in motion(), m2 in motion(), f in force()) {
            let lhs = m1.cross_motion(m2).dot(f);
            let rhs = -m2.dot(m1.cross_force(f));
            prop_assert!((lhs - rhs).abs() <= 1e-9 * (1.0 + lhs.abs()));
        }

        #[test]
        fn self_cross_is_zero(m in motion()) {
            let c = m.cross_motion(m);
            prop_assert!(c.ang.length() < 1e-9 && c.lin.length() < 1e-9 * (1.0 + m.lin.length() * m.ang.length()));
        }

        /// Power of a force applied at a point equals force · point velocity.
        #[test]
        fn force_at_point_power(m in motion(), f in vec3(), p in vec3()) {
            let w = SpatialForce::from_force_at_point(f, p);
            let lhs = m.dot(w);
            let rhs = f.dot(m.point_velocity(p));
            prop_assert!((lhs - rhs).abs() <= 1e-9 * (1.0 + lhs.abs()));
        }
    }
}
