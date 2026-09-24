//! Coordinate-frame conventions and the (only) conversions between them.
//!
//! * **World**: ENU — x east, y north, z up (REP-103). Gravity is `-Z`.
//! * **Body**: FLU — x forward, y left, z up.
//! * **Bevy** (viewer): right-handed, Y up, −Z forward. ENU `(x, y, z)` ↦ Bevy `(x, z, −y)`.
//! * **parry `HeightField`**: local Y up, rows along local z, columns along local x.
//!   Placed in ENU with a `+90°` rotation about X (local y ↦ world z, local z ↦ world −y),
//!   so its rows must be stored in *decreasing* world-y order.

use glam::{DQuat, DVec3};

/// Standard gravity (m/s²).
pub const STANDARD_GRAVITY: f64 = 9.80665;

/// World gravity vector in ENU.
pub const GRAVITY_ENU: DVec3 = DVec3::new(0.0, 0.0, -STANDARD_GRAVITY);

/// ENU vector → Bevy (f32, Y-up) vector.
#[inline]
pub fn enu_to_bevy_vec(v: DVec3) -> [f32; 3] {
    [v.x as f32, v.z as f32, -v.y as f32]
}

/// Bevy (f32, Y-up) vector → ENU vector.
#[inline]
pub fn bevy_to_enu_vec(v: [f32; 3]) -> DVec3 {
    DVec3::new(v[0] as f64, -v[2] as f64, v[1] as f64)
}

/// ENU rotation → Bevy rotation `[x, y, z, w]`. The basis change is a proper rotation, so
/// the rotation axis is mapped like a vector and the angle is unchanged.
#[inline]
pub fn enu_to_bevy_quat(q: DQuat) -> [f32; 4] {
    [q.x as f32, q.z as f32, -q.y as f32, q.w as f32]
}

/// Bevy rotation `[x, y, z, w]` → ENU rotation.
#[inline]
pub fn bevy_to_enu_quat(q: [f32; 4]) -> DQuat {
    DQuat::from_xyzw(q[0] as f64, -q[2] as f64, q[1] as f64, q[3] as f64)
}

/// Rotation that places a parry `HeightField` (Y-up local frame) into ENU (Z-up).
#[inline]
pub fn heightfield_to_enu_rotation() -> DQuat {
    DQuat::from_rotation_x(std::f64::consts::FRAC_PI_2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::testutil::*;
    use proptest::prelude::*;

    fn to_bevy(v: DVec3) -> DVec3 {
        let b = enu_to_bevy_vec(v);
        DVec3::new(b[0] as f64, b[1] as f64, b[2] as f64)
    }

    #[test]
    fn enu_axes_map_to_bevy_axes() {
        assert_eq!(enu_to_bevy_vec(DVec3::Z), [0.0, 1.0, 0.0]); // up is Bevy +Y
        assert_eq!(enu_to_bevy_vec(DVec3::Y), [0.0, 0.0, -1.0]); // north is Bevy forward (−Z)
        assert_eq!(enu_to_bevy_vec(DVec3::X), [1.0, 0.0, 0.0]); // east is Bevy right (+X)
    }

    #[test]
    fn heightfield_rotation_maps_axes() {
        let r = heightfield_to_enu_rotation();
        assert!((r * DVec3::Y - DVec3::Z).length() < 1e-15);
        assert!((r * DVec3::Z + DVec3::Y).length() < 1e-15);
        assert!((r * DVec3::X - DVec3::X).length() < 1e-15);
    }

    proptest! {
        #[test]
        fn vec_roundtrip(v in vec3()) {
            let b = enu_to_bevy_vec(v);
            prop_assert!((bevy_to_enu_vec(b) - v).length() < 1e-5 * (1.0 + v.length()));
        }

        /// Rotating then converting equals converting then rotating.
        #[test]
        fn quat_conversion_commutes_with_rotation(q in quat(), v in vec3()) {
            let qb = enu_to_bevy_quat(q);
            let qb = DQuat::from_xyzw(qb[0] as f64, qb[1] as f64, qb[2] as f64, qb[3] as f64);
            let lhs = qb * to_bevy(v);
            let rhs = to_bevy(q * v);
            prop_assert!((lhs - rhs).length() < 1e-5 * (1.0 + v.length()));
            let back = bevy_to_enu_quat(enu_to_bevy_quat(q));
            prop_assert!(back.dot(q).abs() > 1.0 - 1e-6);
        }
    }
}
