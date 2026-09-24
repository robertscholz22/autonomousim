//! Quaternion helpers (Hamilton convention, glam `DQuat` stores `(x, y, z, w)`).

use glam::{DMat3, DQuat, DVec3};

/// Integrate an orientation with a constant body-frame angular velocity over `dt`
/// using the exact exponential map: `q ← q ⊗ exp(ω dt)`.
#[inline]
pub fn integrate_body_rate(q: DQuat, omega_body: DVec3, dt: f64) -> DQuat {
    (q * exp_map(omega_body * dt)).normalize()
}

/// Quaternion of a rotation vector (axis × angle).
#[inline]
pub fn exp_map(rotvec: DVec3) -> DQuat {
    let theta2 = rotvec.length_squared();
    if theta2 < 1e-24 {
        // Second-order Taylor expansion keeps full precision near zero.
        let h = 0.5 * rotvec;
        return DQuat::from_xyzw(h.x, h.y, h.z, 1.0 - theta2 / 8.0).normalize();
    }
    let theta = theta2.sqrt();
    let (s, c) = (0.5 * theta).sin_cos();
    let k = s / theta;
    DQuat::from_xyzw(rotvec.x * k, rotvec.y * k, rotvec.z * k, c)
}

/// Rotation vector of a quaternion (inverse of [`exp_map`]), angle in `[0, π]`.
#[inline]
pub fn log_map(q: DQuat) -> DVec3 {
    let q = if q.w < 0.0 { -q } else { q };
    let v = DVec3::new(q.x, q.y, q.z);
    let s = v.length();
    if s < 1e-12 {
        return 2.0 * v;
    }
    let angle = 2.0 * s.atan2(q.w);
    v * (angle / s)
}

/// Continuous 6D rotation representation (first two columns of the rotation matrix),
/// as used for neural-network observations (Zhou et al. 2019).
#[inline]
pub fn rot6d(q: DQuat) -> [f64; 6] {
    let m = DMat3::from_quat(q);
    [m.x_axis.x, m.x_axis.y, m.x_axis.z, m.y_axis.x, m.y_axis.y, m.y_axis.z]
}

/// Heading angle about world +Z of the body x-axis (ENU/FLU: 0 = east, CCW positive).
#[inline]
pub fn yaw(q: DQuat) -> f64 {
    let fwd = q * DVec3::X;
    fwd.y.atan2(fwd.x)
}

/// Angle between the body z-axis and world +Z (0 = level, π = upside down).
#[inline]
pub fn tilt(q: DQuat) -> f64 {
    (q * DVec3::Z).z.clamp(-1.0, 1.0).acos()
}

/// Rotation about world Z by `yaw`.
#[inline]
pub fn from_yaw(yaw: f64) -> DQuat {
    DQuat::from_rotation_z(yaw)
}

/// Angle wrapped to `[−π, π)`.
#[inline]
pub fn wrap_angle(a: f64) -> f64 {
    use std::f64::consts::{PI, TAU};
    (a + PI).rem_euclid(TAU) - PI
}

/// Uniformly distributed random rotation from three uniforms in `[0, 1)` (Shoemake 1992).
pub fn uniform_rotation(u1: f64, u2: f64, u3: f64) -> DQuat {
    use std::f64::consts::TAU;
    let a = (1.0 - u1).sqrt();
    let b = u1.sqrt();
    let (s2, c2) = (TAU * u2).sin_cos();
    let (s3, c3) = (TAU * u3).sin_cos();
    DQuat::from_xyzw(a * s2, a * c2, b * s3, b * c3).normalize()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::testutil::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn exp_log_roundtrip(axis in unit_vec3(), angle in 0.0..3.1f64) {
            let rv = axis * angle;
            let back = log_map(exp_map(rv));
            prop_assert!((back - rv).length() < 1e-9);
        }

        #[test]
        fn exp_matches_axis_angle(axis in unit_vec3(), angle in -3.0..3.0f64) {
            let a = exp_map(axis * angle);
            let b = DQuat::from_axis_angle(axis, angle);
            prop_assert!(a.dot(b).abs() > 1.0 - 1e-12);
        }

        /// Constant body rate integrates exactly regardless of the number of steps.
        #[test]
        fn constant_rate_integration_is_exact(q0 in quat(), w in vec3(), n in 1usize..50) {
            let t = 0.7;
            let mut q = q0;
            for _ in 0..n { q = integrate_body_rate(q, w, t / n as f64); }
            let exact = q0 * exp_map(w * t);
            prop_assert!(q.dot(exact).abs() > 1.0 - 1e-10);
        }

        #[test]
        fn rot6d_columns_orthonormal(q in quat()) {
            let r = rot6d(q);
            let a = DVec3::new(r[0], r[1], r[2]);
            let b = DVec3::new(r[3], r[4], r[5]);
            prop_assert!((a.length() - 1.0).abs() < 1e-12 && (b.length() - 1.0).abs() < 1e-12 && a.dot(b).abs() < 1e-12);
        }

        #[test]
        fn wrap_angle_is_periodic(a in -3.1..3.1f64, k in -5i32..5) {
            prop_assert!((wrap_angle(a + f64::from(k) * std::f64::consts::TAU) - a).abs() < 1e-12);
        }

        #[test]
        fn yaw_roundtrip(y in -3.1..3.1f64) {
            prop_assert!((yaw(from_yaw(y)) - y).abs() < 1e-12);
            prop_assert!(tilt(from_yaw(y)) < 1e-7);
        }
    }

    #[test]
    fn tiny_rotations_are_accurate() {
        let rv = DVec3::new(1e-14, -2e-14, 3e-14);
        let q = exp_map(rv);
        assert!((log_map(q) - rv).length() < 1e-26);
    }

    #[test]
    fn uniform_rotation_is_unit_and_covers_hemisphere() {
        let mut up = 0;
        let n = 2000;
        for i in 0..n {
            let u = |k: u64| ((i as u64 * 2654435761 + k * 40503) % 10007) as f64 / 10007.0;
            let q = uniform_rotation(u(1), u(2), u(3));
            assert!((q.length() - 1.0).abs() < 1e-12);
            if (q * DVec3::Z).z > 0.0 {
                up += 1;
            }
        }
        // Body z-axis is uniform on the sphere → about half point up.
        assert!((up as f64 / n as f64 - 0.5).abs() < 0.05, "{up}");
    }
}
