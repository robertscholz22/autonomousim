//! proptest strategies and approximate-equality assertions shared by the math tests.

use super::*;
use glam::{DMat3, DQuat, DVec3};
use proptest::prelude::*;

pub fn vec3() -> impl Strategy<Value = DVec3> {
    prop::array::uniform3(-5.0..5.0f64).prop_map(DVec3::from_array)
}

pub fn unit_vec3() -> impl Strategy<Value = DVec3> {
    prop::array::uniform3(-1.0..1.0f64)
        .prop_filter("non-degenerate", |a| DVec3::from_array(*a).length() > 0.1)
        .prop_map(|a| DVec3::from_array(a).normalize())
}

pub fn quat() -> impl Strategy<Value = DQuat> {
    (0.0..1.0f64, 0.0..1.0f64, 0.0..1.0f64).prop_map(|(a, b, c)| quat::uniform_rotation(a, b, c))
}

pub fn motion() -> impl Strategy<Value = SpatialMotion> {
    (vec3(), vec3()).prop_map(|(a, l)| SpatialMotion::new(a, l))
}

pub fn force() -> impl Strategy<Value = SpatialForce> {
    (vec3(), vec3()).prop_map(|(a, l)| SpatialForce::new(a, l))
}

pub fn xform() -> impl Strategy<Value = Xform> {
    (quat(), vec3()).prop_map(|(q, p)| Xform::from_pose(p, q))
}

pub fn pose() -> impl Strategy<Value = Pose> {
    (vec3(), quat()).prop_map(|(p, q)| Pose::new(p, q))
}

/// Physically valid rigid inertia (principal moments satisfy the triangle inequality).
pub fn rigid_inertia() -> impl Strategy<Value = RigidInertia> {
    (0.1..10.0f64, vec3(), prop::array::uniform3(0.01..2.0f64), quat()).prop_map(|(m, c, s, q)| {
        // Moments of a mass distribution with second moments s: (s_y+s_z, s_x+s_z, s_x+s_y).
        let d = DVec3::new(s[1] + s[2], s[0] + s[2], s[0] + s[1]) * m;
        let r = DMat3::from_quat(q);
        RigidInertia::new(m, c * 0.3, r * DMat3::from_diagonal(d) * r.transpose())
    })
}

#[track_caller]
pub fn assert_vec_close(a: DVec3, b: DVec3, tol: f64) {
    let scale = 1.0 + a.length().max(b.length());
    assert!((a - b).length() <= tol * scale, "{a:?} != {b:?}");
}

#[track_caller]
pub fn assert_motion_close(a: SpatialMotion, b: SpatialMotion, tol: f64) {
    assert_vec_close(a.ang, b.ang, tol);
    assert_vec_close(a.lin, b.lin, tol);
}

#[track_caller]
pub fn assert_force_close(a: SpatialForce, b: SpatialForce, tol: f64) {
    assert_vec_close(a.ang, b.ang, tol);
    assert_vec_close(a.lin, b.lin, tol);
}

#[track_caller]
pub fn assert_xform_close(a: Xform, b: Xform, tol: f64) {
    assert!(crate::math::linalg::mat3_max_abs(a.rot - b.rot) <= tol, "{a:?} != {b:?}");
    assert_vec_close(a.pos, b.pos, tol);
}
