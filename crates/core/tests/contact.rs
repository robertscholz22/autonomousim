//! Validation of the penalty contact model: rest penetration, restitution, stiction on an
//! incline (no creep), sliding and rolling accelerations.

use autonomousim_core::contact::*;
use autonomousim_core::dynamics::*;
use autonomousim_core::geometry::NoObstacles;
use autonomousim_core::material::{MaterialId, MaterialTable};
use autonomousim_core::math::{Pose, RigidInertia, SpatialForce};
use autonomousim_core::terrain::{FlatTerrain, PlaneTerrain, Terrain};
use glam::{DQuat, DVec3};

const G: DVec3 = DVec3::new(0.0, 0.0, -9.80665);
const DT: f64 = 0.002;

/// A single free rigid body with sphere colliders, stepped with semi-implicit Euler.
struct Body {
    model: MultibodyModel,
    state: MbState,
    ws: AbaWorkspace,
    colliders: Vec<SphereCollider>,
    contact: ContactModel,
    cache: ContactCache,
    scratch: ContactScratch,
    f_ext: Vec<SpatialForce>,
    contacts: Vec<ContactPoint>,
}

impl Body {
    fn new(inertia: RigidInertia, colliders: Vec<SphereCollider>, contact: ContactModel, pose: Pose) -> Self {
        let mut model = MultibodyModel::new();
        model.add_link("body", None, JointType::Free, Pose::IDENTITY, inertia);
        let mut state = model.neutral_state();
        state.q[..3].copy_from_slice(&pose.pos.to_array());
        state.q[3..7].copy_from_slice(&pose.rot.to_array());
        let ws = AbaWorkspace::new(&model);
        Self {
            f_ext: vec![SpatialForce::ZERO; 1],
            model,
            state,
            ws,
            colliders,
            contact,
            cache: ContactCache::default(),
            scratch: ContactScratch::default(),
            contacts: Vec::new(),
        }
    }

    fn sphere(mass: f64, radius: f64, contact: ContactModel, pos: DVec3) -> Self {
        let c = SphereCollider::new(0, DVec3::ZERO, radius, 0);
        Self::new(RigidInertia::sphere(mass, radius), vec![c], contact, Pose::from_translation(pos))
    }

    fn set_world_velocity(&mut self, v: DVec3) {
        let rot = DQuat::from_slice(&self.state.q[3..7]);
        self.state.v[3..6].copy_from_slice(&(rot.inverse() * v).to_array());
    }

    fn pos(&self) -> DVec3 {
        DVec3::from_slice(&self.state.q[..3])
    }

    fn world_velocity(&self) -> DVec3 {
        DQuat::from_slice(&self.state.q[3..7]) * DVec3::from_slice(&self.state.v[3..6])
    }

    fn step(&mut self, terrain: &dyn Terrain, gravity: DVec3) {
        let materials = MaterialTable::standard();
        let scene = StaticScene { terrain, obstacles: &NoObstacles, materials: &materials };
        forward_kinematics(&self.model, &self.state.q, &self.state.v, &mut self.ws.kin);
        self.f_ext.fill(SpatialForce::ZERO);
        self.contacts.clear();
        compute_contacts(
            &scene,
            &self.ws.kin,
            &self.colliders,
            &self.contact,
            DT,
            &mut self.cache,
            &mut self.scratch,
            &mut self.f_ext,
            &mut self.contacts,
        );
        let tau = [0.0; 6];
        aba_with_kinematics(&self.model, &tau, &self.f_ext, gravity, &mut self.ws).unwrap();
        semi_implicit_euler(&self.model, &mut self.state, &self.ws.qdd, DT);
    }
}

#[test]
fn drop_settles_at_static_penetration() {
    let (m, r) = (1.0, 0.1);
    let contact = ContactModel::for_mass(m, DT);
    for (material, scale) in [(MaterialId::ROCK, 1.0), (MaterialId::MUD, 0.3)] {
        let ground = FlatTerrain::new(0.0, material);
        let mut body = Body::sphere(m, r, contact, DVec3::new(0.0, 0.0, 1.0));
        for _ in 0..(3.0 / DT) as usize {
            body.step(&ground, G);
            assert!(body.state.is_finite());
        }
        let expected = m * 9.80665 / (contact.solid.stiffness * scale * scale);
        let depth = r - body.pos().z;
        assert!((depth / expected - 1.0).abs() < 1e-3, "{material:?}: depth {depth} vs {expected}");
        assert!(body.world_velocity().length() < 1e-6);
        assert_eq!(body.contacts.len(), 1);
        assert!((body.contacts[0].normal_force - m * 9.80665).abs() < 1e-6);
    }
}

/// Rebound/impact speed ratio of the continuous 1-D model `m ẍ = max(0, −k x − c ẋ)` (x < 0).
fn reference_restitution(m: f64, k: f64, c: f64) -> f64 {
    let f = |x: f64, v: f64| if x < 0.0 { (-k * x - c * v).max(0.0) / m } else { 0.0 };
    let h = 1e-6;
    let (mut x, mut v) = (-1e-12, -1.0);
    while x < 0.0 {
        let (k1x, k1v) = (v, f(x, v));
        let (k2x, k2v) = (v + 0.5 * h * k1v, f(x + 0.5 * h * k1x, v + 0.5 * h * k1v));
        let (k3x, k3v) = (v + 0.5 * h * k2v, f(x + 0.5 * h * k2x, v + 0.5 * h * k2v));
        let (k4x, k4v) = (v + h * k3v, f(x + h * k3x, v + h * k3v));
        x += h / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
        v += h / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);
    }
    v
}

#[test]
fn restitution_matches_damping_ratio() {
    let (m, r) = (0.5, 0.05);
    for zeta in [0.05, 0.1, 0.3, 0.8] {
        let params = PenaltyParams::from_frequency(m, 0.2 / DT, zeta);
        let contact = ContactModel { solid: params, foliage: params };
        let ground = FlatTerrain::new(0.0, MaterialId::ROCK);
        let want = reference_restitution(m, params.stiffness, params.damping);
        // Average over impact phases relative to the step grid.
        let mut sum = 0.0;
        let n = 16;
        for j in 0..n {
            let mut body = Body::sphere(m, r, contact, DVec3::new(0.0, 0.0, r + 0.01 + 2.0 * DT * j as f64 / n as f64));
            body.set_world_velocity(DVec3::new(0.0, 0.0, -2.0));
            let mut touched = false;
            for _ in 0..1000 {
                body.step(&ground, DVec3::ZERO);
                touched |= !body.contacts.is_empty();
                if touched && body.contacts.is_empty() && body.pos().z > r {
                    break;
                }
            }
            assert!(body.cache.is_empty());
            sum += body.world_velocity().z / 2.0;
        }
        let e = sum / n as f64;
        // Linear model without the force clamp (for orientation): exp(−ζπ/√(1−ζ²)).
        let unclamped = (-zeta * std::f64::consts::PI / (1.0 - zeta * zeta).sqrt()).exp();
        assert!((e - want).abs() < 0.02, "ζ = {zeta}: e = {e:.4}, continuous {want:.4}, unclamped {unclamped:.4}");
    }
}

/// Box on four corner spheres resting on an incline of angle `theta`.
fn box_on_incline(theta: f64) -> (Body, PlaneTerrain, DVec3) {
    let m = 2.0;
    let ground = PlaneTerrain::incline(theta, MaterialId::GRASS);
    let feet = [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)]
        .map(|(x, y)| SphereCollider::new(0, DVec3::new(0.18 * x, 0.18 * y, -0.05), 0.02, 0));
    let rot = DQuat::from_rotation_y(-theta);
    let n = rot * DVec3::Z;
    let body = Body::new(
        RigidInertia::cuboid(m, DVec3::new(0.4, 0.4, 0.1)),
        feet.to_vec(),
        ContactModel::for_mass(m / 4.0, DT),
        Pose::new(n * 0.07, rot),
    );
    let downhill = DVec3::new(-theta.cos(), 0.0, -theta.sin());
    (body, ground, downhill)
}

#[test]
fn sticks_on_incline_without_creep() {
    let theta = 20f64.to_radians(); // tan θ = 0.36 < μ_grass = 0.45
    let (mut body, ground, _) = box_on_incline(theta);
    for _ in 0..(1.0 / DT) as usize {
        body.step(&ground, G);
    }
    let settled = body.pos();
    for _ in 0..(9.0 / DT) as usize {
        body.step(&ground, G);
    }
    let drift = (body.pos() - settled).length();
    assert!(drift < 1e-6, "crept {drift} m in 9 s");
    assert_eq!(body.contacts.len(), 4);
    assert!(body.contacts.iter().all(|c| !c.slipping));
    assert_eq!(body.cache.len(), 4);
}

#[test]
fn slides_with_coulomb_acceleration() {
    let theta = 35f64.to_radians(); // tan θ = 0.70 > μ_grass = 0.45
    let (mut body, ground, downhill) = box_on_incline(theta);
    let mut v = Vec::new();
    for k in 0..(1.5 / DT) as usize {
        body.step(&ground, G);
        if k + 1 == (0.5 / DT) as usize || k + 1 == (1.5 / DT) as usize {
            v.push(body.world_velocity().dot(downhill));
        }
    }
    let a = v[1] - v[0];
    let mu = MaterialTable::standard().get(MaterialId::GRASS).friction;
    let want = 9.80665 * (theta.sin() - mu * theta.cos());
    assert!((a / want - 1.0).abs() < 0.02, "a = {a}, want {want}");
    assert!(body.contacts.iter().all(|c| c.slipping));
}

#[test]
fn sphere_rolls_without_bristle_resistance() {
    let theta = 10f64.to_radians();
    let (m, r) = (1.0, 0.1);
    let ground = PlaneTerrain::incline(theta, MaterialId::ROCK);
    let n = DVec3::new(-theta.sin(), 0.0, theta.cos());
    let mut body = Body::sphere(m, r, ContactModel::for_mass(m, DT), n * (r - 1e-3));
    let downhill = DVec3::new(-theta.cos(), 0.0, -theta.sin());
    let mut v = Vec::new();
    for k in 0..(1.5 / DT) as usize {
        body.step(&ground, G);
        if k + 1 == (0.5 / DT) as usize || k + 1 == (1.5 / DT) as usize {
            v.push(body.world_velocity().dot(downhill));
        }
    }
    let a = v[1] - v[0];
    let want = 5.0 / 7.0 * 9.80665 * theta.sin();
    assert!((a / want - 1.0).abs() < 0.02, "a = {a}, want {want}");
    assert!(!body.contacts[0].slipping);
}

#[test]
fn airborne_bodies_skip_queries_and_clear_cache() {
    let ground = FlatTerrain::new(0.0, MaterialId::ROCK);
    let mut body = Body::sphere(1.0, 0.1, ContactModel::for_mass(1.0, DT), DVec3::new(0.0, 0.0, 0.0995));
    body.step(&ground, G);
    assert_eq!(body.cache.len(), 1);
    body.state.q[2] = 5.0;
    body.step(&ground, G);
    assert!(body.contacts.is_empty() && body.cache.is_empty());
}
