//! Penalty contacts against obstacles of a `StaticWorld` (solid tops, walls, foliage).

use autonomousim_core::contact::*;
use autonomousim_core::dynamics::*;
use autonomousim_core::geometry::HitKind;
use autonomousim_core::math::{Pose, RigidInertia, SpatialForce};
use autonomousim_world::{StaticWorld, testworlds};
use glam::{DQuat, DVec3};

const G: DVec3 = DVec3::new(0.0, 0.0, -9.80665);
const DT: f64 = 0.002;

struct Ball {
    model: MultibodyModel,
    state: MbState,
    ws: AbaWorkspace,
    collider: [SphereCollider; 1],
    contact: ContactModel,
    cache: ContactCache,
    scratch: ContactScratch,
    contacts: Vec<ContactPoint>,
}

impl Ball {
    fn new(mass: f64, radius: f64, pos: DVec3, vel: DVec3) -> Self {
        let mut model = MultibodyModel::new();
        model.add_link("ball", None, JointType::Free, Pose::IDENTITY, RigidInertia::sphere(mass, radius));
        let mut state = model.neutral_state();
        state.q[..3].copy_from_slice(&pos.to_array());
        state.v[3..6].copy_from_slice(&vel.to_array());
        Self {
            ws: AbaWorkspace::new(&model),
            model,
            state,
            collider: [SphereCollider { link: 0, center: DVec3::ZERO, radius, group: 0 }],
            contact: ContactModel::for_mass(mass, DT),
            cache: ContactCache::default(),
            scratch: ContactScratch::default(),
            contacts: Vec::new(),
        }
    }

    fn pos(&self) -> DVec3 {
        DVec3::from_slice(&self.state.q[..3])
    }

    fn vel(&self) -> DVec3 {
        DQuat::from_slice(&self.state.q[3..7]) * DVec3::from_slice(&self.state.v[3..6])
    }

    fn step(&mut self, world: &StaticWorld, gravity: DVec3) {
        let scene =
            StaticScene { terrain: world.terrain(), obstacles: world.obstacles(), materials: world.materials() };
        forward_kinematics(&self.model, &self.state.q, &self.state.v, &mut self.ws.kin);
        let mut f_ext = [SpatialForce::ZERO];
        self.contacts.clear();
        compute_contacts(
            &scene,
            &self.ws.kin,
            &self.collider,
            &self.contact,
            DT,
            &mut self.cache,
            &mut self.scratch,
            &mut f_ext,
            &mut self.contacts,
        );
        aba_with_kinematics(&self.model, &[0.0; 6], &f_ext, gravity, &mut self.ws).unwrap();
        semi_implicit_euler(&self.model, &mut self.state, &self.ws.qdd, DT);
    }
}

#[test]
fn rests_on_pillar_top() {
    let world = testworlds::pillars(1, 10.0, 0.5, 8.0);
    let mut ball = Ball::new(1.0, 0.1, DVec3::new(0.1, 0.0, 9.0), DVec3::ZERO);
    for _ in 0..(3.0 / DT) as usize {
        ball.step(&world, G);
    }
    let concrete = world.material(autonomousim_core::material::MaterialId::CONCRETE).stiffness_scale;
    let expected = 8.1 - 9.80665 / (ball.contact.solid.stiffness * concrete * concrete);
    assert!((ball.pos().z - expected).abs() < 1e-6, "z = {}, want {expected}", ball.pos().z);
    assert_eq!(ball.contacts.len(), 1);
    assert_eq!(ball.contacts[0].kind, HitKind::Solid(0));
}

#[test]
fn bounces_off_wall() {
    let world = testworlds::walled_arena(10.0, 3.0);
    let mut ball = Ball::new(1.0, 0.1, DVec3::new(8.0, 0.0, 1.0), DVec3::new(5.0, 0.0, 0.0));
    let mut hit = None;
    for _ in 0..1000 {
        ball.step(&world, DVec3::ZERO);
        if let Some(c) = ball.contacts.first() {
            hit = Some(c.kind);
        }
        if hit.is_some() && ball.contacts.is_empty() {
            break;
        }
    }
    assert!(matches!(hit, Some(HitKind::Solid(_))));
    let v = ball.vel();
    // ζ = 0.8 with the force clamp gives e ≈ 0.18 (see the core contact tests).
    assert!(v.x < 0.0 && (v.x / -5.0 - 0.17).abs() < 0.03, "rebound {v}");
    assert!(v.y.abs() < 1e-9 && v.z.abs() < 1e-9);
}

#[test]
fn foliage_is_soft() {
    // Fly into the side of the canopy (radius 1.67 m at z = 6) without gravity.
    let world = testworlds::single_tree();
    let mut ball = Ball::new(1.0, 0.1, DVec3::new(-3.0, 0.0, 6.0), DVec3::new(2.0, 0.0, 0.0));
    let (mut max_depth, mut touched): (f64, bool) = (0.0, false);
    for _ in 0..(3.0 / DT) as usize {
        ball.step(&world, DVec3::ZERO);
        for c in &ball.contacts {
            assert_eq!(c.kind, HitKind::Foliage(1));
            max_depth = max_depth.max(c.depth);
            touched = true;
        }
    }
    // Heavily damped (ζ = 2): stopped within a few centimetres, then pushed back out.
    assert!(touched && ball.contacts.is_empty());
    assert!(max_depth > 0.01 && max_depth < 0.05, "{max_depth}");
    let outward = DVec3::new(-6.0, 0.0, 2.5).normalize();
    assert!(ball.vel().dot(outward) > 0.0, "{}", ball.vel());
}
