//! Static-world and environment query timings (targets in docs/PLAN.md: terrain height ≤ 50 ns,
//! surface query ≤ 100 ns).

use autonomousim_core::contact::{
    ContactCache, ContactModel, ContactPoint, ContactScratch, SphereCollider, StaticScene, compute_contacts,
};
use autonomousim_core::dynamics::{JointType, KinCache, MultibodyModel, forward_kinematics};
use autonomousim_core::geometry::{HitMask, Ray, StaticGeometry};
use autonomousim_core::math::{Pose, RigidInertia, SpatialForce};
use autonomousim_core::rng::Seed;
use autonomousim_core::terrain::Terrain;
use autonomousim_world::environment::{Atmosphere, Dryden, DrydenScales};
use autonomousim_world::testworlds;
use criterion::{Criterion, criterion_group, criterion_main};
use glam::{DVec2, DVec3};
use std::hint::black_box;

const N: usize = 1024;

fn queries(c: &mut Criterion) {
    let world = testworlds::forest_patch(512.0, 150.0, 1);
    let terrain = world.terrain();
    let mut rng = Seed::from_u64(0).child("bench").rng();
    let xy: Vec<DVec2> = (0..N).map(|_| DVec2::new(rng.range(-250.0, 250.0), rng.range(-250.0, 250.0))).collect();
    let near: Vec<DVec3> = xy.iter().map(|p| p.extend(terrain.height(p.x, p.y) + 0.5)).collect();
    let rays: Vec<Ray> = near
        .iter()
        .map(|p| {
            let yaw = rng.range(0.0, std::f64::consts::TAU);
            let pitch = rng.range(-0.35, 0.05);
            Ray::new(*p + DVec3::Z * 1.5, DVec3::new(yaw.cos() * pitch.cos(), yaw.sin() * pitch.cos(), pitch.sin()))
        })
        .collect();

    let mut i = 0;
    let mut next = move || {
        i = (i + 1) % N;
        i
    };
    c.bench_function("terrain/height", |b| {
        b.iter(|| {
            let p = xy[next()];
            black_box(terrain.height(p.x, p.y))
        })
    });
    c.bench_function("terrain/height_normal", |b| {
        b.iter(|| {
            let p = xy[next()];
            black_box(terrain.height_normal(p.x, p.y))
        })
    });
    c.bench_function("terrain/closest_point", |b| b.iter(|| black_box(terrain.closest_point(near[next()], 1.0))));
    c.bench_function("terrain/raycast_down", |b| {
        b.iter(|| {
            let p = near[next()] + DVec3::Z * 50.0;
            black_box(terrain.raycast(&Ray { origin: p, dir: -DVec3::Z }, 100.0, HitMask::TERRAIN))
        })
    });
    c.bench_function("world/raycast_lidar_100m", |b| {
        b.iter(|| black_box(world.raycast(&rays[next()], 100.0, HitMask::ALL)))
    });
    let mut out = Vec::with_capacity(8);
    c.bench_function("obstacles/sphere_contacts", |b| {
        b.iter(|| {
            out.clear();
            world.obstacles().sphere_contacts(near[next()], 0.1, 0.05, HitMask::COLLIDABLE, &mut out);
            black_box(out.len())
        })
    });

    // Quad-like body with 4 feet and 4 rotor guards, airborne and landed on the forest floor.
    let mut model = MultibodyModel::new();
    model.add_link("body", None, JointType::Free, Pose::IDENTITY, RigidInertia::cuboid(1.5, DVec3::new(0.3, 0.3, 0.1)));
    let mut colliders: Vec<SphereCollider> = [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)]
        .map(|(x, y)| SphereCollider::new(0, DVec3::new(0.1 * x, 0.1 * y, -0.1), 0.02, 0))
        .to_vec();
    colliders.extend(
        [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]
            .map(|(x, y)| SphereCollider::new(0, DVec3::new(0.25 * x, 0.25 * y, 0.0), 0.08, 1)),
    );
    let contact = ContactModel::for_mass(1.5 / 4.0, 0.002);
    let scene = StaticScene { terrain: world.terrain(), obstacles: world.obstacles(), materials: world.materials() };
    let (mut cache, mut scratch, mut out) =
        (ContactCache::default(), ContactScratch::default(), Vec::<ContactPoint>::new());
    let mut f_ext = [SpatialForce::ZERO];
    let mut kin = KinCache::new(&model);
    // A clearing (no obstacle within 1 m) to land in.
    let spot = xy
        .iter()
        .find(|p| {
            let at = p.extend(terrain.height(p.x, p.y) + 0.5);
            world.obstacles().nearest_distance(at, 1.0, HitMask::SOLID | HitMask::FOLIAGE).is_none()
        })
        .unwrap();
    let (h, n) = terrain.height_normal(spot.x, spot.y);
    let landed = Pose::new(spot.extend(h) + n * 0.1195, glam::DQuat::from_rotation_arc(DVec3::Z, n));
    // 5 m above ground between trees: within the bounding box of a canopy but clear of it.
    let (mut cand, fol) = (Vec::new(), HitMask::SOLID | HitMask::FOLIAGE);
    let between = xy
        .iter()
        .map(|p| p.extend(terrain.height(p.x, p.y) + 5.0))
        .find(|&p| {
            cand.clear();
            world.obstacles().query_candidates(p - 0.35, p + 0.35, fol, &mut cand);
            !cand.is_empty() && world.obstacles().nearest_distance(p, 0.4, fol).is_none()
        })
        .unwrap();
    let in_canopy = world
        .obstacles()
        .obstacles()
        .iter()
        .find(|o| o.class == autonomousim_world::ObstacleClass::Foliage)
        .map(|o| o.pose.pos)
        .unwrap();
    for (name, pose, contacts) in [
        ("contact/quad_above_canopy", Pose::from_translation(spot.extend(h + 50.0)), 0),
        ("contact/quad_between_trees", Pose::from_translation(between), 0),
        ("contact/quad_in_canopy", Pose::from_translation(in_canopy), 8),
        ("contact/quad_landed", landed, 4),
    ] {
        let mut q = model.neutral_state();
        q.q[..3].copy_from_slice(&pose.pos.to_array());
        q.q[3..7].copy_from_slice(&pose.rot.to_array());
        forward_kinematics(&model, &q.q, &q.v, &mut kin);
        out.clear();
        compute_contacts(&scene, &kin, &colliders, &contact, 0.002, &mut cache, &mut scratch, &mut f_ext, &mut out);
        assert_eq!(out.len(), contacts, "{name}");
        c.bench_function(name, |b| {
            b.iter(|| {
                out.clear();
                compute_contacts(
                    &scene,
                    &kin,
                    &colliders,
                    &contact,
                    0.002,
                    &mut cache,
                    &mut scratch,
                    &mut f_ext,
                    &mut out,
                );
                black_box(out.len())
            })
        });
    }

    let atm = Atmosphere::default();
    c.bench_function("atmosphere/at_altitude", |b| b.iter(|| black_box(atm.at_altitude(black_box(612.0)))));
    let scales = DrydenScales::low_altitude(20.0, 7.7);
    let mut d = Dryden::stationary(&mut rng);
    c.bench_function("wind/dryden_step", |b| b.iter(|| d.step(0.002, black_box(5.0), &scales, &mut rng)));
}

criterion_group!(benches, queries);
criterion_main!(benches);
