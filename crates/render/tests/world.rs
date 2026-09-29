//! Maps and vehicles on the GPU against ray casts: the depth and class images of test and
//! procedural maps against the static world's ray cast (what the LiDAR sees), and vehicles
//! against a CPU ray cast of their posed rig triangles (the LiDAR sees agents as spheres).

use std::sync::{Arc, OnceLock};

use autonomousim_core::geometry::{HitKind, HitMask, Ray};
use autonomousim_core::math::Pose;
use autonomousim_core::rng::Seed;
use autonomousim_procgen::rural::{self, RuralConfig};
use autonomousim_procgen::wild::{self, WildConfig};
use autonomousim_render::{
    AdapterChoice, CameraPose, Frame, GpuContext, GpuRig, GpuWorld, Intrinsics, Renderer, SemanticClass, Shading, View,
    WorldOptions, obstacle_class, terrain_class,
};
use autonomousim_scene::rig::{Placement, Rig};
use autonomousim_sim::scenario::{GroupSpec, MapSource, SpawnSpec, Testworld, VehicleRef};
use autonomousim_sim::{Scenario, WorldInstance};
use autonomousim_world::obstacles::tags;
use autonomousim_world::{StaticWorld, testworlds};
use glam::{DQuat, DVec3};

fn ctx() -> &'static GpuContext {
    static CTX: OnceLock<GpuContext> = OnceLock::new();
    CTX.get_or_init(|| GpuContext::new(&AdapterChoice::Software).expect("lavapipe (mesa-vulkan-drivers)"))
}

/// A camera at `eye` looking at `target`, level (no roll).
fn look_at(eye: DVec3, target: DVec3) -> CameraPose {
    let d = target - eye;
    let yaw = d.y.atan2(d.x);
    let pitch = (-d.z).atan2(d.truncate().length());
    CameraPose::new(eye, DQuat::from_rotation_z(yaw) * DQuat::from_rotation_y(pitch))
}

fn view(pose: CameraPose) -> View {
    View { pose, intrinsics: Intrinsics::new(96, 72, 90f64.to_radians()), shading: Shading::default() }
}

/// What a ray meets: the thing (for edge detection), its class and axis depth.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Thing {
    Sky,
    Terrain(SemanticClass),
    Water,
    Obstacle(u32),
}

/// What the ray through `(u, w)` meets: the thing, its class, the axis depth and the cosine of
/// the incidence angle.
fn cast(world: &StaticWorld, v: &View, u: f64, w: f64) -> (Thing, SemanticClass, f64, f64) {
    let d = v.pose.orientation * v.intrinsics.ray(u, w);
    let len = d.length();
    let mask = HitMask::TERRAIN | HitMask::WATER | HitMask::SOLID | HitMask::FOLIAGE;
    let Some(hit) = world.raycast(&Ray::new(v.pose.position, d / len), v.intrinsics.far * len, mask) else {
        return (Thing::Sky, SemanticClass::Sky, 0.0, 1.0);
    };
    let depth = hit.toi / len;
    let cos = hit.normal.dot(d / len).abs();
    let (thing, class, depth) = match hit.kind {
        HitKind::Terrain => {
            let c = terrain_class(hit.material);
            (Thing::Terrain(c), c, depth)
        }
        HitKind::Water => (Thing::Water, SemanticClass::Water, depth),
        HitKind::Solid(i) | HitKind::Foliage(i) => {
            (Thing::Obstacle(i), obstacle_class(&world.obstacle_set().obstacles()[i as usize]), depth)
        }
        HitKind::Agent(_) => unreachable!("no agents in a static world"),
    };
    (thing, class, depth, cos)
}

#[derive(Debug, Default)]
struct Stats {
    /// Pixels whose neighbourhood (±0.5 px) meets one thing only.
    clean: usize,
    class_ok: usize,
    depth_ok: usize,
    /// Terrain and water nearer than 100 m, and their depth within 1e-3 relative.
    near_ground: usize,
    near_ground_ok: usize,
    classes: [usize; SemanticClass::ALL.len()],
    /// Depth outside the tolerance, by class.
    depth_bad: [usize; 14],
}

/// Compare `frame` with the ray casts of `v`'s pixel centres.
fn compare(world: &StaticWorld, v: &View, frame: &Frame, stats: &mut Stats) {
    for j in 0..frame.height {
        for i in 0..frame.width {
            let (u, w) = (i as f64 + 0.5, j as f64 + 0.5);
            let (thing, class, t, cos) = cast(world, v, u, w);
            let clean = [(-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5)]
                .iter()
                .all(|&(du, dw)| cast(world, v, u + du, w + dw).0 == thing);
            if !clean {
                continue;
            }
            stats.clean += 1;
            let (got_class, got) = (frame.class_at(i, j), frame.depth_at(i, j) as f64);
            // Road ribbons lie a few centimetres above the terrain the rays hit.
            let ribbon = got_class == SemanticClass::Road.id() && matches!(thing, Thing::Terrain(_));
            if got_class != class.id() && !(ribbon && (got - t).abs() < 0.15) {
                continue;
            }
            stats.class_ok += 1;
            stats.classes[class.id() as usize] += 1;
            let tol = match thing {
                Thing::Sky => 0.0,
                Thing::Terrain(_) | Thing::Water => 0.01 * t + 0.05,
                // Facets of the tessellated shapes, seen at the incidence angle.
                Thing::Obstacle(_) => 0.01 * t + 0.1 / cos.max(0.1),
            };
            let err = (got - t).abs();
            let good = err <= tol + if ribbon { 0.15 } else { 0.0 };
            stats.depth_ok += usize::from(good);
            stats.depth_bad[class.id() as usize] += usize::from(!good);
            if matches!(thing, Thing::Terrain(_) | Thing::Water) && t < 100.0 && !ribbon {
                stats.near_ground += 1;
                stats.near_ground_ok += usize::from(err <= 1e-3 * t + 1e-3);
            }
        }
    }
}

/// Render `views` of `world` and hold the images to the ray casts.
fn check_map(name: &str, world: &StaticWorld, views: &[View], expect: &[SemanticClass]) {
    let ctx = ctx();
    let start = std::time::Instant::now();
    let gpu = GpuWorld::new(ctx, world, WorldOptions::default()).unwrap();
    let built = start.elapsed().as_secs_f64();
    let mut r = Renderer::new(ctx);
    let mut stats = Stats::default();
    let mut draws = Vec::new();
    for v in views {
        draws.clear();
        gpu.draws(v, &mut draws);
        let frame = r.render(ctx, v, &draws).unwrap();
        compare(world, v, &frame, &mut stats);
    }
    let frac = |a: usize, b: usize| a as f64 / b.max(1) as f64;
    let seen: Vec<String> = SemanticClass::ALL
        .iter()
        .filter(|c| stats.classes[c.id() as usize] > 0)
        .map(|c| format!("{} {}", c.name(), stats.classes[c.id() as usize]))
        .collect();
    println!(
        "{name}: {} triangles, uploaded in {built:.2} s; {} clean pixels: class {:.4}, depth {:.4}, near ground {:.4} of {}; {}",
        gpu.triangle_count(),
        stats.clean,
        frac(stats.class_ok, stats.clean),
        frac(stats.depth_ok, stats.class_ok),
        frac(stats.near_ground_ok, stats.near_ground),
        stats.near_ground,
        seen.join(", ")
    );
    assert!(frac(stats.class_ok, stats.clean) > 0.97, "{name}: classes {stats:?}");
    assert!(frac(stats.depth_ok, stats.class_ok) > 0.98, "{name}: depths {stats:?}");
    assert!(frac(stats.near_ground_ok, stats.near_ground) > 0.995, "{name}: near ground {stats:?}");
    for c in expect {
        assert!(stats.classes[c.id() as usize] >= 20, "{name}: too few {} pixels: {stats:?}", c.name());
    }
}

/// Views over a map: from a few points at low and high altitude, turning around.
fn survey(world: &StaticWorld, spots: usize) -> Vec<View> {
    let (lo, hi) = world.extent();
    let mut views = Vec::new();
    for k in 0..spots {
        let f = (k as f64 + 0.5) / spots as f64;
        let x = lo.x + (hi.x - lo.x) * (0.2 + 0.6 * f);
        let y = lo.y + (hi.y - lo.y) * (0.8 - 0.6 * f);
        let ground = world.surface_height(x, y);
        for (h, look) in [(2.0, 40.0), (25.0, 60.0), (120.0, 150.0)] {
            let yaw = 2.1 * k as f64 + h;
            let eye = DVec3::new(x, y, ground + h);
            let target = DVec3::new(x + look * yaw.cos(), y + look * yaw.sin(), ground);
            views.push(view(look_at(eye, target)));
        }
    }
    views
}

/// Views of the first obstacles with `tag`, from `distance` away and `height` above their base.
fn close_ups(world: &StaticWorld, tag: u16, count: usize, distance: f64, height: f64) -> Vec<View> {
    let obstacles = world.obstacle_set().obstacles();
    obstacles
        .iter()
        .filter(|o| o.tag == tag)
        .step_by(7)
        .take(count)
        .enumerate()
        .map(|(k, o)| {
            let a = 1.3 * k as f64;
            let base = o.pose.pos.truncate().extend(world.surface_height(o.pose.pos.x, o.pose.pos.y));
            let eye = base + DVec3::new(distance * a.cos(), distance * a.sin(), height);
            view(look_at(eye, base + DVec3::Z * (0.5 * height)))
        })
        .collect()
}

#[test]
fn test_worlds_match_ray_casts() {
    let forest = testworlds::forest_patch(200.0, 150.0, 3);
    let mut views = survey(&forest, 3);
    views.extend(close_ups(&forest, tags::TRUNK, 4, 8.0, 3.0));
    views.extend(close_ups(&forest, tags::ROCK, 4, 5.0, 1.5));
    use SemanticClass as C;
    check_map("forest patch", &forest, &views, &[C::ForestFloor, C::Trunk, C::Canopy, C::Boulder, C::Sky]);

    let lake = testworlds::lake(200.0, 4.0, -1.0);
    let views = [
        view(look_at(DVec3::new(-60.0, -10.0, 6.0), DVec3::new(0.0, 0.0, -1.0))),
        view(look_at(DVec3::new(0.0, 0.0, 30.0), DVec3::new(20.0, 5.0, 0.0))),
        view(look_at(DVec3::new(10.0, 40.0, 3.0), DVec3::new(0.0, -20.0, -1.0))),
    ];
    check_map("lake", &lake, &views, &[C::Water, C::Soil, C::Sky]);
}

#[test]
fn procedural_maps_match_ray_casts() {
    use SemanticClass as C;
    let (wild, _) = wild::generate(&WildConfig::training(), 11).unwrap();
    let mut views = survey(&wild, 4);
    views.extend(close_ups(&wild, tags::TRUNK, 4, 10.0, 3.0));
    views.extend(close_ups(&wild, tags::ROCK, 3, 6.0, 2.0));
    check_map("wild training map", &wild, &views, &[C::Grass, C::ForestFloor, C::Trunk, C::Canopy, C::Sky]);

    let (farm, _) = rural::generate(&RuralConfig::training(), 5).unwrap();
    let mut views = survey(&farm, 4);
    views.extend(close_ups(&farm, tags::BUILDING, 4, 30.0, 6.0));
    // Along the roads, from above.
    for road in farm.roads().roads().iter().take(3) {
        let p = road.line.points();
        let (a, b) = (p[p.len() / 2].truncate(), p[p.len() / 2 + 1].truncate());
        let ground = farm.surface_height(a.x, a.y);
        let d = (b - a).normalize();
        views.push(view(look_at(a.extend(ground + 12.0) - d.extend(0.0) * 10.0, (a + d * 15.0).extend(ground))));
    }
    check_map("rural training map", &farm, &views, &[C::Grass, C::Road, C::Building, C::Sky]);
}

/// Nearest hit of a ray on the triangles of `meshes` placed at `poses`: depth along the ray.
fn cast_triangles(tris: &[[DVec3; 3]], o: DVec3, d: DVec3) -> Option<f64> {
    let mut best: Option<f64> = None;
    for [a, b, c] in tris {
        let (e1, e2) = (*b - *a, *c - *a);
        let p = d.cross(e2);
        let det = e1.dot(p);
        if det.abs() < 1e-14 {
            continue;
        }
        let s = o - *a;
        let u = s.dot(p) / det;
        let q = s.cross(e1);
        let v = d.dot(q) / det;
        let t = e2.dot(q) / det;
        if u >= 0.0 && v >= 0.0 && u + v <= 1.0 && t > 0.0 && best.is_none_or(|x| t < x) {
            best = Some(t);
        }
    }
    best
}

/// A vehicle spawned by the simulation and run for a moment.
fn spawned(vehicle: &str, air: bool, trailers: Vec<String>) -> WorldInstance {
    let sc = Scenario {
        map: MapSource::Testworld(Testworld::Flat { size: 400.0 }),
        groups: vec![GroupSpec {
            vehicle: VehicleRef::Name(vehicle.into()),
            trailers,
            spawn: if air { SpawnSpec { agl: [5.0, 5.0], ..Default::default() } } else { SpawnSpec::default() },
            ..Default::default()
        }],
        ..Default::default()
    };
    let mut world = WorldInstance::new(Arc::new(sc.compile().unwrap()), Seed::from_u64(2));
    for _ in 0..25 {
        world.step();
    }
    world
}

#[test]
fn vehicles_match_their_triangles() {
    let ctx = ctx();
    let flat = testworlds::flat(400.0);
    let ground = GpuWorld::new(ctx, &flat, WorldOptions::default()).unwrap();
    let mut r = Renderer::new(ctx);
    let cases = [
        ("sedan_like", false, vec![]),
        ("farm_tractor", false, vec!["farm_trailer".to_string()]),
        ("iris_like", true, vec![]),
        ("bo105_like", true, vec![]),
        ("quadtilt_like", true, vec![]),
        ("motorcycle_sport", false, vec![]),
    ];
    for (name, air, trailers) in cases {
        let world = spawned(name, air, trailers);
        let vehicle = &world.agent(0).vehicle;
        let rig = Rig::new(vehicle);
        let gpu = GpuRig::new(ctx, &rig);
        let mut placements = Vec::new();
        rig.place(vehicle, &mut placements);
        let pose = vehicle.pose();
        // CPU copy of the placed triangles.
        let mut tris = Vec::new();
        for (m, p) in rig.meshes.iter().zip(&placements) {
            let Placement { pose: local, scale } = *p;
            let to_world = |v: [f32; 3]| (pose * local).transform_point(glam::Vec3::from_array(v).as_dvec3() * scale);
            for t in m.indices.as_chunks::<3>().0 {
                tris.push(t.map(|i| to_world(m.positions[i as usize])));
            }
        }
        let (lo, hi) =
            tris.iter().flatten().fold((DVec3::INFINITY, DVec3::NEG_INFINITY), |(lo, hi), p| (lo.min(*p), hi.max(*p)));
        let centre = 0.5 * (lo + hi);
        let size = (hi - lo).length();
        for (k, dir) in
            [DVec3::new(-1.0, -0.6, 0.5), DVec3::new(0.7, 0.9, 0.3), DVec3::new(0.2, -0.3, 1.5)].iter().enumerate()
        {
            let v = view(look_at(centre + dir.normalize() * 0.7 * size, centre));
            let own = k == 2;
            let class = if own { SemanticClass::OwnVehicle } else { SemanticClass::Vehicle };
            let mut draws = Vec::new();
            ground.draws(&v, &mut draws);
            gpu.draws(pose, &placements, class, &mut draws);
            let frame = r.render(ctx, &v, &draws).unwrap();
            let (mut clean, mut ok, mut vehicle_px) = (0, 0, 0);
            for j in 0..frame.height {
                for i in 0..frame.width {
                    let (u, w) = (i as f64 + 0.5, j as f64 + 0.5);
                    let see = |du: f64, dw: f64| {
                        let d = v.pose.orientation * v.intrinsics.ray(u + du, w + dw);
                        // The 400 m map around the origin.
                        let floor = (d.z < 0.0)
                            .then(|| -v.pose.position.z / d.z)
                            .filter(|t| (v.pose.position + *t * d).truncate().abs().max_element() < 200.0);
                        match (cast_triangles(&tris, v.pose.position, d), floor) {
                            (Some(t), f) if f.is_none_or(|f| t < f) => (true, t),
                            (_, Some(f)) => (false, f),
                            _ => (false, 0.0),
                        }
                    };
                    let (hit, t) = see(0.0, 0.0);
                    if [(-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5)].iter().any(|&(a, b)| see(a, b).0 != hit) {
                        continue;
                    }
                    clean += 1;
                    let want = match (hit, t) {
                        (true, _) => class,
                        (false, 0.0) => SemanticClass::Sky,
                        _ => SemanticClass::Grass,
                    };
                    let got = frame.depth_at(i, j) as f64;
                    let good = frame.class_at(i, j) == want.id() && (got - t).abs() <= 1e-3 * t + 1e-3;
                    ok += usize::from(good);
                    vehicle_px += usize::from(hit);
                }
            }
            println!("{name} view {k}: {vehicle_px} vehicle pixels, {ok} of {clean} clean pixels match");
            assert!(vehicle_px > 50, "{name}: {vehicle_px} vehicle pixels");
            // Thin parts (wires, rider limbs) may fall between pixel centres.
            assert!(ok as f64 > 0.995 * clean as f64, "{name} view {k}: {ok} of {clean}");
        }
    }
}

/// Parts are placed from the simulated state: a trailer's wheels sit where the simulation
/// has them, a tiltrotor's pods turn with the mount tilts.
#[test]
fn rigs_follow_the_state() {
    let world = spawned("farm_tractor", false, vec!["farm_trailer".to_string()]);
    let vehicle = &world.agent(0).vehicle;
    let w = vehicle.as_wheeled().unwrap();
    let rig = Rig::new(vehicle);
    let mut placements = Vec::new();
    rig.place(vehicle, &mut placements);
    assert_eq!(placements.len(), rig.len());
    // The wheel meshes are the last parts, in wheel order.
    let wheels = &placements[placements.len() - w.num_wheels()..];
    let chassis = w.unit_pose(0).inverse();
    for (k, p) in wheels.iter().enumerate() {
        let want: Pose = chassis * w.wheel_pose(k);
        assert!((p.pose.pos - want.pos).length() < 1e-9, "wheel {k}");
    }

    let mut world = spawned("quadtilt_like", true, vec![]);
    let t = world.agent_mut(0).vehicle.as_tiltrotor_mut().unwrap();
    let n = t.rotor_count();
    let tilts: Vec<f64> = (0..n).map(|k| 0.3 + 0.2 * k as f64).collect();
    t.show(&autonomousim_vehicles::tiltrotor::TiltrotorDisplay {
        rotor_speeds: vec![300.0; n],
        throttles: vec![0.5; n],
        tilts: tilts.clone(),
        channels: [0.1, -0.15, 0.05],
        electric_power: 100.0,
        airspeed: 10.0,
        alpha: 0.05,
        beta: 0.0,
    });
    let vehicle = &world.agent(0).vehicle;
    let rig = Rig::new(vehicle);
    rig.place(vehicle, &mut placements);
    let pods = &placements[placements.len() - n..];
    for (k, p) in pods.iter().enumerate() {
        let axis = p.pose.rot * DVec3::Z;
        assert!((axis - DVec3::new(tilts[k].sin(), 0.0, tilts[k].cos())).length() < 1e-12, "{k}: {axis}");
    }
}
