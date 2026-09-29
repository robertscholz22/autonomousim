//! Camera sensors in worlds and batches, rendered on lavapipe (Mesa's software rasterizer), so
//! that the images do not depend on the machine.

use autonomousim_core::math::Pose;
use autonomousim_core::rng::Seed;
use autonomousim_render::{AdapterChoice, SemanticClass};
use autonomousim_sensors::{CameraImage, Sensor};
use autonomousim_sim::camera::{self, Cameras};
use autonomousim_sim::{BatchSim, CompiledScenario, Scenario, WorldInstance};
use glam::{DQuat, DVec3};
use std::sync::Arc;

fn gpu() {
    camera::use_adapter(&AdapterChoice::Software);
    assert!(camera::gpu().unwrap().is_software(), "the tests render on lavapipe");
}

fn compile(toml: &str) -> Arc<CompiledScenario> {
    Arc::new(Scenario::from_toml(toml).unwrap().compile().unwrap())
}

/// Drones over a forest with a down camera (25 Hz, policy at 50 Hz), a second one delayed by
/// two frames and a noisy forward one.
const FOREST: &str = r#"
    name = "cameras"
    map = { type = "testworld", kind = "forest_patch", size = 120.0, density = 150.0, seed = 5 }
    [[groups]]
    name = "drones"
    count = 3
    vehicle = "iris_like"
    action_mode = "velocity"
    spawn = { agl = [4.0, 12.0] }
    sensors = [
        { name = "down", type = "camera", width = 32, height = 24, fov_deg = 80.0, rate_hz = 25, mount = { rotation = [0.0, 1.5707963267948966, 0.0] } },
        { name = "late", type = "camera", width = 32, height = 24, fov_deg = 80.0, rate_hz = 25, latency = 2, mount = { rotation = [0.0, 1.5707963267948966, 0.0] } },
        { name = "front", type = "camera", width = 32, height = 24, rate_hz = 50, noise = { pixel = 0.02, depth = 0.001, exposure = 0.1 } },
    ]
    obs = [
        { term = "lin_vel_body" },
        { term = "camera", sensor = "down", output = "rgb" },
        { term = "camera", sensor = "down", output = "depth", range = 20.0 },
        { term = "camera", sensor = "front", output = "semantic" },
    ]
"#;

fn actions(step: u64, env: usize, n: usize) -> Vec<f32> {
    (0..n).map(|i| 0.8 * ((step * 7 + env as u64 * 13 + i as u64 * 3) as f32 * 0.37).sin()).collect()
}

fn step(b: &mut BatchSim, k: u64) {
    let n = b.num_envs();
    let g = &b.scenario().groups[0];
    let a: Vec<f32> = (0..n).flat_map(|i| actions(k, i, g.spec.count * g.act_dim())).collect();
    b.step(&[&a]);
}

fn camera_image(w: &WorldInstance, agent: usize, sensor: usize) -> Option<(u64, CameraImage)> {
    match &w.agent(agent).sensors[sensor] {
        Sensor::Camera(c) => c.latest().map(|f| (f.tick, f.value.clone())),
        _ => panic!("not a camera"),
    }
}

#[test]
fn images_are_deterministic_and_independent_of_the_batch() {
    gpu();
    let sc = compile(FOREST);
    assert_eq!(sc.groups[0].obs.image_shape(), Some([24, 32, 5]));
    assert_eq!(sc.groups[0].obs_dim(), 3);
    let mut a = BatchSim::from_compiled(sc.clone(), 1, 7, 1).unwrap();
    let mut b = BatchSim::from_compiled(sc.clone(), 1, 7, 1).unwrap();
    let mut wide = BatchSim::from_compiled(sc, 64, 7, 4).unwrap();
    let len = a.images(0).len();
    assert_eq!(len, 3 * 24 * 32 * 5);
    assert_eq!(wide.images(0).len(), 64 * len);
    let mut changed = 0;
    for k in 0..12 {
        let before = a.images(0).to_vec();
        for s in [&mut a, &mut b, &mut wide] {
            step(s, k);
        }
        assert_eq!(a.images(0), b.images(0), "step {k}");
        assert_eq!(a.images(0), &wide.images(0)[..len], "step {k}");
        assert_eq!(a.world(0).state_hash(), wide.world(0).state_hash(), "step {k}");
        changed += usize::from(before != a.images(0));
    }
    assert!(changed >= 6, "the images change as the drones move ({changed})");
    // The worlds differ, and so do their images.
    assert_ne!(&wide.images(0)[..len], &wide.images(0)[len..2 * len]);
    // Resets render a first frame; the same seed gives the same images.
    a.reset(None, Some(&[3]));
    b.reset(None, Some(&[3]));
    assert_eq!(a.images(0), b.images(0));
    assert!(a.images(0).iter().any(|&x| x != 0));
}

#[test]
fn frames_arrive_at_the_camera_rate_with_exact_latency() {
    gpu();
    let sc = compile(FOREST);
    let mut b = BatchSim::from_compiled(sc, 1, 3, 1).unwrap();
    // Every frame of the undelayed camera by capture tick.
    let mut frames = std::collections::BTreeMap::new();
    let record = |b: &BatchSim, frames: &mut std::collections::BTreeMap<u64, CameraImage>| {
        let (tick, img) = camera_image(b.world(0), 1, 0).unwrap();
        frames.insert(tick, img);
    };
    record(&b, &mut frames);
    // After the reset every delay slot holds the first frame.
    assert_eq!(camera_image(b.world(0), 1, 1).unwrap().0, 0);
    for k in 0..16 {
        step(&mut b, k);
        let w = b.world(0);
        let tick = w.clock().tick;
        record(&b, &mut frames);
        // The down cameras capture every other policy step (20 physics ticks at 500 Hz).
        let (down, _) = camera_image(w, 1, 0).unwrap();
        assert_eq!(down, tick - tick % 20, "step {k}");
        // The delayed camera shows the frame captured two frames earlier, bit for bit.
        let (late, img) = camera_image(w, 1, 1).unwrap();
        assert_eq!(late, down.saturating_sub(40), "step {k}");
        assert_eq!(&img, &frames[&late], "step {k}");
        // The front camera renders every policy step.
        assert_eq!(camera_image(w, 1, 2).unwrap().0, tick, "step {k}");
    }
    assert_eq!(frames.len(), 9);
}

#[test]
fn noise_is_seeded() {
    gpu();
    let sc = compile(FOREST);
    let mut a = WorldInstance::new(sc.clone(), Seed::from_u64(4));
    let mut b = WorldInstance::new(sc.clone(), Seed::from_u64(4));
    let mut cams = Cameras::new(camera::gpu().unwrap(), &sc);
    cams.update(&mut a).unwrap();
    cams.update(&mut b).unwrap();
    let (_, na) = camera_image(&a, 0, 2).unwrap();
    assert_eq!(na, camera_image(&b, 0, 2).unwrap().1);
    // The same frame without noise differs, but not by much.
    let mut clean = sc.spec.clone();
    let ours = &mut clean.groups[0].sensors[2].config;
    let autonomousim_sensors::SensorConfig::Camera(c) = ours else { panic!() };
    c.noise = Default::default();
    let clean = Arc::new(clean.compile().unwrap());
    let mut c = WorldInstance::new(clean.clone(), Seed::from_u64(4));
    Cameras::new(camera::gpu().unwrap(), &clean).update(&mut c).unwrap();
    let (_, nc) = camera_image(&c, 0, 2).unwrap();
    assert_ne!(na.rgb, nc.rgb);
    assert_eq!(na.class, nc.class);
    let mean = na.rgb.iter().zip(&nc.rgb).map(|(x, y)| (f64::from(*x) - f64::from(*y)).abs()).sum::<f64>()
        / na.rgb.len() as f64;
    assert!(mean > 1.0 && mean < 40.0, "mean pixel change {mean}");
}

/// A drone with a down camera 10 m over flat grass, and another drone 2 m below it.
#[test]
fn a_down_camera_sees_the_ground_and_the_drone_below() {
    gpu();
    let sc = compile(
        r#"
        name = "down"
        map = { type = "testworld", kind = "flat", size = 200.0 }
        [[groups]]
        name = "drones"
        count = 2
        vehicle = "iris_like"
        sensors = [ { name = "down", type = "camera", width = 33, height = 33, fov_deg = 60.0, rate_hz = 50, mount = { position = [0.0, 0.0, -0.1], rotation = [0.0, 1.5707963267948966, 0.0] } } ]
        obs = [
            { term = "camera", sensor = "down", output = "depth", range = 25.0 },
            { term = "camera", sensor = "down", output = "semantic" },
        ]
        "#,
    );
    let mut w = WorldInstance::new(sc.clone(), Seed::from_u64(1));
    let mut cams = Cameras::new(camera::gpu().unwrap(), &sc);
    let level = |x: f64, z: f64| Pose::new(DVec3::new(x, 0.0, z), DQuat::IDENTITY);
    w.place_agent(0, level(0.0, 10.0), DVec3::ZERO, DVec3::ZERO);
    w.place_agent(1, level(40.0, 3.0), DVec3::ZERO, DVec3::ZERO);
    cams.update(&mut w).unwrap();
    let (_, img) = camera_image(&w, 0, 0).unwrap();
    let centre = 16 * 33 + 16;
    assert!((f64::from(img.depth[centre]) - 9.9).abs() < 1e-4, "depth {}", img.depth[centre]);
    assert_eq!(img.class[centre], SemanticClass::Grass.id());
    let own = SemanticClass::OwnVehicle.id();
    assert!(img.class.iter().all(|&c| c == SemanticClass::Grass.id() || c == own));
    // The observation: depth over 25 m, then the class.
    let mut obs = vec![0u8; 2 * 33 * 33 * 2];
    w.observe_images(0, &mut obs);
    assert_eq!(obs[2 * centre], (9.9f64 / 25.0 * 255.0).round() as u8);
    assert_eq!(obs[2 * centre + 1], SemanticClass::Grass.id());

    // The second drone right below (its hub is 8 cm long, the arms are thin and the rotor
    // discs are not drawn, so it covers few pixels).
    w.place_agent(1, level(0.0, 8.0), DVec3::ZERO, DVec3::ZERO);
    cams.update(&mut w).unwrap();
    let (_, img) = camera_image(&w, 0, 0).unwrap();
    let other = img.class.iter().filter(|&&c| c == SemanticClass::Vehicle.id()).count();
    assert!(other >= 5, "{other} pixels of the drone below");
    assert_eq!(img.class[centre], SemanticClass::Vehicle.id());
    assert!(img.depth[centre] > 1.7 && img.depth[centre] < 1.9, "depth {}", img.depth[centre]);
    // Its own camera looks down at the grass past nothing of its own.
    let (_, below) = camera_image(&w, 1, 0).unwrap();
    assert!(!below.class.contains(&SemanticClass::Vehicle.id()));
}

#[test]
fn invalid_camera_setups_are_rejected() {
    let base = |sensor: &str, obs: &str| {
        format!(
            r#"
            name = "bad"
            map = {{ type = "testworld", kind = "flat", size = 50.0 }}
            [[groups]]
            name = "d"
            sensors = [ {{ name = "c", type = "camera", {sensor} }}, {{ name = "d", type = "camera", width = 16 }} ]
            obs = [ {obs} ]
            "#
        )
    };
    let ok = base("rate_hz = 25", r#"{ term = "camera", sensor = "c", output = "rgb" }"#);
    assert!(Scenario::from_toml(&ok).unwrap().compile().is_ok());
    for (sensor, obs) in [
        // Faster than the policy (50 Hz), or not a whole number of policy steps.
        ("rate_hz = 100", r#"{ term = "camera", sensor = "c", output = "rgb" }"#),
        ("rate_hz = 20", r#"{ term = "camera", sensor = "c", output = "rgb" }"#),
        // No output; an output on another term; a range on a colour image; scaled images.
        ("rate_hz = 25", r#"{ term = "camera", sensor = "c" }"#),
        ("rate_hz = 25", r#"{ term = "height", output = "rgb" }"#),
        ("rate_hz = 25", r#"{ term = "camera", sensor = "c", output = "rgb", range = 5.0 }"#),
        ("rate_hz = 25", r#"{ term = "camera", sensor = "c", output = "rgb", scale = 2.0 }"#),
        // Two image sizes in one group.
        (
            "rate_hz = 25",
            r#"{ term = "camera", sensor = "c", output = "rgb" }, { term = "camera", sensor = "d", output = "depth" }"#,
        ),
    ] {
        let toml = base(sensor, obs);
        assert!(Scenario::from_toml(&toml).unwrap().compile().is_err(), "{sensor} / {obs}");
    }
}
