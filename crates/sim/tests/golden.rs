//! Golden trajectories: hashes of states, observations, events and recordings over short
//! batched runs of several scenarios, so that refactors can prove they change nothing.
//!
//! Regenerate after an intended change of the simulation output with
//! `AUTONOMOUSIM_BLESS=1 cargo test -p autonomousim-sim --test golden`.

use autonomousim_sim::record::{MemorySink, Recorder, RecorderConfig, Recording};
use autonomousim_sim::{BatchSim, Scenario};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// All multirotor action modes, sensors including LiDAR and GPS, agent contacts, wind and
/// parameter randomisation.
const MODES: &str = r#"
    name = "modes"
    map = { type = "testworld", kind = "forest_patch", size = 120.0, density = 150.0, seed = 5 }
    [randomize_environment]
    wind_speed = [1.0, 4.0]
    turbulence_w20 = [5.0, 10.0]
    gust_rate = 10.0
    [[groups]]
    name = "ctbr"
    count = 3
    action_mode = "ctbr"
    sensors = [ { name = "imu", type = "imu" }, { name = "gps", type = "gps" }, { name = "lidar", type = "lidar" } ]
    obs = [ { term = "goal_rel_body" }, { term = "imu", sensor = "imu" }, { term = "lidar_log", sensor = "lidar" }, { term = "motor_speeds" }, { term = "last_action" } ]
    randomize = { mass = 0.1, k_thrust = 0.05, motor_tau = 0.2 }
    spawn = { tilt_deg = 20.0, speed = 1.0, rates = 1.0 }
    [[groups]]
    name = "velocity"
    count = 2
    vehicle = "iris_like"
    action_mode = "velocity"
    sensors = [ { name = "baro", type = "baro" }, { name = "mag", type = "mag" }, { name = "range", type = "rangefinder" } ]
    goals = { kind = "random", count = 3 }
    [[groups]]
    name = "motors"
    action_mode = "motors"
    [[groups]]
    name = "attitude"
    action_mode = "attitude"
    vehicle = "iris_like"
    [[groups]]
    name = "position"
    action_mode = "position"
    count = 2
"#;

/// A small wild map pool with LiDAR navigation, like `assets/scenarios/forest.toml`.
const FOREST: &str = r#"
    name = "forest_small"
    map = { type = "wild", seed = 3, count = 2, cache = false, config = { size = 128.0 } }
    [randomize_environment]
    wind_speed = [0.0, 4.0]
    turbulence_w20 = [0.0, 7.7]
    gust_rate = 2.0
    [[groups]]
    name = "iris"
    vehicle = "iris_like"
    action_mode = "velocity"
    spawn = { agl = [2.0, 4.0], clearance = 1.5 }
    goals = { kind = "random", count = 3, distance = [10.0, 30.0], agl = [3.0, 6.0], radius = 2.0 }
    sensors = [ { name = "lidar", type = "lidar" }, { name = "imu", type = "imu" } ]
    obs = [ { term = "goal_rel_body" }, { term = "rot6d" }, { term = "agl" }, { term = "lidar_log", sensor = "lidar" } ]
    [[groups]]
    name = "cf"
    count = 2
    action_mode = "ctbr"
    spawn = { agl = [1.0, 2.0], clearance = 1.5 }
"#;

/// Ground vehicles of every drive type and action mode on a small off-road map, with the
/// ground observation terms, LiDAR, goals on drivable ground and the ground events.
const CARS: &str = r#"
    name = "cars"
    map = { type = "wild", preset = "offroad", seed = 5, count = 2, cache = false, config = { size = 128.0 } }
    events = { ground = { stuck_time = 1.0 } }
    [[groups]]
    name = "trucks"
    count = 2
    vehicle = "offroad_4x4"
    action_mode = "vk"
    spawn = { min_separation = 6.0 }
    goals = { kind = "random", count = 2, distance = [15.0, 30.0], radius = 3.0 }
    sensors = [ { name = "lidar", type = "lidar" }, { name = "imu", type = "imu" } ]
    obs = [ { term = "goal_rel_heading" }, { term = "speed" }, { term = "sideslip" }, { term = "pitch_roll" }, { term = "wheel_speeds" }, { term = "wheel_slip" }, { term = "steering" }, { term = "gear_rpm" }, { term = "lidar_log", sensor = "lidar" } ]
    [[groups]]
    name = "sedan"
    vehicle = "sedan_like"
    action_mode = "raw"
    [[groups]]
    name = "skid"
    vehicle = "rover_skid"
    action_mode = "vw"
    [[groups]]
    name = "diff"
    vehicle = "rover_diff"
    action_mode = "per_wheel"
"#;

fn scenarios() -> Vec<(&'static str, String)> {
    let hover = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets/scenarios/hover.toml");
    vec![
        ("hover", std::fs::read_to_string(hover).unwrap()),
        ("modes", MODES.to_string()),
        ("forest", FOREST.to_string()),
        ("cars", CARS.to_string()),
    ]
}

/// Deterministic, varied actions in [−1.2, 1.2].
fn actions(step: u64, env: usize, n: usize) -> Vec<f32> {
    (0..n).map(|i| (1.3 * ((step * 7 + env as u64 * 13 + i as u64 * 3) as f32 * 0.61).sin()).clamp(-1.2, 1.2)).collect()
}

/// Hash of a 120-step run of three worlds (with a partial reset and a full reset), recording
/// world 1.
fn run(toml: &str) -> String {
    let sc = Arc::new(Scenario::from_toml(toml).unwrap().compile().unwrap());
    let mut b = BatchSim::from_compiled(sc.clone(), 3, 11, 2).unwrap();
    let sink = Arc::new(Mutex::new(MemorySink::default()));
    let config = RecorderConfig { state_hz: 50, lidar: true };
    b.attach_recorder(1, Recorder::new(Box::new(sink.clone()), config));
    let mut h = blake3::Hasher::new();
    let outputs = |b: &BatchSim, h: &mut blake3::Hasher| {
        for g in 0..sc.groups.len() {
            for x in b.obs(g) {
                h.update(&x.to_le_bytes());
            }
            for x in b.state(g) {
                h.update(&x.to_le_bytes());
            }
            for x in b.events(g) {
                h.update(&x.to_le_bytes());
            }
        }
        for i in 0..b.num_envs() {
            h.update(&b.world(i).state_hash());
        }
    };
    outputs(&b, &mut h);
    for k in 0..120u64 {
        if k == 50 {
            b.reset(Some(&[true, false, true]), None);
        }
        if k == 90 {
            b.reset(None, Some(&[5, 6, 7]));
        }
        let acts: Vec<Vec<f32>> = sc
            .groups
            .iter()
            .enumerate()
            .map(|(g, grp)| {
                (0..b.num_envs()).flat_map(|e| actions(k, e * 10 + g, grp.spec.count * grp.act_dim())).collect()
            })
            .collect();
        let refs: Vec<&[f32]> = acts.iter().map(|a| a.as_slice()).collect();
        b.step(&refs);
        outputs(&b, &mut h);
    }
    b.detach_recorder(1).unwrap().finish().unwrap();
    let sink = sink.lock().unwrap();
    for t in &sink.topics {
        h.update(t.as_bytes());
    }
    for (c, t, data) in &sink.messages {
        h.update(&c.to_le_bytes());
        h.update(&t.to_le_bytes());
        h.update(data);
    }
    h.finalize().to_string()
}

fn golden_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/golden_trajectories.toml")
}

#[test]
fn golden_trajectories() {
    let hashes: BTreeMap<String, String> =
        scenarios().into_iter().map(|(name, toml)| (name.to_string(), run(&toml))).collect();
    if std::env::var_os("AUTONOMOUSIM_BLESS").is_some() {
        let text = format!(
            "# Golden trajectory hashes (crates/sim/tests/golden.rs); bless with AUTONOMOUSIM_BLESS=1.\n{}",
            toml::to_string(&hashes).unwrap()
        );
        std::fs::write(golden_path(), text).unwrap();
        return;
    }
    let text = std::fs::read_to_string(golden_path()).expect("fixtures/golden_trajectories.toml");
    let golden: BTreeMap<String, String> = toml::from_str(&text).unwrap();
    assert_eq!(hashes, golden, "trajectories changed (bless if intended)");
}

/// Records one world of `hover.toml` for 1.5 s (with a reset) into MCAP bytes.
fn record_hover() -> Vec<u8> {
    let hover = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets/scenarios/hover.toml");
    let sc = Arc::new(Scenario::from_toml(&std::fs::read_to_string(hover).unwrap()).unwrap().compile().unwrap());
    let mut b = BatchSim::from_compiled(sc.clone(), 1, 4, 1).unwrap();
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("golden_hover.mcap");
    b.attach_recorder(0, Recorder::create(&path, RecorderConfig { state_hz: 50, lidar: true }).unwrap());
    for k in 0..75u64 {
        if k == 50 {
            b.reset(None, None);
        }
        b.step(&[&actions(k, 0, sc.groups[0].act_dim())]);
    }
    b.detach_recorder(0).unwrap().finish().unwrap();
    std::fs::read(path).unwrap()
}

fn recording_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/recordings/hover.mcap")
}

/// A recording made before a refactor is still read, and the same run records the same messages.
#[test]
fn golden_recording() {
    let bytes = record_hover();
    if std::env::var_os("AUTONOMOUSIM_BLESS").is_some() {
        std::fs::create_dir_all(recording_path().parent().unwrap()).unwrap();
        std::fs::write(recording_path(), &bytes).unwrap();
        return;
    }
    let old = std::fs::read(recording_path()).expect("fixtures/recordings/hover.mcap");
    let rec = Recording::from_bytes(&old).unwrap();
    assert_eq!((rec.agents.len(), rec.episodes.len(), rec.episodes[0].states[0].len()), (1, 2, 51));
    rec.compile().unwrap();
    // The MCAP summary section is written in hash-map order, so compare the messages.
    let messages = |b: &[u8]| -> Vec<(String, u64, Vec<u8>)> {
        mcap::MessageStream::new(b)
            .unwrap()
            .map(|m| {
                let m = m.unwrap();
                (m.channel.topic.clone(), m.log_time, m.data.into_owned())
            })
            .collect()
    };
    assert!(messages(&old) == messages(&bytes), "recorded messages changed (bless if intended)");
}
