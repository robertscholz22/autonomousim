//! World instances, batches, events and recording.

use autonomousim_control::multirotor::{Setpoint, YawCommand};
use autonomousim_core::rng::Seed;
use autonomousim_sensors::Sensor;
use autonomousim_sensors::lidar::ReturnKind;
use autonomousim_sim::record::{MemorySink, Recorder, RecorderConfig, Recording};
use autonomousim_sim::{BatchSim, CompiledScenario, Events, STATE_DIM, Scenario, WorldInstance};
use glam::DVec3;
use std::sync::{Arc, Mutex};

fn compile(toml: &str) -> Arc<CompiledScenario> {
    Arc::new(Scenario::from_toml(toml).unwrap().compile().unwrap())
}

/// Two groups with different vehicles and action modes, sensors including a LiDAR that sees
/// the other agents, wind with turbulence and parameter randomisation.
const MIXED: &str = r#"
    name = "mixed"
    map = { type = "testworld", kind = "forest_patch", size = 120.0, density = 150.0, seed = 5 }
    [randomize_environment]
    wind_speed = [1.0, 4.0]
    turbulence_w20 = [5.0, 10.0]
    gust_rate = 10.0
    [[groups]]
    name = "cf"
    count = 3
    action_mode = "ctbr"
    sensors = [ { name = "imu", type = "imu" }, { name = "gps", type = "gps" }, { name = "lidar", type = "lidar" } ]
    obs = [ { term = "goal_rel_body" }, { term = "imu", sensor = "imu" }, { term = "lidar_log", sensor = "lidar" } ]
    randomize = { mass = 0.1, k_thrust = 0.05, motor_tau = 0.2 }
    spawn = { tilt_deg = 20.0, speed = 1.0, rates = 1.0 }
    [[groups]]
    name = "iris"
    count = 2
    vehicle = "iris_like"
    action_mode = "velocity"
    sensors = [ { name = "baro", type = "baro" }, { name = "mag", type = "mag" } ]
    goals = { kind = "random", count = 3 }
"#;

/// Deterministic, varied actions in [−1, 1] (a few out of range to exercise clipping).
fn actions(step: u64, env: usize, n: usize) -> Vec<f32> {
    (0..n).map(|i| (1.3 * ((step * 7 + env as u64 * 13 + i as u64 * 3) as f32 * 0.61).sin()).clamp(-1.2, 1.2)).collect()
}

fn step_mixed(w: &mut WorldInstance, step: u64, env: usize) {
    for g in 0..w.scenario().groups.len() {
        let grp = &w.scenario().groups[g];
        let a = actions(step, env * 10 + g, grp.spec.count * grp.act_dim());
        w.set_actions(g, &a);
    }
    w.step();
}

#[test]
fn same_seed_same_trajectory() {
    let sc = compile(MIXED);
    let mut a = WorldInstance::new(sc.clone(), Seed::from_u64(9));
    let mut b = WorldInstance::new(sc.clone(), Seed::from_u64(9));
    let c = WorldInstance::new(sc, Seed::from_u64(10));
    assert_eq!(a.state_hash(), b.state_hash());
    assert_ne!(a.state_hash(), c.state_hash());
    for k in 0..150 {
        step_mixed(&mut a, k, 0);
        step_mixed(&mut b, k, 0);
        assert_eq!(a.state_hash(), b.state_hash(), "step {k}");
    }
    // LiDAR scans were taken, and something moved.
    let scans = a.agents()[..3].iter().filter(|x| matches!(&x.sensors[2], Sensor::Lidar(l) if l.latest().is_some()));
    assert_eq!(scans.count(), 3);
    // Resets continue in lockstep, with a new episode.
    let first = a.agent(0).spawn;
    a.reset(None);
    b.reset(None);
    assert_eq!(a.state_hash(), b.state_hash());
    assert_ne!(a.agent(0).spawn.pos, first.pos);
}

#[test]
fn reset_with_seed_reproduces_and_snapshots_restore() {
    let sc = compile(MIXED);
    let mut w = WorldInstance::new(sc, Seed::from_u64(1));
    let run = |w: &mut WorldInstance, from: u64, n: u64| {
        for k in from..from + n {
            step_mixed(w, k, 0);
        }
        w.state_hash()
    };
    w.reset(Some(42));
    let h0 = w.state_hash();
    let h1 = run(&mut w, 0, 40);
    // Another episode in between does not matter.
    w.reset(None);
    run(&mut w, 0, 5);
    w.reset(Some(42));
    assert_eq!(w.state_hash(), h0);
    assert_eq!(run(&mut w, 0, 40), h1);

    // Snapshot mid-episode, run on, restore and run the same actions again.
    let snap = w.snapshot();
    let h2 = run(&mut w, 40, 60);
    w.restore(&snap);
    assert_eq!(w.state_hash(), snap.state_hash());
    assert_eq!(run(&mut w, 40, 60), h2);
}

fn batch_actions(b: &BatchSim, step: u64) -> Vec<Vec<f32>> {
    let sc = b.scenario();
    sc.groups
        .iter()
        .enumerate()
        .map(|(g, grp)| {
            (0..b.num_envs()).flat_map(|e| actions(step, e * 10 + g, grp.spec.count * grp.act_dim())).collect()
        })
        .collect()
}

fn step_batch(b: &mut BatchSim, step: u64) {
    let acts = batch_actions(b, step);
    let refs: Vec<&[f32]> = acts.iter().map(|a| a.as_slice()).collect();
    b.step(&refs);
}

#[test]
fn batch_worlds_do_not_depend_on_batch_size_or_threads() {
    let sc = compile(MIXED);
    let mut one = BatchSim::from_compiled(sc.clone(), 1, 3, 1).unwrap();
    let mut many = BatchSim::from_compiled(sc.clone(), 12, 3, 4).unwrap();
    let mut serial = BatchSim::from_compiled(sc, 12, 3, 1).unwrap();
    assert_eq!(many.num_threads(), 4);
    for k in 0..60 {
        step_batch(&mut one, k);
        step_batch(&mut many, k);
        step_batch(&mut serial, k);
        assert_eq!(one.world(0).state_hash(), many.world(0).state_hash(), "step {k}");
    }
    for g in 0..2 {
        let n = one.obs(g).len();
        assert_eq!(one.obs(g), &many.obs(g)[..n]);
        assert_eq!(one.state(g), &many.state(g)[..one.state(g).len()]);
        assert_eq!(many.obs(g), serial.obs(g));
        assert_eq!(many.state(g), serial.state(g));
        assert_eq!(many.events(g), serial.events(g));
    }
    for e in 0..12 {
        assert_eq!(many.world(e).state_hash(), serial.world(e).state_hash());
    }
    // Different worlds differ.
    assert_ne!(many.world(0).state_hash(), many.world(1).state_hash());

    // Masked resets: only the selected worlds restart, and seeded ones reproduce.
    let before: Vec<_> = (0..12).map(|e| many.world(e).state_hash()).collect();
    let mask: Vec<bool> = (0..12).map(|e| e % 3 == 0).collect();
    let seeds: Vec<u64> = (0..12).map(|e| 100 + e as u64).collect();
    many.reset(Some(&mask), Some(&seeds));
    for e in 0..12 {
        assert_eq!(many.world(e).state_hash() == before[e], !mask[e], "world {e}");
    }
    let fresh = BatchSim::from_compiled(many.scenario().clone(), 1, 0, 1).map(|mut b| {
        b.reset(None, Some(&[103]));
        b.world(0).state_hash()
    });
    assert_eq!(fresh.unwrap(), many.world(3).state_hash());
    // The batch arrays follow the reset.
    let dim = many.scenario().groups[0].obs_dim() * 3;
    let mut obs = vec![0.0; dim];
    many.world(3).observe(0, &mut obs);
    assert_eq!(&many.obs(0)[3 * dim..4 * dim], &obs[..]);
    assert!(many.events(0)[9..12].iter().all(|&e| e == 0));
}

#[test]
fn parallel_agents_match_serial() {
    // 40 agents: the per-agent phases run in parallel inside the world.
    let sc = compile(
        r#"
        map = { type = "testworld", kind = "flat", size = 100.0 }
        [randomize_environment]
        turbulence_w20 = [8.0, 8.0]
        [[groups]]
        count = 40
        spawn = { layout = { type = "grid", spacing = 0.3 }, agl = [2.0, 2.0] }
        sensors = [ { name = "lidar", type = "lidar" } ]
        "#,
    );
    let mut a = BatchSim::from_compiled(sc.clone(), 1, 0, 1).unwrap();
    let mut b = BatchSim::from_compiled(sc, 1, 0, 6).unwrap();
    for k in 0..40 {
        step_batch(&mut a, k);
        step_batch(&mut b, k);
        assert_eq!(a.world(0).state_hash(), b.world(0).state_hash(), "step {k}");
    }
    // Neighbours 0.3 m apart in random ctbr: some collided with each other.
    assert!(a.world(0).agents().iter().any(|x| x.disabled));
}

#[test]
fn swarm_of_128_holds_position() {
    let sc = compile(
        r#"
        name = "swarm"
        map = { type = "testworld", kind = "flat", size = 200.0 }
        environment = { wind = { mean = [2.0, 1.0] } }
        [[groups]]
        name = "swarm"
        count = 128
        spawn = { layout = { type = "grid", spacing = 1.5 }, agl = [2.0, 2.0], yaw_deg = [0.0, 90.0] }
        "#,
    );
    let mut w = WorldInstance::new(sc, Seed::from_u64(0));
    let start: Vec<DVec3> = w.agents().iter().map(|a| a.vehicle.position()).collect();
    // No actions: every agent holds its spawn position (10 s).
    for _ in 0..500 {
        w.step();
        let bad = w.agents().iter().find(|a| !a.events.is_empty());
        assert!(bad.is_none(), "{:?} at {}", bad.map(|a| (a.id, a.events)), w.time());
    }
    let worst = w.agents().iter().zip(&start).map(|(a, p)| a.vehicle.position().distance(*p)).fold(0.0, f64::max);
    let fastest = w.agents().iter().map(|a| a.vehicle.lin_vel_world().length()).fold(0.0, f64::max);
    assert!(worst < 0.05 && fastest < 0.02, "worst drift {worst} m, speed {fastest} m/s");
    let mut state = vec![0.0; 128 * STATE_DIM];
    w.write_state(0, &mut state);
    for (row, p) in state.as_chunks::<STATE_DIM>().0.iter().zip(&start) {
        // The goal is the spawn position.
        assert_eq!(&row[13..16], &p.to_array());
    }
}

#[test]
fn events_are_raised() {
    let run = |toml: &str, steps: usize, action: Option<[f32; 4]>| {
        let mut w = WorldInstance::new(compile(toml), Seed::from_u64(0));
        let mut seen = Events::NONE;
        for _ in 0..steps {
            if let Some(a) = action {
                w.set_action(0, &a);
            }
            w.step();
            seen |= w.agent(0).events;
        }
        (w, seen)
    };
    let flat = r#"map = { type = "testworld", kind = "flat", size = 100.0 }"#;

    // Motors off from 3 m: hits the ground at ~7.7 m/s.
    let (w, e) = run(&format!("{flat}\n[[groups]]\nspawn = {{ agl = [3.0, 3.0] }}"), 60, Some([0.0, 0.0, 0.0, -1.0]));
    assert!(e.contains(Events::CRASH_TERRAIN) && e.contains(Events::DISABLED), "{e:?}");
    assert!(w.agent(0).disabled && w.agent(0).events == Events::DISABLED);

    // Resting on the ground with no thrust: landed, no crash.
    let (w, e) = run(
        &format!("{flat}\n[[groups]]\nspawn = {{ on_ground = true, motors = \"idle\" }}"),
        50,
        Some([0.0, 0.0, 0.0, -1.0]),
    );
    let last = w.agent(0).events;
    assert!(last.contains(Events::LANDED | Events::GROUND_CONTACT) && !e.is_terminal(), "{e:?} {last:?}");

    // Dropped into the lake.
    let lake = r#"
        map = { type = "testworld", kind = "lake", size = 100.0, depth = 3.0, water_level = -1.0 }
        [[groups]]
        spawn = { region = [[-2.0, -2.0], [2.0, 2.0]], margin = 0.0, avoid_water = false, agl = [1.0, 1.0] }
    "#;
    let (_, e) = run(lake, 50, Some([0.0, 0.0, 0.0, -1.0]));
    assert!(e.contains(Events::WATER) && !e.contains(Events::CRASH_TERRAIN), "{e:?}");

    // Above the ceiling from the start.
    let (_, e) =
        run(&format!("{flat}\nevents = {{ max_agl = 1.5 }}\n[[groups]]\nspawn = {{ agl = [2.0, 2.0] }}"), 1, None);
    assert!(e.contains(Events::OUT_OF_BOUNDS), "{e:?}");
    // Fly out of the map sideways.
    let (_, e) = run(
        &format!(
            "{flat}\nevents = {{ bounds_margin = 45.0 }}\n[[groups]]\naction_mode = \"velocity\"\nspawn = {{ region = [[0.0, 0.0], [0.0, 0.0]], margin = 0.0 }}"
        ),
        200,
        Some([1.0, 0.0, 0.0, 0.0]),
    );
    assert!(e.contains(Events::OUT_OF_BOUNDS), "{e:?}");

    // Two agents spawned inside each other.
    let (w, _) = run(
        &format!("{flat}\n[[groups]]\ncount = 2\nspawn = {{ layout = {{ type = \"grid\", spacing = 0.03 }} }}"),
        1,
        None,
    );
    assert!(w.agents().iter().all(|a| a.events.contains(Events::CRASH_AGENT)));

    // A corrupted state.
    let mut w = WorldInstance::new(compile(flat), Seed::from_u64(0));
    w.agent_mut(0).vehicle.state_mut().v[0] = f64::NAN;
    w.step();
    assert!(w.agent(0).events.contains(Events::NAN | Events::DISABLED));
    let mut obs = vec![0.0; w.scenario().groups[0].obs_dim()];
    w.observe(0, &mut obs);
    assert!(obs.iter().all(|x| x.is_finite()));
}

#[test]
fn lidar_sees_other_agents() {
    let sc = compile(
        r#"
        map = { type = "testworld", kind = "flat", size = 100.0 }
        [[groups]]
        count = 2
        vehicle = "iris_like"
        spawn = { layout = { type = "grid", spacing = 3.0 }, agl = [2.0, 2.0], yaw_deg = [0.0, 0.0] }
        sensors = [ { name = "lidar", type = "lidar", noise = 0.0, pattern = { type = "rings", elevations = [0.0], azimuths = 1, azimuth_fov = 360.0 } } ]
        "#,
    );
    let mut w = WorldInstance::new(sc, Seed::from_u64(0));
    for _ in 0..6 {
        w.step();
    }
    let scan = |i: usize| match &w.agent(i).sensors[0] {
        Sensor::Lidar(l) => l.latest().unwrap().clone(),
        _ => unreachable!(),
    };
    // Agent 0 looks along +x at agent 1; agent 1 looks into the empty distance.
    let (a, b) = (scan(0), scan(1));
    assert_eq!(a.kinds[0], ReturnKind::Agent);
    let r = f64::from(a.ranges[0]);
    let radius = w.shapes()[1].radius;
    assert!(r > 3.0 - radius && r < 3.0, "{r}");
    assert!(b.ranges[0].is_infinite() && b.kinds[0] == ReturnKind::None);
}

#[test]
fn observation_and_state_layout() {
    let sc = compile(r#"map = { type = "testworld", kind = "flat", size = 100.0 }"#);
    let g = &sc.groups[0];
    assert_eq!((g.act_dim(), g.obs_dim()), (4, 19));
    let names = g.obs.layout();
    assert_eq!(names[0], ("goal_rel_world".to_string(), 0, 3));
    assert_eq!(names[4], ("last_action".to_string(), 15, 4));

    let mut w = WorldInstance::new(sc, Seed::from_u64(0));
    w.set_action(0, &[0.5, f32::NAN, 7.0, -0.25]);
    let mut obs = vec![0.0; 19];
    w.observe(0, &mut obs);
    assert_eq!(&obs[15..19], &[0.5, 0.0, 1.0, -0.25]);
    // Goal at the spawn: zero position error, identity-ish attitude.
    assert!(obs[..3].iter().all(|x| x.abs() < 1e-6));
    let mut state = vec![0.0; STATE_DIM];
    w.write_state(0, &mut state);
    let a = w.agent(0);
    assert_eq!(&state[..3], &a.vehicle.position().to_array());
    assert_eq!(state[18], 0.0);
    assert!((state[17] - (a.vehicle.position().z - 0.0)).abs() < 1e-12);
    // Over flat ground the nearest surface is straight below.
    assert!((state[19] - state[17]).abs() < 1e-9, "{} {}", state[19], state[17]);
}

#[test]
fn goals_advance_within_the_radius() {
    let toml = |radius: f64| {
        format!(
            r#"
            map = {{ type = "testworld", kind = "flat", size = 100.0 }}
            [[groups]]
            vehicle = "iris_like"
            spawn = {{ agl = [2.0, 2.0] }}
            goals = {{ kind = "random", count = 3, distance = [6.0, 6.0], agl = [2.0, 2.0], radius = {radius} }}
            "#
        )
    };
    // Fly to the current goal with the position controller until the last one is reached.
    let fly = |w: &mut WorldInstance| {
        let mut reached = Vec::new();
        for step in 0..1000 {
            let goal = w.agent(0).goal();
            w.set_command(0, Setpoint::Position { position: goal.position, yaw: YawCommand::Rate(0.0) });
            w.step();
            let e = w.agent(0).events;
            if e.contains(Events::GOAL_REACHED) {
                reached.push((step, e.contains(Events::FINISHED)));
            }
            if w.agent(0).finished() {
                break;
            }
        }
        reached
    };
    let mut w = WorldInstance::new(compile(&toml(0.5)), Seed::from_u64(3));
    let reached = fly(&mut w);
    assert_eq!(reached.iter().map(|r| r.1).collect::<Vec<_>>(), [false, false, true], "{reached:?}");
    let a = w.agent(0);
    assert_eq!(a.goal_index, 3);
    assert_eq!(a.goal(), a.goals[2]);
    assert!(a.vehicle.position().distance(a.goals[2].position) <= 0.5);
    let mut state = vec![0.0; STATE_DIM];
    w.write_state(0, &mut state);
    assert_eq!(state[18], 3.0);
    // No more events after the last goal.
    w.step();
    assert!(!w.agent(0).events.intersects(Events::GOAL_REACHED | Events::FINISHED));

    // Without a radius the goals only advance explicitly.
    let mut w = WorldInstance::new(compile(&toml(0.0)), Seed::from_u64(3));
    assert!(fly(&mut w).is_empty());
    assert_eq!(w.agent(0).goal_index, 0);
    assert!(w.advance_goal(0) && w.advance_goal(0) && !w.advance_goal(0));
}

#[test]
fn recording_round_trips_through_mcap() {
    let sc = compile(MIXED);
    let path = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("roundtrip.mcap");
    let mut recorded = BatchSim::from_compiled(sc.clone(), 2, 7, 2).unwrap();
    let mut plain = BatchSim::from_compiled(sc.clone(), 2, 7, 2).unwrap();
    let config = RecorderConfig { state_hz: 50, lidar: true };
    assert!(recorded.attach_recorder(1, Recorder::create(&path, config.clone()).unwrap()).is_none());
    // 1 s, a reset, 0.5 s.
    for k in 0..75 {
        if k == 50 {
            recorded.reset(None, None);
            plain.reset(None, None);
        }
        step_batch(&mut recorded, k);
        step_batch(&mut plain, k);
    }
    // Recording does not change the simulation.
    assert_eq!(recorded.world(1).state_hash(), plain.world(1).state_hash());
    recorded.detach_recorder(1).unwrap().finish().unwrap();

    let bytes = std::fs::read(&path).unwrap();
    let mut counts = std::collections::BTreeMap::<String, usize>::new();
    let mut meta = None;
    let mut last_time = 0;
    for m in mcap::MessageStream::new(&bytes).unwrap() {
        let m = m.unwrap();
        assert_eq!(m.channel.message_encoding, "json");
        assert!(m.log_time >= last_time || m.channel.topic.ends_with("lidar"));
        last_time = last_time.max(m.log_time);
        let v: serde_json::Value = serde_json::from_slice(&m.data).unwrap();
        if m.channel.topic == "/meta" {
            meta = Some(v);
        }
        *counts.entry(m.channel.topic.clone()).or_default() += 1;
    }
    let meta = meta.unwrap();
    let spec: Scenario = serde_json::from_value(meta["scenario"].clone()).unwrap();
    assert_eq!(spec, sc.spec);
    assert_eq!(meta["agents"].as_array().unwrap().len(), 5);
    assert_eq!(counts["/meta"], 1);
    assert_eq!(counts["/episode"], 2);
    // 1.5 s at 50 Hz plus the two initial states.
    assert_eq!(counts["/agent/0/state"], 77);
    assert_eq!(counts["/agent/4/pose"], 77);
    assert_eq!(counts["/agent/3/action"], 75);
    // 10 Hz scans while the agent was active (it may have crashed).
    assert!((1..=16).contains(&counts["/agent/0/lidar"]));
    assert!(!counts.contains_key("/agent/3/lidar"));
    assert_eq!(last_time, 1_500_000_000);

    // Typed read-back.
    let rec = Recording::read(&path).unwrap();
    assert_eq!(rec.scenario, sc.spec);
    assert_eq!((rec.physics_hz, rec.policy_hz, rec.state_hz), (sc.spec.physics_hz, sc.spec.policy_hz, 50));
    assert_eq!((rec.agents.len(), rec.agents[4].group.as_str()), (5, sc.groups[1].spec.name.as_str()));
    assert_eq!(rec.episodes.len(), 2);
    let (first, second) = (&rec.episodes[0], &rec.episodes[1]);
    assert_eq!((first.number, second.number), (0, 1));
    assert_eq!((first.states[0].len(), second.states[4].len(), second.actions[3].len()), (51, 26, 25));
    assert_eq!((first.duration(), second.start, second.duration()), (1.0, 1.0, 0.5));
    assert!(!first.scans[0].is_empty() && first.scans[3].is_empty());
    let w = recorded.world(1);
    for (i, a) in w.agents().iter().enumerate() {
        // Floats survive the JSON round trip exactly.
        let last = second.states[i].last().unwrap();
        assert_eq!((last.position, last.orientation), (a.vehicle.position(), a.vehicle.orientation()));
        assert_eq!(last.motors, a.vehicle.as_multirotor().unwrap().motor_speeds());
        assert_eq!(second.goals[i], a.goals);
    }
    // The scenario compiles again, with its maps checked against the recorded hashes.
    assert_eq!(rec.compile().unwrap().map_hashes, sc.map_hashes);
    let mut tampered = rec.clone();
    tampered.map_hashes[0] = "0".repeat(64);
    assert!(tampered.compile().unwrap_err().to_string().contains("differ"));
}

#[test]
fn memory_sink_reports_events() {
    let sc = compile(
        r#"
        map = { type = "testworld", kind = "flat", size = 100.0 }
        [[groups]]
        spawn = { agl = [3.0, 3.0] }
        "#,
    );
    let sink = Arc::new(Mutex::new(MemorySink::default()));
    let mut rec = Recorder::new(Box::new(sink.clone()), RecorderConfig::default());
    let mut w = WorldInstance::new(sc, Seed::from_u64(0));
    rec.on_reset(&w);
    for _ in 0..60 {
        w.set_action(0, &[0.0, 0.0, 0.0, -1.0]);
        rec.on_actions(&w);
        w.step_with(&mut |w| rec.on_tick(w));
    }
    rec.finish().unwrap();
    let sink = sink.lock().unwrap();
    let events = sink.topic("/events");
    assert!(!events.is_empty());
    let names: Vec<&str> = events[0].1["events"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    assert!(names.contains(&"crash_terrain"), "{names:?}");
    let states = sink.topic("/agent/0/state");
    assert_eq!(states.len(), 61);
    assert!(states.windows(2).all(|w| w[1].0 - w[0].0 == 20_000_000));
    assert!(states.last().unwrap().1["disabled"].as_bool().unwrap());
}

#[test]
fn wild_map_pool_is_drawn_per_episode() {
    let sc = compile(
        r#"
        map = { type = "wild", seed = 7, count = 3, cache = false, config = { size = 128.0 } }
        [[groups]]
        count = 2
        action_mode = "velocity"
        spawn = { agl = [2.0, 4.0], clearance = 1.5 }
        "#,
    );
    assert_eq!((sc.maps.len(), sc.map_hashes.len()), (3, 3));
    for (k, (m, h)) in sc.maps.iter().zip(&sc.map_hashes).enumerate() {
        assert_eq!((m.meta.generator.as_str(), m.meta.seed), ("wild", 7 + k as u64));
        assert_eq!(*h, m.content_hash());
    }
    assert_ne!(sc.map_hashes[0], sc.map_hashes[1]);

    let sink = Arc::new(Mutex::new(MemorySink::default()));
    let mut rec = Recorder::new(Box::new(sink.clone()), RecorderConfig::default());
    let mut w = WorldInstance::new(sc.clone(), Seed::from_u64(1));
    let mut used = [0; 3];
    for _ in 0..24 {
        assert!(Arc::ptr_eq(w.map(), &sc.maps[w.map_index()]));
        used[w.map_index()] += 1;
        rec.on_reset(&w);
        // Spawned in free air above the generated terrain, and hovering there is uneventful.
        for a in w.agents() {
            assert!((1.5..4.5).contains(&a.agl_now(w.map())), "agl {}", a.agl_now(w.map()));
        }
        for _ in 0..5 {
            w.set_actions(0, &[0.0; 8]);
            w.step();
        }
        assert!(w.agents().iter().all(|a| !a.events.is_terminal()), "{:?}", w.agent(0).events);
        w.reset(None);
    }
    assert!(used.iter().all(|&n| n > 0), "{used:?}");
    rec.finish().unwrap();
    let sink = sink.lock().unwrap();
    let meta = &sink.topic("/meta")[0].1;
    assert_eq!(meta["maps"][2]["hash"].as_str().unwrap(), sc.map_hashes[2].to_string());
    let episodes = sink.topic("/episode");
    assert_eq!(episodes.len(), 24);
    assert!(episodes.iter().all(|e| e.1["map"].as_u64().unwrap() < 3));
}
