//! Wheeled agents in worlds: compilation of family-scoped groups, spawning on the ground,
//! driving through the action modes, events, determinism and recording.

use autonomousim_control::AgentActionMode;
use autonomousim_control::ground::{GroundActionMode, GroundSetpoint};
use autonomousim_core::math::quat::yaw;
use autonomousim_core::rng::Seed;
use autonomousim_sensors::Sensor;
use autonomousim_sim::record::{MemorySink, Recorder, RecorderConfig, Recording};
use autonomousim_sim::{BatchSim, CompiledScenario, Events, Scenario, WorldInstance};
use autonomousim_vehicles::{Family, Vehicle};
use std::sync::{Arc, Mutex};

fn compile(toml: &str) -> Arc<CompiledScenario> {
    Arc::new(Scenario::from_toml(toml).unwrap().compile().unwrap())
}

/// Two sedans (default action mode), a skid-steer rover in `vw` and a drone, on flat ground.
const MIXED: &str = r#"
    name = "ground"
    physics_hz = 1000
    map = { type = "testworld", kind = "flat", size = 200.0 }
    [[groups]]
    name = "cars"
    count = 2
    vehicle = "sedan_like"
    spawn = { min_separation = 8.0, clearance = 3.0 }
    sensors = [ { name = "imu", type = "imu" } ]
    obs = [ { term = "lin_vel_body" }, { term = "imu", sensor = "imu" }, { term = "last_action" } ]
    [[groups]]
    name = "rover"
    vehicle = "rover_skid"
    action_mode = "vw"
    spawn = { min_separation = 8.0 }
    [[groups]]
    name = "drone"
    action_mode = "velocity"
    spawn = { min_separation = 8.0, agl = [2.0, 2.0] }
"#;

fn step(w: &mut WorldInstance, actions: &[&[f32]]) {
    for (g, a) in actions.iter().enumerate() {
        w.set_actions(g, a);
    }
    w.step();
}

#[test]
fn groups_compile_per_family() {
    let sc = compile(MIXED);
    let [cars, rover, drone] = &sc.groups[..] else { panic!() };
    assert_eq!((cars.family(), rover.family(), drone.family()), (Family::Wheeled, Family::Wheeled, Family::Multirotor));
    // The family's default mode is filled in, and ground vehicles start on the ground.
    assert_eq!(cars.action_mode(), AgentActionMode::Ground(GroundActionMode::Vk));
    assert_eq!(sc.spec.groups[0].action_mode, Some(GroundActionMode::Vk.into()));
    assert!(sc.spec.groups[0].spawn.on_ground && !sc.spec.groups[2].spawn.on_ground);
    assert_eq!((cars.act_dim(), rover.act_dim(), drone.act_dim()), (2, 2, 4));
    assert_eq!(cars.obs_dim(), 3 + 6 + 2);
    // The filled-in scenario round-trips through JSON; ground settings are only written when set.
    let json = sc.spec.to_json();
    assert_eq!(Scenario::from_json(&json).unwrap(), sc.spec);
    assert!(!json.contains("ground_controller") && !json.contains("ground_action_limits"));
    let with_limits =
        MIXED.replace("action_mode = \"vw\"", "action_mode = \"vw\"\nground_action_limits = { speed = 1.0 }");
    let sc2 = compile(&with_limits);
    assert!(sc2.spec.to_json().contains("ground_action_limits"));

    // Settings of the other family, or modes and observations that do not fit, are errors.
    for (bad, why) in [
        ("[[groups]]\nvehicle = \"sedan_like\"\naction_mode = \"ctbr\"", "does not drive wheeled"),
        ("[[groups]]\naction_mode = \"vk\"", "does not drive multirotor"),
        ("[[groups]]\nvehicle = \"sedan_like\"\nrandomize = { mass = 0.1 }", "`randomize` does not apply"),
        ("[[groups]]\nvehicle = \"sedan_like\"\ncontroller = { gravity = 9.7 }", "`controller` does not apply"),
        ("[[groups]]\nground_controller = { speed_time_constant = 1.0 }", "`ground_controller` does not apply"),
        ("[[groups]]\nvehicle = \"sedan_like\"\nobs = [ { term = \"motor_speeds\" } ]", "needs a vehicle with rotors"),
        ("[[groups]]\naction_mode = \"warp\"", "unknown action mode"),
    ] {
        let err = Scenario::from_toml(bad).and_then(Scenario::compile).unwrap_err().to_string();
        assert!(err.contains(why), "{bad}: {err}");
    }
}

#[test]
fn ground_vehicles_rest_then_drive() {
    let sc = compile(MIXED);
    let mut w = WorldInstance::new(sc.clone(), Seed::from_u64(3));
    let start: Vec<_> = w.agents().iter().map(|a| a.vehicle.pose()).collect();
    for a in &w.agents()[..3] {
        assert!(matches!(a.vehicle, Vehicle::Wheeled(_)));
        // At rest on its wheels: the chassis is above the ground, its goal on the ground
        // where it stands.
        let p = a.vehicle.position();
        assert!(p.z > 0.1 && p.z < 1.0, "{p}");
        assert_eq!(a.goal().position.truncate(), p.truncate());
        assert!(a.goal().position.z.abs() < 0.01);
    }
    // One second of standing still (zero speed): nothing moves, nothing happens.
    for _ in 0..50 {
        step(&mut w, &[&[0.0; 4], &[0.0; 2], &[0.0; 4]]);
    }
    for (a, p) in w.agents().iter().zip(&start).take(3) {
        assert!(a.vehicle.position().distance(p.pos) < 0.02, "{}: {}", a.id, a.vehicle.position() - p.pos);
        assert!(a.events.is_empty(), "{}: {:?}", a.id, a.events);
    }
    // The chassis feels gravity, and the (noisy) IMU reads it.
    let v = &w.agent(0).vehicle;
    let up = v.orientation().inverse() * glam::DVec3::Z * 9.81;
    assert!(v.specific_force_body().distance(up) < 0.05, "{} vs {up}", v.specific_force_body());
    let Sensor::Imu(imu) = &w.agent(0).sensors[0] else { panic!() };
    let f = imu.latest().unwrap().value.accel;
    assert!((f.z - 9.81).abs() < 1.0 && f.truncate().length() < 1.0, "{f}");

    // Drive: the cars ahead at half speed, turning gently; the rover ahead and turning.
    let (cars, rover) = (&sc.groups[0], &sc.groups[1]);
    let v_car = 0.5 * cars.action_map.as_ground().unwrap().speed();
    let v_rover = 0.5 * rover.action_map.as_ground().unwrap().speed();
    for _ in 0..250 {
        step(&mut w, &[&[0.5, 0.1, 0.5, 0.1], &[0.5, 0.3], &[0.0; 4]]);
    }
    for (i, v) in [(0, v_car), (1, v_car), (2, v_rover)] {
        let a = w.agent(i);
        let speed = a.vehicle.lin_vel_body().x;
        assert!((speed - v).abs() < 0.1 * v, "agent {i}: {speed} vs {v}");
        assert!(yaw(a.vehicle.orientation()) != yaw(start[i].rot), "agent {i} turns");
        assert!(!a.events.is_terminal(), "{i}: {:?}", a.events);
        // Moved forward along the initial heading, at least at first.
        let d = a.vehicle.position() - start[i].pos;
        assert!(d.dot(start[i].rot * glam::DVec3::X) > 0.0 && d.length() > 2.0, "{i}: {d}");
    }
    // The drone held its position meanwhile.
    assert!(w.agent(3).vehicle.position().distance(start[3].pos) < 0.2);
    // Observations: forward speed first.
    let mut obs = vec![0.0; 2 * cars.obs_dim()];
    w.observe(0, &mut obs);
    assert!((f64::from(obs[0]) - v_car).abs() < 0.1 * v_car);
    assert_eq!(&obs[9..11], &[0.5, 0.1]);

    // Direct commands: stop the first car.
    w.set_command(0, GroundSetpoint::SpeedCurvature { speed: 0.0, curvature: 0.0 });
    for _ in 0..250 {
        w.step();
    }
    assert!(w.agent(0).vehicle.lin_vel_body().length() < 0.05, "{}", w.agent(0).vehicle.lin_vel_body());
    assert!(w.agent(1).vehicle.lin_vel_body().x > 0.9 * v_car);
}

#[test]
#[should_panic(expected = "for a wheeled vehicle")]
fn commands_must_fit_the_family() {
    let mut w = WorldInstance::new(compile(MIXED), Seed::from_u64(0));
    w.set_command(0, autonomousim_control::multirotor::Setpoint::Motors([0.0; 8]));
}

#[test]
fn driving_into_a_tree_is_a_crash() {
    let sc = compile(
        r#"
        physics_hz = 1000
        map = { type = "testworld", kind = "forest_patch", size = 120.0, density = 400.0, seed = 2 }
        [[groups]]
        count = 4
        vehicle = "offroad_4x4"
        action_mode = "vk"
        spawn = { clearance = 2.5, min_separation = 6.0 }
        disable_on_terminal = true
        "#,
    );
    let mut w = WorldInstance::new(sc, Seed::from_u64(1));
    let mut seen = Events::NONE;
    for _ in 0..600 {
        w.set_actions(0, &[1.0, 0.0].repeat(4));
        w.step();
        seen = w.agents().iter().fold(seen, |e, a| e | a.events);
    }
    assert!(seen.contains(Events::CRASH_OBSTACLE), "{seen:?}");
    assert!(w.agents().iter().any(|a| a.disabled));
}

#[test]
fn deterministic_across_batches_and_recorded() {
    let sc = compile(MIXED);
    let acts = |k: u64, n: usize| -> Vec<Vec<f32>> {
        sc.groups
            .iter()
            .enumerate()
            .map(|(g, grp)| {
                (0..n * grp.spec.count * grp.act_dim())
                    .map(|i| 0.8 * ((k * 3 + g as u64 * 5 + i as u64) as f32 * 0.37).sin())
                    .collect()
            })
            .collect()
    };
    let mut one = BatchSim::from_compiled(sc.clone(), 1, 5, 1).unwrap();
    let mut many = BatchSim::from_compiled(sc.clone(), 4, 5, 3).unwrap();
    let sink = Arc::new(Mutex::new(MemorySink::default()));
    many.attach_recorder(0, Recorder::new(Box::new(sink.clone()), RecorderConfig::default()));
    for k in 0..60 {
        let a1 = acts(k, 1);
        let a4: Vec<Vec<f32>> = acts(k, 1).iter().map(|a| a.repeat(4)).collect();
        one.step(&a1.iter().map(|a| a.as_slice()).collect::<Vec<_>>());
        many.step(&a4.iter().map(|a| a.as_slice()).collect::<Vec<_>>());
        assert_eq!(one.world(0).state_hash(), many.world(0).state_hash(), "step {k}");
    }
    // The hash covers the ground vehicles' own state.
    let mut other = one.world(0).clone();
    assert_eq!(other.state_hash(), one.world(0).state_hash());
    other.agent_mut(0).vehicle.state_mut().v[7] += 1e-9;
    assert_ne!(other.state_hash(), one.world(0).state_hash());
    many.detach_recorder(0).unwrap().finish().unwrap();

    let sink = sink.lock().unwrap();
    let meta = &sink.topic("/meta")[0].1;
    assert_eq!(meta["vehicles"]["cars"]["type"], "wheeled");
    assert!(meta["vehicles"]["drone"].get("type").is_none());
    let car = sink.topic("/agent/0/state").pop().unwrap().1;
    assert!(car.get("motors").is_none());
    assert_eq!(car["wheels"].as_array().unwrap().len(), 4);
    assert!(car["wheels"][0]["load"].as_f64().unwrap() > 1000.0);
    let drone = sink.topic("/agent/3/state").pop().unwrap().1;
    assert_eq!(drone["motors"].as_array().unwrap().len(), 4);
    assert!(drone.get("wheels").is_none());
}

#[test]
fn ground_recordings_read_back() {
    let sc = compile(MIXED);
    let path = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("ground.mcap");
    let mut b = BatchSim::from_compiled(sc.clone(), 1, 2, 1).unwrap();
    b.attach_recorder(0, Recorder::create(&path, RecorderConfig::default()).unwrap());
    for _ in 0..25 {
        b.step(&[&[0.6, 0.0, 0.6, 0.0], &[0.3, 0.0], &[0.0; 4]]);
    }
    b.detach_recorder(0).unwrap().finish().unwrap();
    let rec = Recording::read(&path).unwrap();
    assert_eq!(rec.scenario, sc.spec);
    assert_eq!(rec.agents[0].vehicle, "sedan_like");
    let ep = &rec.episodes[0];
    let (car, drone) = (ep.states[0].last().unwrap(), ep.states[3].last().unwrap());
    let w = b.world(0);
    let sedan = w.agent(0).vehicle.as_wheeled().unwrap();
    assert_eq!(car.position, sedan.position());
    assert_eq!((car.wheels.len(), car.gear, car.steering), (4, sedan.powertrain().gear, sedan.steering_angle()));
    assert_eq!(car.wheels[2].spin, sedan.wheel(2).spin);
    assert!(car.motors.is_empty() && drone.wheels.is_empty() && drone.motors.len() == 4);
    rec.compile().unwrap();
}
