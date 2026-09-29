//! Tiltrotors in the simulation: trimmed air spawns (rotors up in hover, forward in cruise),
//! spawns on the gear, the `raw` and the default `velocity` action modes, and recordings.

use autonomousim_core::rng::Seed;
use autonomousim_core::terrain::Terrain;
use autonomousim_sim::record::{Recorder, RecorderConfig, Recording};
use autonomousim_sim::{BatchSim, CompiledScenario, Events, Scenario, WorldInstance};
use autonomousim_vehicles::tiltrotor::Tiltrotor;
use std::f64::consts::FRAC_PI_2;
use std::sync::Arc;

fn scenario(spawn: &str) -> Arc<CompiledScenario> {
    scenario_in(spawn, "action_mode = \"raw\"")
}

fn scenario_in(spawn: &str, mode: &str) -> Arc<CompiledScenario> {
    let toml = format!(
        r#"
        name = "tilt"
        map = {{ type = "testworld", kind = "flat", size = 2000.0 }}
        [[groups]]
        name = "air"
        vehicle = "quadtilt_like"
        {mode}
        obs = [ {{ term = "rot6d" }}, {{ term = "last_action" }} ]
        spawn = {{ region = [[-100.0, -100.0], [100.0, 100.0]], {spawn} }}
        "#
    );
    Arc::new(Scenario::from_toml(&toml).unwrap().compile().unwrap())
}

fn tilt(w: &WorldInstance) -> &Tiltrotor {
    w.agent(0).vehicle.as_tiltrotor().unwrap()
}

fn run(w: &mut WorldInstance, seconds: f64) -> Events {
    let mut all = Events::NONE;
    for _ in 0..(seconds / w.scenario().policy_dt()).round() as usize {
        w.step();
        all |= w.agent(0).events;
    }
    all
}

/// Held at the trim, hover and cruise continue (open loop, so only briefly).
#[test]
fn air_spawns_are_trimmed() {
    for (speed, mount) in [(0.0, 0.0), (20.0, FRAC_PI_2)] {
        let sc = scenario(&format!("agl = [100.0, 100.0], clearance = 0.0, airspeed = [{speed}, {speed}]"));
        assert_eq!(sc.groups[0].act_dim(), 11);
        assert_eq!(sc.groups[0].spec.action_mode.unwrap().name(), "raw");
        let mut w = WorldInstance::new(sc, Seed::from_u64(1));
        let t = tilt(&w);
        assert!(t.tilts().iter().all(|&x| (x - mount).abs() < 1e-4), "{speed}: {:?}", t.tilts());
        let v0 = t.lin_vel_world();
        assert!((v0.length() - speed).abs() < 1e-9, "{speed}: {v0}");
        let (p0, omega) = (t.position(), t.rotor_speeds().to_vec());
        let e = run(&mut w, 1.0);
        assert!(e.is_empty(), "{speed}: {e:?}");
        let t = tilt(&w);
        assert!(t.lin_vel_world().distance(v0) < 0.2, "{speed}: {} vs {v0}", t.lin_vel_world());
        assert!(t.ang_vel_body().length() < 0.05, "{speed}: {}", t.ang_vel_body());
        assert!(t.position().distance(p0 + v0) < 0.2, "{speed}: {} vs {}", t.position(), p0 + v0);
        for (a, b) in t.rotor_speeds().iter().zip(&omega) {
            assert!((a / b - 1.0).abs() < 1e-3, "{speed}: {a} vs {b}");
        }
    }
}

/// On the gear with the throttles closed it stays put; opened up it lifts off.
#[test]
fn stands_on_the_gear_and_lifts_off() {
    let sc = scenario("on_ground = true");
    let mut w = WorldInstance::new(sc, Seed::from_u64(2));
    let p0 = tilt(&w).position();
    let e = run(&mut w, 3.0);
    assert!(e.contains(Events::LANDED) && !e.is_terminal(), "{e:?}");
    assert!(tilt(&w).position().distance(p0) < 0.05, "{} vs {p0}", tilt(&w).position());
    // Throttles at 0.8 (+0.6), rotors up (tilt 0 of −0.15…π/2), surfaces centred.
    let up = (2.0 * 0.15 / (FRAC_PI_2 + 0.15) - 1.0) as f32;
    let action = [0.6, 0.6, 0.6, 0.6, up, up, up, up, 0.0, 0.0, 0.0];
    w.set_actions(0, &action);
    let e = run(&mut w, 2.0);
    assert!(!e.is_terminal(), "{e:?}");
    let t = tilt(&w);
    assert!(t.position().z > p0.z + 1.0 && t.contacts().is_empty(), "{}", t.position());
}

#[test]
fn records_the_tiltrotor_state() {
    let sc = scenario("agl = [30.0, 30.0], clearance = 0.0");
    let path = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("tiltrotor.mcap");
    let mut b = BatchSim::from_compiled(sc, 1, 5, 1).unwrap();
    b.attach_recorder(
        0,
        Recorder::create(&path, RecorderConfig { state_hz: 50, lidar: false, ..Default::default() }).unwrap(),
    );
    let action = [0.0, 0.0, 0.0, 0.0, -0.8, -0.8, -0.8, -0.8, 0.1, 0.0, 0.0];
    for _ in 0..25 {
        b.step(&[&action]);
    }
    b.detach_recorder(0).unwrap().finish().unwrap();
    let rec = Recording::read(&path).unwrap();
    let last = rec.episodes[0].states[0].last().unwrap();
    let live = b.world(0).agent(0).vehicle.as_tiltrotor().unwrap();
    assert_eq!(last.throttles, vec![0.5; 4]);
    assert_eq!((last.tilts.as_slice(), last.motors.as_slice()), (live.tilts(), live.rotor_speeds()));
    let [a, e, r] = live.channels();
    assert_eq!(last.surfaces, [a, e, r, 0.0]);
    assert!(a > 0.0 && last.engine_power == live.electric_power() && last.engine_power > 0.0);
    assert!(last.position.distance(live.position()) < 1e-9);
}

/// The default `velocity` mode from the gear: climb out, convert to the wing and fly, convert
/// back to a hover, descend and settle on the gear with the rotors idling.
#[test]
fn velocity_mode_flies_a_circuit() {
    let sc = scenario_in("on_ground = true", "");
    assert_eq!(sc.groups[0].spec.action_mode.unwrap().name(), "velocity");
    assert_eq!(sc.groups[0].act_dim(), 4);
    let mut w = WorldInstance::new(sc, Seed::from_u64(3));
    let z0 = tilt(&w).position().z;
    w.set_actions(0, &[0.0, 0.0, 1.0, 0.0]);
    let e = run(&mut w, 8.0);
    assert!(!e.is_terminal() && tilt(&w).position().z > z0 + 12.0, "{e:?} {}", tilt(&w).position());
    // 20 m/s on the wing.
    let forward = w.scenario().groups[0].action_map.as_tiltrotor().unwrap().speeds()[0];
    let cruise = (20.0 / forward) as f32;
    w.set_actions(0, &[cruise, 0.0, 0.0, 0.0]);
    let e = run(&mut w, 25.0);
    let t = tilt(&w);
    assert!(!e.is_terminal(), "{e:?}");
    assert!(
        (t.lin_vel_world().length() - 20.0).abs() < 0.3 && t.tilts().iter().all(|x| *x > 1.5),
        "{}",
        t.lin_vel_world()
    );
    w.set_actions(0, &[0.0, 0.0, 0.0, 0.0]);
    let e = run(&mut w, 30.0);
    let t = tilt(&w);
    assert!(!e.is_terminal() && t.lin_vel_world().length() < 0.2 && t.tilts().iter().all(|x| x.abs() < 0.05), "{e:?}");
    w.set_actions(0, &[0.0, 0.0, -0.5, 0.0]);
    let e = run(&mut w, 25.0);
    let t = tilt(&w);
    assert!(e.contains(Events::LANDED) && !e.is_terminal(), "{e:?} {}", t.position());
    assert!(t.lin_vel_world().length() < 0.05 && t.input().throttle.iter().all(|x| *x < 0.2), "{:?}", t.input());
}

/// Yard goals: the tiltrotor starts on its gear on one farm yard's pad, clear of obstacles
/// around it and overhead, and its goal is another yard's pad at a distance in the range; it
/// lifts off straight up from there, turning.
#[test]
fn yard_goals_join_two_farm_pads() {
    let sc = Scenario::from_toml(
        r#"
        name = "delivery"
        map = { type = "rural", seed = 5, count = 2, preset = "training", config = { size = 1024.0, farms = { spacing = 200.0 } } }
        [[groups]]
        vehicle = "quadtilt_like"
        spawn = { on_ground = true }
        goals = { kind = "yard", distance = [300.0, 700.0], agl = [0.0, 0.0] }
        "#,
    )
    .unwrap();
    let sc = Arc::new(sc.compile().unwrap());
    let mut w = WorldInstance::new(sc, Seed::from_u64(1));
    for episode in 0..8u64 {
        w.reset(Some(episode));
        let yards = autonomousim_sim::bay::yards(w.map());
        let pads: Vec<glam::DVec2> = yards.iter().map(|y| autonomousim_sim::bay::pad(w.map(), y)).collect();
        let (p, goal) = (tilt(&w).position(), w.agent(0).goal().position);
        let is_pad = |q: glam::DVec2| pads.iter().any(|x| (*x - q).length() < 1e-9);
        assert!(is_pad(p.truncate()) && is_pad(goal.truncate()), "{episode}: {p} {goal}");
        let d = (goal - p).truncate().length();
        assert!((280.0..=720.0).contains(&d), "{episode}: {d}");
        assert!((goal.z - w.map().surface_height(goal.x, goal.y)).abs() < 1e-9);
        for h in [1.0, 10.0, 30.0] {
            let q = p.truncate().extend(w.map().terrain().height(p.x, p.y) + h);
            let c = w.map().obstacle_clearance(q, 20.0);
            assert!(c > 6.0, "{episode}: clearance {c} at {h} m over the pad");
        }
        // Up 20 m turning at the full yaw rate (asked for on the gear too, which holds the
        // heading until lift-off), no strike.
        let events = run(&mut w, 1.0);
        assert!(tilt(&w).lin_vel_world().length() < 0.05 && !events.is_terminal(), "{episode}: {events:?}");
        let climb = 1.0 / w.scenario().groups[0].action_map.as_tiltrotor().unwrap().speeds()[2];
        let z = tilt(&w).position().z;
        for _ in 0..(10.0 / w.scenario().policy_dt()) as usize {
            w.set_actions(0, &[0.0, 0.0, (2.0 * climb) as f32, -1.0]);
            w.step();
            assert!(!w.agent(0).events.is_terminal(), "{episode}: {:?}", w.agent(0).events);
        }
        assert!(tilt(&w).position().z > z + 15.0, "{episode}");
    }
}
