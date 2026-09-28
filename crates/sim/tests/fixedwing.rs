//! Fixed-wing aircraft in the simulation: trimmed air spawns (with wind), spawns on the gear,
//! the `surfaces` action mode, the stall event, recordings and the controlled `guidance` mode in
//! turbulence.

use autonomousim_core::rng::Seed;
use autonomousim_sim::record::{Recorder, RecorderConfig, Recording};
use autonomousim_sim::{BatchSim, CompiledScenario, Events, Scenario, WorldInstance};
use autonomousim_vehicles::fixedwing::FixedWing;
use std::sync::Arc;

fn compile(toml: &str) -> Arc<CompiledScenario> {
    Arc::new(Scenario::from_toml(toml).unwrap().compile().unwrap())
}

fn scenario(vehicle: &str, spawn: &str, extra: &str) -> Arc<CompiledScenario> {
    compile(&format!(
        r#"
        name = "fw"
        map = {{ type = "testworld", kind = "flat", size = 4000.0 }}
        {extra}
        [[groups]]
        name = "air"
        vehicle = "{vehicle}"
        action_mode = "surfaces"
        obs = [ {{ term = "air_data" }}, {{ term = "rot6d" }}, {{ term = "last_action" }} ]
        spawn = {{ region = [[-300.0, -300.0], [300.0, 300.0]], {spawn} }}
        "#
    ))
}

fn aircraft(w: &WorldInstance) -> &FixedWing {
    w.agent(0).vehicle.as_fixed_wing().unwrap()
}

fn run(w: &mut WorldInstance, seconds: f64) -> Events {
    let mut all = Events::NONE;
    for _ in 0..(seconds / w.scenario().policy_dt()).round() as usize {
        w.step();
        all |= w.agent(0).events;
    }
    all
}

#[test]
fn air_spawns_are_trimmed() {
    let sc = scenario("c172_like", "agl = [300.0, 300.0], airspeed = [45.0, 50.0], clearance = 0.0", "");
    let g = &sc.groups[0];
    assert_eq!(g.act_dim(), 4);
    assert_eq!(sc.spec.physics_hz, 500);
    let mut w = WorldInstance::new(sc.clone(), Seed::from_u64(1));
    let a = aircraft(&w);
    let v0 = a.lin_vel_world().length();
    assert!((45.0..=50.0).contains(&v0), "{v0}");
    let z0 = a.position().z;
    // Held at trim, level flight continues.
    let e = run(&mut w, 5.0);
    assert!(e.is_empty(), "{e:?}");
    let a = aircraft(&w);
    assert!((a.lin_vel_world().length() - v0).abs() < 0.5 && (a.position().z - z0).abs() < 2.0);
    // The state row carries the air data.
    let mut s = vec![0.0; autonomousim_sim::STATE_DIM];
    w.write_state(0, &mut s);
    assert!((s[30] - a.flow().airspeed).abs() < 1e-9);
}

#[test]
fn trimmed_in_the_wind() {
    let wind = "environment = { wind = { mean = [8.0, 0.0] } }";
    let sc = scenario("aerosonde_like", "agl = [100.0, 100.0], clearance = 0.0", wind);
    let mut w = WorldInstance::new(sc.clone(), Seed::from_u64(2));
    let a = aircraft(&w);
    // Default airspeed: 1.5 stall speeds, relative to the air (the ground speed differs).
    let d = a.def();
    let density = 1.225 * (1.0 - 2.2558e-5 * 100.0f64).powf(4.2559);
    let expected = 1.5 * d.stall_speed(density, 9.80665);
    let air = w.agent(0).air().wind;
    assert!(air.length() > 7.0, "wind {air}");
    let airspeed = (a.lin_vel_world() - air).length();
    assert!((airspeed - expected).abs() < 0.05 * expected, "{airspeed} vs {expected}");
    assert!((a.lin_vel_world().length() - expected).abs() > 1.0, "ground speed equals airspeed");
    let e = run(&mut w, 3.0);
    assert!(!e.is_terminal() && !e.contains(Events::STALL), "{e:?}");
}

#[test]
fn stands_on_the_gear_and_takes_off() {
    let sc = scenario("aerosonde_like", "on_ground = true, yaw_deg = [0.0, 0.0]", "");
    let mut w = WorldInstance::new(sc.clone(), Seed::from_u64(3));
    let p0 = aircraft(&w).position();
    // Held on the brakes.
    let e = run(&mut w, 3.0);
    assert!(e.contains(Events::LANDED) && !e.is_terminal() && !e.contains(Events::STALL), "{e:?}");
    assert!(aircraft(&w).position().distance(p0) < 0.05);
    assert!(aircraft(&w).wheels().iter().all(|c| c.is_some()));
    // Full throttle, a little up elevator: rolls, lifts off and climbs.
    w.set_actions(0, &[0.0, 0.3, 0.0, 1.0]);
    let e = run(&mut w, 15.0);
    let a = aircraft(&w);
    assert!(!e.is_terminal(), "{e:?}");
    assert!(a.position().z > p0.z + 10.0 && !a.weight_on_wheels(), "{}", a.position());
    assert!(a.position().x > p0.x + 50.0);
}

#[test]
fn full_up_elevator_stalls() {
    let sc = scenario("c172_like", "agl = [500.0, 500.0], airspeed = [35.0, 35.0], clearance = 0.0", "");
    let mut w = WorldInstance::new(sc, Seed::from_u64(4));
    w.set_actions(0, &[0.0, 1.0, 0.0, -1.0]);
    let e = run(&mut w, 8.0);
    assert!(e.contains(Events::STALL), "{e:?}");
    assert!(!Events::STALL.is_terminal());
}

#[test]
fn records_the_aircraft_state() {
    let sc = scenario("aerosonde_like", "agl = [80.0, 80.0], clearance = 0.0", "");
    let path = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("fixedwing.mcap");
    let mut b = BatchSim::from_compiled(sc, 1, 5, 1).unwrap();
    b.attach_recorder(0, Recorder::create(&path, RecorderConfig { state_hz: 50, lidar: false }).unwrap());
    for _ in 0..50 {
        b.step(&[&[0.2, 0.1, 0.0, 0.5]]);
    }
    b.detach_recorder(0).unwrap().finish().unwrap();
    let rec = Recording::read(&path).unwrap();
    let last = rec.episodes[0].states[0].last().unwrap();
    let live = b.world(0).agent(0).vehicle.as_fixed_wing().unwrap();
    assert_eq!(last.surfaces, live.surfaces());
    assert_eq!((last.throttle, last.rotor_speed), (0.75, live.rotor_speed()));
    assert_eq!((last.airspeed, last.alpha), (live.flow().airspeed, live.flow().alpha));
    assert_eq!(last.gear_loads, vec![0.0; 3]);
    assert!(last.surfaces[0] > 0.0 && last.position.distance(live.position()) < 1e-9);
}

/// The default controller in `guidance` mode through the scenario: a zero action (straight,
/// level, mid-range airspeed) keeps altitude and the ground track in light turbulence over a
/// crosswind.
#[test]
fn guidance_mode_holds_course_and_altitude() {
    for vehicle in ["aerosonde_like", "c172_like"] {
        let sc = compile(&format!(
            r#"
            name = "fw"
            map = {{ type = "testworld", kind = "flat", size = 8000.0 }}
            environment = {{ wind = {{ mean = [0.0, 4.0], turbulence_w20 = 7.7 }} }}
            [[groups]]
            name = "air"
            vehicle = "{vehicle}"
            action_mode = "guidance"
            obs = [ {{ term = "air_data" }} ]
            spawn = {{ region = [[-100.0, -100.0], [100.0, 100.0]], agl = [400.0, 400.0], yaw_deg = [0.0, 0.0], clearance = 0.0 }}
            "#
        ));
        let mut w = WorldInstance::new(sc, Seed::from_u64(5));
        w.set_actions(0, &[0.0, 0.0, 0.0]);
        // The first 20 s trade some height for the speed-up from the spawn speed (1.5 V_s).
        let mut e = run(&mut w, 20.0);
        let p1 = aircraft(&w).position();
        let z0 = p1.z;
        e |= run(&mut w, 40.0);
        let a = aircraft(&w);
        let v = a.lin_vel_world();
        assert!(!e.is_terminal() && !e.contains(Events::STALL), "{vehicle}: {e:?}");
        assert!((a.position().z - z0).abs() < 10.0, "{vehicle}: altitude {} → {}", z0, a.position().z);
        // A zero course rate holds the ground track (crabbing into the wind), not the heading.
        let track = (a.position() - p1).truncate();
        assert!(v.truncate().angle_to(track).abs() < 0.1, "{vehicle}: course {v} vs track {track}");
    }
}
