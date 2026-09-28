//! Helicopters in the simulation: trimmed air spawns (hover and forward flight), spawns on the
//! skids, the `sticks` and `velocity` action modes, recordings and landing-zone goals.

use autonomousim_core::geometry::{HitMask, StaticGeometry};
use autonomousim_core::rng::Seed;
use autonomousim_core::terrain::Terrain;
use autonomousim_sim::record::{Recorder, RecorderConfig, Recording};
use autonomousim_sim::{BatchSim, CompiledScenario, Events, Scenario, WorldInstance};
use autonomousim_vehicles::rotorcraft::Helicopter;
use std::sync::Arc;

fn scenario(vehicle: &str, spawn: &str) -> Arc<CompiledScenario> {
    scenario_in(vehicle, spawn, "action_mode = \"sticks\"")
}

fn scenario_in(vehicle: &str, spawn: &str, mode: &str) -> Arc<CompiledScenario> {
    let toml = format!(
        r#"
        name = "heli"
        map = {{ type = "testworld", kind = "flat", size = 2000.0 }}
        [[groups]]
        name = "air"
        vehicle = "{vehicle}"
        {mode}
        obs = [ {{ term = "rot6d" }}, {{ term = "last_action" }} ]
        spawn = {{ region = [[-100.0, -100.0], [100.0, 100.0]], {spawn} }}
        "#
    );
    Arc::new(Scenario::from_toml(&toml).unwrap().compile().unwrap())
}

fn heli(w: &WorldInstance) -> &Helicopter {
    w.agent(0).vehicle.as_helicopter().unwrap()
}

fn run(w: &mut WorldInstance, seconds: f64) -> Events {
    let mut all = Events::NONE;
    for _ in 0..(seconds / w.scenario().policy_dt()).round() as usize {
        w.step();
        all |= w.agent(0).events;
    }
    all
}

/// Held at the trim, hover and forward flight continue (open loop, so only briefly).
#[test]
fn air_spawns_are_trimmed() {
    for (vehicle, speed) in [("bo105_like", None), ("bo105_like", Some(40.0)), ("xcell60_like", None)] {
        let airspeed = speed.map_or(String::new(), |v| format!(", airspeed = [{v}, {v}]"));
        let sc = scenario(vehicle, &format!("agl = [100.0, 100.0], clearance = 0.0{airspeed}"));
        assert_eq!(sc.groups[0].act_dim(), 4);
        assert_eq!(sc.groups[0].spec.action_mode.unwrap().name(), "sticks");
        let mut w = WorldInstance::new(sc, Seed::from_u64(1));
        let h = heli(&w);
        let v0 = h.lin_vel_world();
        assert!((v0.length() - speed.unwrap_or(0.0)).abs() < 1e-9, "{vehicle}: {v0}");
        let (p0, omega) = (h.position(), h.def().engine.rated_speed);
        let e = run(&mut w, 1.0);
        assert!(e.is_empty(), "{vehicle}: {e:?}");
        let h = heli(&w);
        assert!(h.lin_vel_world().distance(v0) < 0.3, "{vehicle}: {} vs {v0}", h.lin_vel_world());
        assert!(h.ang_vel_body().length() < 0.05, "{vehicle}: {}", h.ang_vel_body());
        let expected = p0 + v0 * 1.0;
        assert!(h.position().distance(expected) < 0.3, "{vehicle}: {} vs {expected}", h.position());
        assert!(
            (h.rotor_speed() / omega - 1.0).abs() < 1e-3,
            "{vehicle}: {} {omega} {}",
            h.rotor_speed(),
            h.position()
        );
    }
}

/// On the skids with the collective down it stays put; full collective lifts it off.
#[test]
fn stands_on_the_skids_and_lifts_off() {
    for vehicle in ["bo105_like", "xcell60_like"] {
        let sc = scenario(vehicle, "on_ground = true");
        let mut w = WorldInstance::new(sc, Seed::from_u64(2));
        let p0 = heli(&w).position();
        let e = run(&mut w, 3.0);
        assert!(e.contains(Events::LANDED) && !e.is_terminal(), "{vehicle}: {e:?}");
        assert!(heli(&w).position().distance(p0) < 0.05, "{vehicle}: {} vs {p0}", heli(&w).position());
        // Collective up with the pedal and cyclic near their hover trim: it climbs away.
        let t = heli(&w).trim(0.0, 1.225, 9.80665).unwrap().controls;
        let action = [t.collective + 0.4, t.longitudinal, t.lateral, t.pedal].map(|x| x as f32);
        w.set_actions(0, &action);
        let e = run(&mut w, 2.0);
        assert!(!e.is_terminal(), "{vehicle}: {e:?}");
        let h = heli(&w);
        assert!(h.position().z > p0.z + 1.0 && h.contacts().is_empty(), "{vehicle}: {}", h.position());
    }
}

#[test]
fn records_the_helicopter_state() {
    let sc = scenario("xcell60_like", "agl = [30.0, 30.0], clearance = 0.0");
    let path = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("helicopter.mcap");
    let mut b = BatchSim::from_compiled(sc, 1, 5, 1).unwrap();
    b.attach_recorder(0, Recorder::create(&path, RecorderConfig { state_hz: 50, lidar: false }).unwrap());
    for _ in 0..25 {
        b.step(&[&[0.1, 0.0, 0.0, 0.5]]);
    }
    b.detach_recorder(0).unwrap().finish().unwrap();
    let rec = Recording::read(&path).unwrap();
    let last = rec.episodes[0].states[0].last().unwrap();
    let live = b.world(0).agent(0).vehicle.as_helicopter().unwrap();
    assert_eq!(last.controls, [0.1f32, 0.0, 0.0, 0.5].map(f64::from));
    assert_eq!(last.pitches, live.pitches());
    assert_eq!((last.rotor_speed, last.engine_power), (live.rotor_speed(), live.engine_power()));
    assert_eq!(last.flap, [live.main_rotor_state().flap, live.tail_rotor_state().flap]);
    assert!(last.coning[0] > 0.0 && last.position.distance(live.position()) < 1e-9);
}

/// The default `velocity` mode: half forward speed, then back to a hover in place.
#[test]
fn velocity_mode_flies_forward_and_stops() {
    let sc = scenario_in("xcell60_like", "agl = [50.0, 50.0], clearance = 0.0", "");
    assert_eq!(sc.groups[0].spec.action_mode.unwrap().name(), "velocity");
    let mut w = WorldInstance::new(sc.clone(), Seed::from_u64(3));
    let fwd = sc.groups[0].action_map.as_helicopter().unwrap().speeds()[0];
    w.set_actions(0, &[0.5, 0.0, 0.0, 0.0]);
    let e = run(&mut w, 25.0);
    assert!(e.is_empty(), "{e:?}");
    let h = heli(&w);
    let heading = h.orientation() * glam::DVec3::X;
    let v = h.lin_vel_world();
    assert!((v.dot(heading.with_z(0.0).normalize()) - 0.5 * fwd).abs() < 0.3, "{v} vs {fwd}");
    w.set_actions(0, &[0.0; 4]);
    let e = run(&mut w, 20.0);
    assert!(e.is_empty(), "{e:?}");
    assert!(heli(&w).lin_vel_world().length() < 0.2, "{}", heli(&w).lin_vel_world());
}

/// Landing-zone goals on a generated wild map: flat within the slope, dry, and clear of trees
/// and rocks within the clearance up to 40 m.
#[test]
fn landing_zones_are_flat_dry_and_open() {
    let toml = r#"
        name = "landing"
        map = { type = "wild", preset = "offroad", seed = 3, cache = false }
        [[groups]]
        name = "heli"
        count = 8
        vehicle = "xcell60_like"
        spawn = { agl = [40.0, 40.0], margin = 50.0 }
        goals = { kind = "random", distance = [100.0, 300.0], agl = [0.0, 0.0], landing_slope = 0.1, clearance = 3.0, margin = 50.0, radius = 0.0 }
    "#;
    let sc = Arc::new(Scenario::from_toml(toml).unwrap().compile().unwrap());
    for seed in 0..3 {
        let w = WorldInstance::new(sc.clone(), Seed::from_u64(seed));
        let map = w.map().clone();
        let t = map.terrain();
        for a in w.agents() {
            let p = a.goals[0].position;
            let h = t.height(p.x, p.y);
            assert!((p.z - map.surface_height(p.x, p.y)).abs() < 1e-9, "goal {p} not on the surface");
            for k in 0..16 {
                let q = p.truncate() + glam::DVec2::from_angle(f64::from(k) * std::f64::consts::TAU / 16.0) * 3.0;
                assert!((t.height(q.x, q.y) - h).abs() <= 0.1 * 3.0 + 0.05, "goal {p}: slope at {q}");
                assert!(t.water_level(q.x, q.y).is_none_or(|l| l <= t.height(q.x, q.y)), "goal {p}: water at {q}");
            }
            for z in [1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 40.0] {
                let c = glam::DVec3::new(p.x, p.y, h + z);
                let near = map.obstacles().nearest_distance(c, 3.0, HitMask::SOLID | HitMask::FOLIAGE);
                assert!(near.is_none(), "goal {p}: obstacle {near:?} m from {c}");
            }
        }
    }
    // Validated before the map is built: ground vehicles and non-positive slopes are rejected.
    let flat = toml.replace(
        r#"{ type = "wild", preset = "offroad", seed = 3, cache = false }"#,
        r#"{ type = "testworld", kind = "flat", size = 200.0 }"#,
    );
    let err = |t: &str| Scenario::from_toml(t).unwrap().compile().expect_err("rejected").to_string();
    assert!(err(&flat.replace("xcell60_like", "offroad_4x4")).contains("landing_slope"));
    assert!(err(&flat.replace("landing_slope = 0.1", "landing_slope = 0.0")).contains("invalid goals"));
}
