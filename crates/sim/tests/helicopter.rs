//! Helicopters in the simulation: trimmed air spawns (hover and forward flight), spawns on the
//! skids, the `sticks` action mode and recordings.

use autonomousim_core::rng::Seed;
use autonomousim_sim::record::{Recorder, RecorderConfig, Recording};
use autonomousim_sim::{BatchSim, CompiledScenario, Events, Scenario, WorldInstance};
use autonomousim_vehicles::rotorcraft::Helicopter;
use std::sync::Arc;

fn scenario(vehicle: &str, spawn: &str) -> Arc<CompiledScenario> {
    let toml = format!(
        r#"
        name = "heli"
        map = {{ type = "testworld", kind = "flat", size = 2000.0 }}
        [[groups]]
        name = "air"
        vehicle = "{vehicle}"
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
