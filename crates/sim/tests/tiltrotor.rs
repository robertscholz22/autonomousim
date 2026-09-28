//! Tiltrotors in the simulation: trimmed air spawns (rotors up in hover, forward in cruise),
//! spawns on the gear, the `raw` action mode and recordings.

use autonomousim_core::rng::Seed;
use autonomousim_sim::record::{Recorder, RecorderConfig, Recording};
use autonomousim_sim::{BatchSim, CompiledScenario, Events, Scenario, WorldInstance};
use autonomousim_vehicles::tiltrotor::Tiltrotor;
use std::f64::consts::FRAC_PI_2;
use std::sync::Arc;

fn scenario(spawn: &str) -> Arc<CompiledScenario> {
    let toml = format!(
        r#"
        name = "tilt"
        map = {{ type = "testworld", kind = "flat", size = 2000.0 }}
        [[groups]]
        name = "air"
        vehicle = "quadtilt_like"
        action_mode = "raw"
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
    b.attach_recorder(0, Recorder::create(&path, RecorderConfig { state_hz: 50, lidar: false }).unwrap());
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
