//! Two-wheelers in scenarios: upright spawns on their feet (also across slopes), riding in
//! `vk`, their observation terms, events and recordings.

use autonomousim_control::ground::GroundSetpoint;
use autonomousim_core::rng::Seed;
use autonomousim_sim::record::{Recorder, RecorderConfig, Recording};
use autonomousim_sim::{BatchSim, CompiledScenario, Events, Scenario, WorldInstance};
use glam::{DVec3, EulerRot};
use std::sync::Arc;

fn compile(toml: &str) -> Arc<CompiledScenario> {
    Arc::new(Scenario::from_toml(toml).unwrap().compile().unwrap())
}

/// On rural maps, spawned on the roads and anywhere off them (on slopes up to their feet's
/// `max_slope_deg`), both presets stand on their feet for 3 s without a terminal event.
#[test]
fn spawn_on_rural_maps() {
    for vehicle in ["motorcycle_sport", "bicycle_city"] {
        for spawn in ["on_road = true", "margin = 20.0"] {
            let sc = compile(&format!(
                r#"
                name = "rural"
                physics_hz = 1000
                policy_hz = 20
                map = {{ type = "rural", seed = 0, count = 2, cache = false }}
                [[groups]]
                vehicle = "{vehicle}"
                spawn = {{ {spawn} }}
                "#
            ));
            for seed in 0..8 {
                let mut w = WorldInstance::new(sc.clone(), Seed::from_u64(seed));
                let mut seen = Events::NONE;
                for _ in 0..3 * sc.spec.policy_hz {
                    w.set_actions(0, &[0.0, 0.0]);
                    w.step();
                    seen = Events(seen.0 | events(&w).0);
                }
                let r = roll(&w);
                assert!(
                    !seen.intersects(Events::TERMINAL) && r.abs() < 0.4,
                    "{vehicle} {spawn} seed {seed}: {seen:?}, roll {r:.3}"
                );
            }
        }
    }
}

/// Drive grids clear the feet, which stick out beyond the handlebars.
#[test]
fn drive_grid_width_includes_the_feet() {
    for (vehicle, half) in [("motorcycle_sport", 0.51), ("bicycle_city", 0.45)] {
        let def = autonomousim_vehicles::SharedDef::from(autonomousim_vehicles::presets::get(vehicle).unwrap());
        let w = autonomousim_sim::drive::half_width(def.as_wheeled().unwrap());
        assert!((w - half).abs() < 1e-9, "{vehicle}: {w}");
    }
}

/// One two-wheeler in `vk` on `map`, heading `yaw` (degrees), with all its terms observed.
fn scenario(vehicle: &str, map: &str, yaw: f64) -> Arc<CompiledScenario> {
    compile(&format!(
        r#"
        name = "two_wheeler"
        physics_hz = 1000
        policy_hz = 20
        map = {map}
        [[groups]]
        vehicle = "{vehicle}"
        spawn = {{ yaw_deg = [{yaw}, {yaw}] }}
        ground_action_limits = {{ speed = 20.0 }}
        obs = [ {{ term = "lean" }}, {{ term = "steering" }}, {{ term = "rider_lean" }}, {{ term = "feet" }},
                {{ term = "speed" }} ]
        "#
    ))
}

const FLAT: &str = r#"{ type = "testworld", kind = "flat", size = 1000.0 }"#;

const OBS_DIM: usize = 2 + 2 + 2 + 1 + 1;

fn events(w: &WorldInstance) -> Events {
    let mut e = [0u32];
    w.write_events(0, &mut e);
    Events(e[0])
}

fn roll(w: &WorldInstance) -> f64 {
    w.agent(0).vehicle.pose().rot.to_euler(EulerRot::ZYX).2
}

/// Standing still for 3 s on flat ground and on a slope as steep as their feet's
/// `max_slope_deg` (across it and diagonally up and down it), both presets stay on their feet
/// without a terminal event (the feet are gear): upright on the flat, leaning downhill on the
/// slope.
#[test]
fn spawn_upright_on_their_feet() {
    for (vehicle, slope) in [("motorcycle_sport", 15.0), ("bicycle_city", 12.0)] {
        let def = autonomousim_vehicles::SharedDef::from(autonomousim_vehicles::presets::get(vehicle).unwrap());
        assert_eq!(def.as_wheeled().unwrap().feet.unwrap().max_slope_deg, slope);
        // Up towards +x. Heading +y, uphill is to the right: the bike leans left.
        let incline = format!(r#"{{ type = "testworld", kind = "incline", size = 200.0, angle_deg = {slope} }}"#);
        let cases = [(FLAT, 0.0, -1.0), (FLAT, 90.0, -1.0), (&incline, 90.0, -1.0), (&incline, -90.0, 1.0)];
        let diagonal = [45.0, 135.0, -45.0, -135.0].map(|yaw: f64| (incline.as_str(), yaw, -yaw.signum()));
        for (map, yaw, side) in cases.into_iter().chain(diagonal) {
            let sc = scenario(vehicle, map, yaw);
            assert_eq!(sc.groups[0].obs_dim(), OBS_DIM);
            let mut w = WorldInstance::new(sc.clone(), Seed::from_u64(1));
            let start = w.agent(0).vehicle.pose().pos;
            let initial = roll(&w);
            let mut seen = Events::NONE;
            for _ in 0..3 * sc.spec.policy_hz {
                w.set_actions(0, &[0.0, 0.0]);
                w.step();
                seen = Events(seen.0 | events(&w).0);
            }
            let r = roll(&w);
            let moved = (w.agent(0).vehicle.pose().pos - start).length();
            let msg =
                format!("{vehicle} {map} yaw {yaw}: spawned at {initial:.3} rad, now {r:.3} rad, moved {moved:.3} m");
            assert!(!seen.intersects(Events::TERMINAL), "{msg}: {seen:?}");
            if map == FLAT {
                assert!(r.abs() < 1e-3 && moved < 0.01, "{msg}");
            } else {
                assert!(side * r > 0.05 && side * r < 0.45, "{msg}");
            }
            assert!((r - initial).abs() < 0.12 && moved < 0.1, "{msg}");
            let mut obs = [0.0f32; OBS_DIM];
            w.observe(0, &mut obs);
            assert_eq!(obs[6], 1.0, "{msg}: feet down");
        }
    }
}

/// The motorcycle launches from its feet, rides a curve in `vk` and records; its terms match
/// the vehicle's state.
#[test]
fn motorcycle_rides_observes_and_records() {
    let sc = scenario("motorcycle_sport", FLAT, 0.0);
    let path = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("motorcycle.mcap");
    let mut b = BatchSim::from_compiled(sc.clone(), 1, 1, 1).unwrap();
    b.attach_recorder(0, Recorder::create(&path, RecorderConfig { state_hz: 20, ..Default::default() }).unwrap());
    let hz = sc.spec.policy_hz as usize;
    let mut leaned = 0.0f64;
    // The lean term's rate against the change of its angle (trapezoidal rule over a step).
    let (mut prev, mut fastest, mut rate_error) = ([0.0f32; 2], 0.0f64, 0.0f64);
    for k in 0..16 * hz {
        // 10 m/s, then a left turn at a fifth of the full curvature.
        let turn = if k >= 8 * hz { 0.2 } else { 0.0 };
        b.step(&[&[0.5, turn]]);
        let w = b.world(0);
        let e = events(w);
        assert!(!e.intersects(Events::TERMINAL), "{e:?} at {:.2} s", k as f64 / hz as f64);
        leaned = leaned.min(roll(w));
        let mut obs = [0.0f32; OBS_DIM];
        w.observe(0, &mut obs);
        if k > 0 {
            let change = f64::from(obs[0] - prev[0]) * hz as f64;
            rate_error = rate_error.max((change - 0.5 * f64::from(obs[1] + prev[1])).abs());
            fastest = fastest.max(f64::from(obs[1]).abs());
        }
        prev = [obs[0], obs[1]];
    }
    assert!(fastest > 0.2 && rate_error < 0.05 * fastest, "roll rates up to {fastest}, error {rate_error}");
    let w = b.world(0);
    let bike = w.agent(0).vehicle.as_wheeled().unwrap();
    let v = &w.agent(0).vehicle;
    let (pose, rates) = (v.pose(), v.ang_vel_body());
    let speed = v.lin_vel_world().length();
    assert!((speed - 10.0).abs() < 0.5, "speed {speed}");
    assert!(!bike.feet_down());
    // Turning left, it leans left.
    assert!(leaned < -0.1, "lean {leaned}");

    let mut obs = [0.0f32; OBS_DIM];
    w.observe(0, &mut obs);
    let (_, pitch, r) = pose.rot.to_euler(EulerRot::ZYX);
    let rate = rates.x + (rates.y * r.sin() + rates.z * r.cos()) * pitch.tan();
    let (steer, steer_rate) = bike.steering_head().unwrap();
    let (rider, rider_rate) = bike.rider_lean();
    let expected = [r, rate, steer, steer_rate, rider, rider_rate, 0.0];
    for (i, (&o, e)) in obs.iter().zip(expected).enumerate() {
        assert!((f64::from(o) - e).abs() < 1e-5 * (1.0 + e.abs()), "term value {i}: {o} against {e}");
    }
    b.detach_recorder(0).unwrap().finish().unwrap();
    let rec = Recording::read(&path).unwrap();
    let last = rec.episodes[0].states[0].last().unwrap();
    let w = b.world(0);
    let bike = w.agent(0).vehicle.as_wheeled().unwrap();
    let (_, w_head) = bike.def().steering_head().unwrap();
    assert_eq!(last.position, bike.position());
    assert_eq!(last.steering, bike.steering_angle());
    assert_eq!(last.steer_torque, bike.wheel(w_head).steer_torque);
    assert_eq!(last.joints, bike.joints());
    assert_eq!(*last.joints.last().unwrap(), bike.rider_lean().0);
    assert!(!last.feet && last.steer_torque != 0.0);
    assert!(rec.episodes[0].states[0][0].feet, "starts on its feet");
    rec.compile().unwrap();
}

/// Riding with a steady steering torque and no balance, the motorcycle falls over: the rider
/// or the bodywork touching the ground is a crash.
#[test]
fn falling_over_is_a_crash() {
    let sc = scenario("motorcycle_sport", FLAT, 0.0);
    let mut w = WorldInstance::new(sc.clone(), Seed::from_u64(2));
    let hz = sc.spec.policy_hz as usize;
    for _ in 0..4 * hz {
        w.set_actions(0, &[0.5, 0.0]);
        w.step();
    }
    assert!(!events(&w).intersects(Events::TERMINAL));
    let mut seen = Events::NONE;
    for _ in 0..4 * hz {
        w.set_command(0, GroundSetpoint::Pedal { drive: 0.0, steering: 0.3, handbrake: false, lean: 0.0 });
        w.step();
        seen = Events(seen.0 | events(&w).0);
        if seen.intersects(Events::TERMINAL) {
            break;
        }
    }
    assert!(seen.intersects(Events::CRASH_TERRAIN), "{seen:?}, roll {:.2}", roll(&w));
    assert!(w.agent(0).vehicle.pose().rot * DVec3::Z != DVec3::Z);
}
