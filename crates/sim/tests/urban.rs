//! Urban sites in the simulation: drones land on rooftop pads, cars get bay goals in parking
//! lots and parking lanes.

use autonomousim_control::ground::GroundSetpoint;
use autonomousim_core::math::Pose;
use autonomousim_core::math::quat::wrap_angle;
use autonomousim_core::rng::Seed;
use autonomousim_sim::{CompiledScenario, Events, STATE_DIM, STATE_FIELDS, Scenario, WorldInstance};
use autonomousim_world::BayKind;
use glam::{DQuat, DVec2, DVec3};
use std::sync::Arc;

fn compile(toml: &str) -> Arc<CompiledScenario> {
    Arc::new(Scenario::from_toml(toml).unwrap().compile().unwrap())
}

/// A quadrotor set down just above each rooftop pad with its motors off comes to rest on the
/// pad: landed, no crash.
#[test]
fn drones_rest_on_rooftop_pads() {
    let sc = compile(
        r#"
        map = { type = "urban", seed = 1, count = 1 }
        [[groups]]
        vehicle = "iris_like"
        "#,
    );
    let mut w = WorldInstance::new(sc.clone(), Seed::from_u64(0));
    let pads = w.map().sites().pads.clone();
    assert!(!pads.is_empty());
    let bottom = sc.groups[0].bottom;
    for (k, pad) in pads.iter().enumerate() {
        w.reset(None);
        let pose = Pose::new(pad.centre + DVec3::Z * (bottom + 0.05), DQuat::from_rotation_z(pad.yaw));
        w.place_agent(0, pose, DVec3::ZERO, DVec3::ZERO);
        let mut seen = Events::NONE;
        for _ in 0..100 {
            w.set_action(0, &[0.0, 0.0, 0.0, -1.0]);
            w.step();
            seen |= w.agent(0).events;
        }
        let p = w.agent(0).vehicle.position();
        assert!(!seen.is_terminal() && w.agent(0).events.contains(Events::LANDED), "pad {k}: {seen:?}");
        assert!((p.z - bottom - pad.centre.z).abs() < 0.05, "pad {k}: at {p}, pad at {}", pad.centre);
        assert!(p.truncate().distance(pad.centre.truncate()) < 0.2);
    }
}

fn tail(w: &WorldInstance) -> (DVec2, f64) {
    let mut s = vec![0.0; STATE_DIM];
    w.write_state(0, &mut s);
    let k: usize = STATE_FIELDS.iter().take_while(|(n, _)| *n != "tail").map(|(_, d)| d).sum();
    (DVec2::new(s[k], s[k + 1]), s[k + 2])
}

/// Bay goals on urban maps: the goal is the tail at a marked bay of the kinds asked for (at
/// the inner end of a lot's bay facing out, at the rear end of a street's bay facing along it);
/// the car spawns in line, its tail the sampled distance from the bay in the aisle or the lane
/// beside, and settles there without events.
#[test]
fn bay_goals_in_lots_and_on_streets() {
    for kind in [BayKind::Lot, BayKind::Street] {
        let name = if kind == BayKind::Lot { "lot" } else { "street" };
        let sc = compile(&format!(
            r#"
            map = {{ type = "urban", seed = 2, count = 1 }}
            [[groups]]
            vehicle = "sedan_like"
            action_mode = "vk"
            goals = {{ kind = "bay", distance = [6.0, 12.0], bay = {{ kinds = ["{name}"] }} }}
            "#
        ));
        let mut w = WorldInstance::new(sc, Seed::from_u64(0));
        for episode in 0..10u64 {
            w.reset(Some(episode));
            let goal = w.agent(0).goal();
            let bays = &w.map().sites().bays;
            let bay = bays
                .iter()
                .filter(|b| b.kind == kind)
                .min_by(|a, b| {
                    let d = |x: &autonomousim_world::ParkingBay| x.centre.truncate().distance(goal.position.truncate());
                    d(a).total_cmp(&d(b))
                })
                .unwrap();
            let half = 0.5 * bay.size.x - 0.3;
            let (end, yaw) = match kind {
                BayKind::Lot => (bay.point(DVec2::new(half, 0.0)), bay.yaw + std::f64::consts::PI),
                BayKind::Street => (bay.point(DVec2::new(-half, 0.0)), bay.yaw),
            };
            assert!(end.distance(goal.position.truncate()) < 1e-9, "{name} {episode}");
            assert!(wrap_angle(goal.yaw - yaw).abs() < 1e-9);
            w.set_command(0, GroundSetpoint::SpeedCurvature { speed: 0.0, curvature: 0.0 });
            for _ in 0..(2.0 / w.scenario().policy_dt()) as usize {
                w.step();
                assert!(!w.agent(0).events.is_terminal(), "{name} {episode}: {:?}", w.agent(0).events);
            }
            let (t, heading) = tail(&w);
            let d = t.distance(goal.position.truncate());
            assert!((2.0..16.0).contains(&d), "{name} {episode}: tail {d} m from the bay");
            let turn = wrap_angle(heading - goal.yaw).abs();
            let expect = if kind == BayKind::Lot { std::f64::consts::FRAC_PI_2 } else { 0.0 };
            assert!((turn - expect).abs() < 0.25, "{name} {episode}: heading {turn}");
        }
    }
}
