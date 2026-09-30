//! Kinematic and hybrid ground vehicles (M8b step 1): the kinematic model against the full one,
//! promotion and demotion, determinism.

use autonomousim_control::ground::GroundSetpoint;
use autonomousim_core::math::Pose;
use autonomousim_core::rng::Seed;
use autonomousim_sim::{BatchSim, CompiledScenario, Scenario, WorldInstance};
use glam::{DVec2, DVec3};
use std::sync::Arc;

fn compile(toml: &str) -> Arc<CompiledScenario> {
    Arc::new(Scenario::from_toml(toml).unwrap().compile().unwrap())
}

fn one_car(physics: &str) -> String {
    format!(
        r#"
        physics_hz = 1000
        policy_hz = 50
        map = {{ type = "testworld", kind = "flat", size = 600.0 }}
        [[groups]]
        name = "car"
        vehicle = "sedan_like"
        physics = "{physics}"
        spawn = {{ region = [[-250.0, 0.0], [-250.0, 0.0]], yaw_deg = [0.0, 0.0] }}
        "#
    )
}

/// Start agent `i` at its pose rolling forward at `speed`.
fn launch(w: &mut WorldInstance, i: usize, speed: f64) {
    let pose = w.agent(i).vehicle.pose();
    let v = pose.rot * DVec3::X * speed;
    w.place_agent(i, pose, DVec3::new(v.x, v.y, 0.0), DVec3::ZERO);
}

/// A 3.5 m lane change over 60 m at 15 m/s after 2 s straight, then 2 s straight: the
/// chassis origin's track.
fn lane_change(physics: &str) -> Vec<DVec2> {
    let sc = compile(&one_car(physics));
    let mut w = WorldInstance::new(sc, Seed::from_u64(1));
    let (v, d, s_len) = (15.0, 3.5, 60.0);
    launch(&mut w, 0, v);
    let tau = std::f64::consts::TAU;
    let mut track = Vec::new();
    for k in 0..400 {
        let s = v * (k as f64 * 0.02 - 2.0);
        let curvature = if (0.0..s_len).contains(&s) { d / s_len * tau / s_len * (tau * s / s_len).sin() } else { 0.0 };
        w.set_command(0, GroundSetpoint::SpeedCurvature { speed: v, curvature });
        w.step();
        track.push(w.agent(0).vehicle.position().truncate());
    }
    track
}

/// Lateral distance from `p` to the track `line` (running along +x) at `p.x`; `None` outside it.
fn to_line(p: DVec2, line: &[DVec2]) -> Option<f64> {
    line.windows(2).find(|s| (s[0].x..=s[1].x).contains(&p.x)).map(|s| {
        let t = (p.x - s[0].x) / (s[1].x - s[0].x);
        (p.y - (s[0].y + t * (s[1].y - s[0].y))).abs()
    })
}

#[test]
fn kinematic_lane_change_follows_the_full_model() {
    let (full, kin) = (lane_change("full"), lane_change("kinematic"));
    // Both change lanes by about 3.5 m.
    let offset = |t: &[DVec2]| t.last().unwrap().y - t[0].y;
    assert!(
        (offset(&full) - 3.5).abs() < 0.5 && (offset(&kin) - 3.5).abs() < 0.5,
        "{} {}",
        offset(&full),
        offset(&kin)
    );
    // The paths agree within 0.2 m, and so do the positions over time.
    let path = kin.iter().filter_map(|&p| to_line(p, &full)).fold(0.0, f64::max);
    let timed = kin.iter().zip(&full).map(|(a, b)| a.distance(*b)).fold(0.0, f64::max);
    eprintln!("lane change: path deviation {path:.3} m, position deviation {timed:.3} m");
    assert!(path < 0.2, "path deviation {path}");
    assert!(timed < 0.5, "position deviation {timed}");
}

/// A hybrid car driving straight at 12 m/s, and a learning car parked away from it.
const HYBRID: &str = r#"
    physics_hz = 1000
    policy_hz = 50
    map = { type = "testworld", kind = "flat", size = 600.0 }
    [[groups]]
    name = "npc"
    vehicle = "sedan_like"
    physics = "hybrid"
    hybrid = { promote = 20.0, demote = 30.0, calm = 1.0 }
    spawn = { region = [[-250.0, 0.0], [-250.0, 0.0]], yaw_deg = [0.0, 0.0] }
    [[groups]]
    name = "ego"
    vehicle = "sedan_like"
    spawn = { region = [[200.0, 200.0], [200.0, 200.0]], yaw_deg = [0.0, 0.0] }
"#;

fn park(w: &mut WorldInstance, i: usize, xy: DVec2) {
    let pose = w.agent(i).vehicle.pose();
    w.place_agent(i, Pose::new(xy.extend(pose.pos.z), pose.rot), DVec3::ZERO, DVec3::ZERO);
}

fn speed(w: &WorldInstance, i: usize) -> f64 {
    w.agent(i).vehicle.lin_vel_world().truncate().length()
}

#[test]
fn promotion_and_demotion_are_seamless() {
    let sc = compile(HYBRID);
    let mut w = WorldInstance::new(sc, Seed::from_u64(2));
    assert!(w.agent(0).is_kinematic() && !w.agent(1).is_kinematic());
    launch(&mut w, 0, 12.0);
    assert!(w.agent(0).is_kinematic());
    let drive = |w: &mut WorldInstance| {
        w.set_command(0, GroundSetpoint::SpeedCurvature { speed: 12.0, curvature: 0.0 });
        w.set_command(1, GroundSetpoint::SpeedCurvature { speed: 0.0, curvature: 0.0 });
        w.step();
    };
    for _ in 0..50 {
        drive(&mut w);
    }
    assert!(w.agent(0).is_kinematic());
    // Park the learning car beside the road ahead: promoted within 20 m.
    let x = w.agent(0).vehicle.position().x;
    park(&mut w, 1, DVec2::new(x + 30.0, 8.0));
    let mut promoted = None;
    for k in 0..100 {
        let (pose, v) = (w.agent(0).vehicle.pose(), speed(&w, 0));
        w.switch_physics();
        if !w.agent(0).is_kinematic() {
            // Exactly where it was, at the same speed.
            assert_eq!(w.agent(0).vehicle.pose(), pose);
            assert!((speed(&w, 0) - v).abs() < 1e-9);
            promoted = Some((k, v));
            break;
        }
        drive(&mut w);
    }
    let (_, v0) = promoted.expect("promoted near the learning car");
    // In full, no jolt: the speed stays within 0.05 m/s and the tyre loads near their static
    // values over the next second.
    let loads: Vec<f64> = w.agent(0).vehicle.as_wheeled().unwrap().def().rest_state().unwrap().loads.clone();
    let (mut dv, mut dfz) = (0.0f64, 0.0f64);
    for _ in 0..50 {
        drive(&mut w);
        assert!(!w.agent(0).is_kinematic());
        dv = dv.max((speed(&w, 0) - v0).abs());
        for (wh, l) in w.agent(0).vehicle.as_wheeled().unwrap().wheels().zip(&loads) {
            dfz = dfz.max((wh.tire.fz - l).abs() / l);
        }
    }
    eprintln!("after promotion: speed deviation {dv:.4} m/s, tyre load deviation {:.1} %", 100.0 * dfz);
    assert!(dv < 0.05, "speed deviation {dv}");
    assert!(dfz < 0.1, "tyre load deviation {dfz}");
    // Past the learning car and 30 m on, it is demoted once calm: where it was, at its speed.
    let mut demoted = false;
    for _ in 0..300 {
        let (pose, v) = (w.agent(0).vehicle.pose(), speed(&w, 0));
        w.switch_physics();
        if w.agent(0).is_kinematic() {
            let now = w.agent(0).vehicle.pose();
            assert_eq!(now.pos.truncate(), pose.pos.truncate());
            assert!((now.pos.z - pose.pos.z).abs() < 0.02, "height {} → {}", pose.pos.z, now.pos.z);
            assert!((speed(&w, 0) - v).abs() < 0.05, "speed {v} → {}", speed(&w, 0));
            let gap = w.agent(0).vehicle.position() - w.agent(1).vehicle.position();
            assert!(gap.x > 0.0 && gap.truncate().length() > 30.0, "{gap}");
            demoted = true;
            break;
        }
        drive(&mut w);
    }
    assert!(demoted, "demoted away from the learning car");
}

#[test]
fn kinematic_vehicles_push_one_sidedly() {
    // A kinematic car drives into a parked full car: the full car is shoved, the kinematic one
    // keeps its speed.
    let toml = r#"
        physics_hz = 1000
        policy_hz = 50
        map = { type = "testworld", kind = "flat", size = 600.0 }
        [[groups]]
        name = "npc"
        vehicle = "sedan_like"
        physics = "kinematic"
        spawn = { region = [[-20.0, 0.0], [-20.0, 0.0]], yaw_deg = [0.0, 0.0] }
        disable_on_terminal = false
        [[groups]]
        name = "ego"
        vehicle = "sedan_like"
        spawn = { region = [[0.0, 0.0], [0.0, 0.0]], yaw_deg = [0.0, 0.0] }
        disable_on_terminal = false
    "#;
    let mut w = WorldInstance::new(compile(toml), Seed::from_u64(4));
    launch(&mut w, 0, 1.5);
    let x1 = w.agent(1).vehicle.position().x;
    for _ in 0..600 {
        w.set_command(0, GroundSetpoint::SpeedCurvature { speed: 1.5, curvature: 0.0 });
        w.set_command(1, GroundSetpoint::SpeedCurvature { speed: 0.0, curvature: 0.0 });
        w.step();
        // Horizontal speed: 1.5 m/s along the chassis x axis, pitched at rest.
        assert!((speed(&w, 0) - 1.5).abs() < 2e-3, "{}", speed(&w, 0));
    }
    let pushed = w.agent(1).vehicle.position().x - x1;
    assert!(pushed > 2.0, "pushed {pushed} m");
}

/// 40 hybrid cars driving about and a learning car driving through them.
const CROWD: &str = r#"
    physics_hz = 500
    policy_hz = 25
    map = { type = "testworld", kind = "flat", size = 300.0 }
    [[groups]]
    name = "npc"
    count = 40
    vehicle = "sedan_like"
    physics = "hybrid"
    hybrid = { promote = 25.0, demote = 35.0, calm = 0.5 }
    action_mode = "vk"
    spawn = { region = [[-60.0, -60.0], [60.0, 60.0]], min_separation = 8.0 }
    disable_on_terminal = false
    [[groups]]
    name = "ego"
    vehicle = "sedan_like"
    spawn = { region = [[-100.0, 0.0], [-100.0, 0.0]], yaw_deg = [0.0, 0.0] }
    disable_on_terminal = false
"#;

fn crowd_run(threads: usize) -> ([u8; 32], usize) {
    let mut b = BatchSim::from_compiled(compile(CROWD), 2, 5, threads).unwrap();
    let npc: Vec<f32> = (0..80).flat_map(|k| [0.3, if k % 2 == 0 { 0.2 } else { -0.2 }]).collect();
    let mut switches = 0;
    let mut modes: Vec<bool> = (0..40).map(|i| b.world(0).agent(i).is_kinematic()).collect();
    for _ in 0..200 {
        b.step(&[&npc, &[0.5, 0.0, 0.5, 0.0]]);
        for (i, m) in modes.iter_mut().enumerate() {
            let now = b.world(0).agent(i).is_kinematic();
            switches += usize::from(now != *m);
            *m = now;
        }
    }
    let mut h = blake3::Hasher::new();
    for i in 0..2 {
        h.update(&b.world(i).state_hash());
    }
    (*h.finalize().as_bytes(), switches)
}

#[test]
fn hybrid_switching_is_deterministic_across_threads() {
    let (one, switches) = crowd_run(1);
    assert!(switches >= 4, "{switches} switches");
    let (four, again) = crowd_run(4);
    assert_eq!(one, four);
    assert_eq!(switches, again);
}

#[test]
fn physics_modes_are_checked() {
    let base = one_car("kinematic");
    // Kinematic groups need a speed command: a driver or `vk`/`vw`.
    let raw = base.replace("physics = \"kinematic\"", "physics = \"kinematic\"\naction_mode = \"raw\"");
    let err = Scenario::from_toml(&raw).and_then(Scenario::compile).unwrap_err().to_string();
    assert!(err.contains("`vk` or `vw`"), "{err}");
    let drone = "[[groups]]\nphysics = \"hybrid\"";
    let err = Scenario::from_toml(drone).and_then(Scenario::compile).unwrap_err().to_string();
    assert!(err.contains("need ground vehicles"), "{err}");
    let bad =
        base.replace("physics = \"kinematic\"", "physics = \"hybrid\"\nhybrid = { promote = 50.0, demote = 40.0 }");
    let err = Scenario::from_toml(&bad).and_then(Scenario::compile).unwrap_err().to_string();
    assert!(err.contains("promote < demote"), "{err}");
    // Full physics is the default and is not written out.
    let sc = compile(&one_car("full"));
    assert!(!sc.spec.to_json().contains("\"physics\":"));
    assert!(compile(&base).spec.to_json().contains("\"kinematic\""));
}
