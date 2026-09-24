//! Ground vehicles on generated terrain: spawns on drivable ground, reachable goals, ground
//! events and observation terms, and a batch of 64 cars driving to their goals on an `offroad`
//! wild map.

use autonomousim_core::geometry::{HitMask, StaticGeometry};
use autonomousim_core::math::Pose;
use autonomousim_core::math::quat::yaw;
use autonomousim_core::rng::Seed;
use autonomousim_core::terrain::Terrain;
use autonomousim_sim::drive::{self, DriveGrid};
use autonomousim_sim::{Agent, BatchSim, CompiledScenario, Events, Scenario, WorldInstance};
use glam::{DQuat, DVec2, DVec3};
use std::sync::Arc;

fn compile(toml: &str) -> Arc<CompiledScenario> {
    Arc::new(Scenario::from_toml(toml).unwrap().compile().unwrap())
}

/// 4×4s with chained goals on small offroad maps, routed with 2 m to spare beside trunks; the
/// observation starts with the goal in the heading frame.
const OFFROAD: &str = r#"
    name = "offroad"
    map = { type = "wild", preset = "offroad", seed = 7, count = 2, cache = false, config = { size = 256.0 } }
    [[groups]]
    name = "trucks"
    count = 8
    vehicle = "offroad_4x4"
    drivable = { margin = 2.0 }
    spawn = { min_separation = 8.0 }
    goals = { kind = "random", count = 3, distance = [25.0, 50.0], radius = 3.0 }
    obs = [
        { term = "goal_rel_heading" }, { term = "speed" }, { term = "sideslip" }, { term = "pitch_roll" },
        { term = "wheel_speeds" }, { term = "wheel_slip" }, { term = "steering" }, { term = "gear_rpm" },
    ]
"#;

#[test]
fn spawns_and_goals_are_on_drivable_reachable_ground() {
    let sc = compile(OFFROAD);
    // Ground vehicles choose the 1 kHz preset.
    assert_eq!((sc.spec.physics_hz, sc.decimation), (1000, 20));
    let g = &sc.groups[0];
    assert_eq!(g.drive.len(), 2);
    for grid in &g.drive {
        assert!(grid.drivable_share() > 0.5, "{}", grid.drivable_share());
    }
    let mut w = WorldInstance::new(sc.clone(), Seed::from_u64(1));
    for episode in 0..4 {
        if episode > 0 {
            w.reset(None);
        }
        let map = w.map().clone();
        let grid = &g.drive[w.map_index()];
        for a in w.agents() {
            let p = a.vehicle.position();
            let xy = p.truncate();
            assert_eq!(grid.component(xy), Some(grid.largest()), "spawn {p} off the main area");
            assert!(drive::slope(&map, xy, 1.5) <= 15f64.to_radians(), "spawn {p} too steep");
            // Clear of trunks and rocks by the vehicle's radius.
            assert!(map.obstacles().nearest_distance(p, g.radius, HitMask::SOLID).is_none_or(|d| d >= g.radius));
            // Aligned with the ground below it (over the wheelbase, within its roughness).
            let h = |dx: f64, dy: f64| map.terrain().height(p.x + dx, p.y + dy);
            let n = DVec3::new(h(-1.5, 0.0) - h(1.5, 0.0), h(0.0, -1.5) - h(0.0, 1.5), 3.0).normalize();
            let up = a.vehicle.orientation() * DVec3::Z;
            assert!(up.angle_between(n) < 8f64.to_radians(), "{p}: up {up}, normal {n}");
            // Goals lie on the ground (at the rest height) and can be driven to.
            let mut prev = xy;
            for goal in &a.goals {
                let q = goal.position.truncate();
                assert!(grid.reachable(xy, q), "goal {q} unreachable from {xy}");
                let path = grid.path(prev, q).unwrap();
                assert!(path.len() >= 2);
                let h = map.terrain().height(q.x, q.y);
                assert!((goal.position.z - h - g.rest.pos.z).abs() < 1e-9);
                prev = q;
            }
        }
        // Standing still on the slopes for a second: no events, hardly any creep.
        let start: Vec<DVec3> = w.agents().iter().map(|a| a.vehicle.position()).collect();
        for _ in 0..50 {
            w.set_actions(0, &[0.0; 16]);
            w.step();
            for a in w.agents() {
                assert!(a.events.is_empty() || a.events == Events::GROUND_CONTACT, "{}: {:?}", a.id, a.events);
            }
        }
        for (a, s) in w.agents().iter().zip(&start) {
            assert!(a.vehicle.position().distance(*s) < 0.1, "{}: crept {}", a.id, a.vehicle.position() - *s);
        }
    }
}

#[test]
fn ground_observation_terms() {
    let sc = compile(OFFROAD);
    let g = &sc.groups[0];
    // 3 + speed, sideslip + 2 + 4 wheel speeds + 4 slips + steering + gear, rpm.
    assert_eq!(g.obs_dim(), 3 + 1 + 1 + 2 + 4 + 4 + 1 + 2);
    let layout = g.obs.layout();
    assert_eq!(layout[4], ("wheel_speeds".to_string(), 7, 4));
    let mut w = WorldInstance::new(sc.clone(), Seed::from_u64(2));
    // Ahead at a third of full speed, turning left.
    for _ in 0..150 {
        w.set_actions(0, &[0.33, 0.3].repeat(8));
        w.step();
    }
    let mut obs = vec![0.0f32; 8 * g.obs_dim()];
    w.observe(0, &mut obs);
    for (k, a) in w.agents().iter().enumerate() {
        if a.events.is_terminal() {
            continue;
        }
        let o = &obs[k * g.obs_dim()..(k + 1) * g.obs_dim()];
        let v = a.vehicle.lin_vel_body();
        let rel = a.goal().position - a.vehicle.position();
        let heading =
            autonomousim_core::math::quat::from_yaw(autonomousim_core::math::quat::yaw(a.vehicle.orientation()));
        assert!((f64::from(o[0]) - (heading.inverse() * rel).x).abs() < 1e-3);
        assert!((f64::from(o[3]) - v.x).abs() < 1e-4 && v.x > 1.0, "{} vs {v}", o[3]);
        assert!(f64::from(o[4]).abs() < 0.2, "sideslip {}", o[4]);
        let (_, pitch, roll) = a.vehicle.orientation().to_euler(glam::EulerRot::ZYX);
        assert!((f64::from(o[5]) - pitch).abs() < 1e-6 && (f64::from(o[6]) - roll).abs() < 1e-6);
        // Rolling wheels turn at about the ground speed; slip is small.
        for w in 0..4 {
            assert!((f64::from(o[7 + w]) - v.x).abs() < 0.2 * v.x, "wheel {w}: {} vs {}", o[7 + w], v.x);
            assert!(o[11 + w].abs() < 0.1, "slip {w}: {}", o[11 + w]);
        }
        // Steering left, in a forward gear with the engine turning.
        assert!(o[15] > 0.01, "steering {}", o[15]);
        assert!(o[16] >= 1.0 && o[17] > 0.5, "gear {} at {} krpm", o[16], o[17]);
    }
    // Wheel terms need a ground vehicle.
    let err = Scenario::from_toml("[[groups]]\nobs = [ { term = \"wheel_slip\" } ]")
        .and_then(Scenario::compile)
        .unwrap_err()
        .to_string();
    assert!(err.contains("needs a ground vehicle"), "{err}");
}

/// A sedan on a flat test world with short stuck time, and a lake.
const EVENTS: &str = r#"
    map = { type = "testworld", kind = "lake", size = 200.0, depth = 4.0, water_level = -1.0 }
    events = { ground = { stuck_time = 1.0, stuck_distance = 0.3 } }
    [[groups]]
    count = 3
    vehicle = "sedan_like"
    spawn = { min_separation = 10.0, region = [[-90.0, -90.0], [-40.0, 90.0]] }
"#;

#[test]
fn rollover_stuck_and_water_events() {
    let sc = compile(EVENTS);
    let mut w = WorldInstance::new(sc.clone(), Seed::from_u64(4));
    // Agent 0 on its side, agent 1 in the middle of the lake, agent 2 parked.
    let a0 = w.agent(0).vehicle.pose();
    let side = Pose::new(a0.pos + DVec3::Z * 1.0, a0.rot * DQuat::from_rotation_x(1.4));
    w.agent_mut(0).vehicle.place(side, DVec3::ZERO, DVec3::ZERO);
    let lake = Pose::new(DVec3::new(0.0, 0.0, -2.5), DQuat::IDENTITY);
    w.agent_mut(1).vehicle.place(lake, DVec3::ZERO, DVec3::ZERO);
    let mut seen = [Events::NONE; 3];
    let mut first_stuck = None;
    for k in 0..75 {
        w.set_actions(0, &[0.0; 6]);
        w.step();
        for (s, a) in seen.iter_mut().zip(w.agents()) {
            *s |= a.events;
        }
        if first_stuck.is_none() && w.agent(2).events.contains(Events::STUCK) {
            first_stuck = Some(k + 1);
        }
    }
    assert!(seen[0].contains(Events::ROLLOVER | Events::DISABLED), "{:?}", seen[0]);
    assert!(seen[1].contains(Events::WATER | Events::DISABLED), "{:?}", seen[1]);
    // Stuck after a second at rest (50 policy steps), and not terminal.
    assert_eq!(first_stuck, Some(50));
    assert!(!w.agent(2).disabled && w.agent(2).events.contains(Events::STUCK));
    // Moving clears it.
    for _ in 0..50 {
        w.set_actions(0, &[0.5, 0.0].repeat(3));
        w.step();
    }
    assert!(!w.agent(2).events.contains(Events::STUCK), "{:?}", w.agent(2).events);
}

/// Pure pursuit of a point `LOOKAHEAD` m ahead along the grid path to the current goal, on the
/// `vk` action.
struct Pursuit {
    goal: DVec3,
    path: Vec<DVec2>,
}

const LOOKAHEAD: f64 = 8.0;

impl Pursuit {
    fn action(&mut self, a: &Agent, grid: &DriveGrid, v_max: f64, k_max: f64) -> [f32; 2] {
        let p = a.vehicle.position().truncate();
        let goal = a.goal().position;
        if goal != self.goal || self.path.is_empty() {
            self.goal = goal;
            self.path = grid.path(p, goal.truncate()).unwrap_or_else(|| vec![p, goal.truncate()]);
        }
        // Drop passed points, then aim at the first one beyond the lookahead.
        while self.path.len() > 1 && self.path[0].distance(p) < LOOKAHEAD {
            self.path.remove(0);
        }
        let target = self.path[0];
        let heading = yaw(a.vehicle.orientation());
        let rel = DVec2::from_angle(-heading).rotate(target - p);
        let curvature = 2.0 * rel.y / rel.length_squared().max(1.0);
        let speed = if rel.x > rel.y.abs() { 6.0 } else { 3.0 };
        [(speed / v_max) as f32, (curvature / k_max).clamp(-1.0, 1.0) as f32]
    }
}

/// 64 4×4s (8 worlds × 8) follow grid paths to their goals on offroad maps: they drive, most
/// reach goals, few hit trees or leave the map. (The follower ignores other agents, so
/// collisions between cars are only counted.)
#[test]
fn batch_of_64_drives_to_goals() {
    let sc = compile(OFFROAD);
    let g = &sc.groups[0];
    let map = g.action_map.as_ground().unwrap();
    let (v_max, k_max) = (map.speed(), map.curvature());
    let mut b = BatchSim::from_compiled(sc.clone(), 8, 3, 4).unwrap();
    let n = 8 * 8;
    let positions = |b: &BatchSim| -> Vec<DVec3> {
        (0..8).flat_map(|e| b.world(e).agents().iter().map(|a| a.vehicle.position())).collect()
    };
    let start = positions(&b);
    let mut pursuit: Vec<Pursuit> = (0..n).map(|_| Pursuit { goal: DVec3::NAN, path: Vec::new() }).collect();
    let mut reached = vec![0usize; n];
    let mut seen = vec![Events::NONE; n];
    let mut acts = vec![0.0f32; n * 2];
    // 20 s.
    for _ in 0..1000 {
        for e in 0..8 {
            let w = b.world(e);
            let grid = &g.drive[w.map_index()];
            for (k, a) in w.agents().iter().enumerate() {
                let i = e * 8 + k;
                acts[2 * i..2 * i + 2].copy_from_slice(&pursuit[i].action(a, grid, v_max, k_max));
            }
        }
        b.step(&[&acts]);
        for (i, e) in b.events(0).iter().enumerate() {
            let e = Events(*e);
            seen[i] |= e;
            if e.contains(Events::GOAL_REACHED) {
                reached[i] += 1;
            }
        }
    }
    let moved = positions(&b).iter().zip(&start).filter(|(p, s)| p.distance(**s) > 20.0).count();
    let crashed = seen.iter().filter(|e| e.is_terminal() && !e.contains(Events::CRASH_AGENT)).count();
    let collided = seen.iter().filter(|e| e.contains(Events::CRASH_AGENT)).count();
    let goals: usize = reached.iter().sum();
    let with_goal = reached.iter().filter(|r| **r > 0).count();
    println!("moved {moved}/64, crashed {crashed}, collided {collided}, goals {goals}, agents with a goal {with_goal}");
    assert!(seen.iter().all(|e| !e.contains(Events::NAN)));
    assert!(moved >= 48, "{moved} of 64 moved");
    assert!(crashed <= 6, "{crashed} crashed");
    assert!(with_goal >= 40 && goals >= 50, "{with_goal} reached {goals} goals");
}

#[test]
fn cars_do_not_depend_on_batch_size_or_threads() {
    let sc = compile(OFFROAD);
    let dim = sc.groups[0].spec.count * sc.groups[0].act_dim();
    let mut one = BatchSim::from_compiled(sc.clone(), 1, 9, 1).unwrap();
    let mut many = BatchSim::from_compiled(sc.clone(), 4, 9, 3).unwrap();
    let mut serial = BatchSim::from_compiled(sc, 4, 9, 1).unwrap();
    for k in 0..40 {
        let act = |n: usize| -> Vec<f32> { (0..n * dim).map(|i| ((k * 5 + i) as f32 * 0.37).sin()).collect() };
        one.step(&[&act(1)]);
        many.step(&[&act(4)]);
        serial.step(&[&act(4)]);
        assert_eq!(one.world(0).state_hash(), many.world(0).state_hash(), "step {k}");
    }
    for e in 0..4 {
        assert_eq!(many.world(e).state_hash(), serial.world(e).state_hash());
    }
    assert_eq!(many.obs(0), serial.obs(0));
    assert_eq!(many.events(0), serial.events(0));
    assert_eq!(one.obs(0), &many.obs(0)[..one.obs(0).len()]);
}
