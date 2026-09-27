//! `path` goals: planned paths from the spawn through random goals off the roads, followed as
//! the agent's route, and recorded.

use autonomousim_core::math::quat::{wrap_angle, yaw};
use autonomousim_core::rng::Seed;
use autonomousim_sim::{BatchSim, CompiledScenario, Scenario, WorldInstance};
use std::sync::Arc;

fn compile(toml: &str) -> Arc<CompiledScenario> {
    Arc::new(Scenario::from_toml(toml).unwrap().compile().unwrap())
}

const CROSS_COUNTRY: &str = r#"
    name = "cross_country"
    physics_hz = 1000
    map = { type = "rural", seed = 3, count = 2, cache = false }
    [[groups]]
    name = "apc"
    count = 2
    vehicle = "tracked_apc"
    action_mode = "vw"
    drivable = { max_slope_deg = 30.0, spawn_slope_deg = 10.0, margin = 2.0, resistance_cost = 30.0 }
    spawn = { margin = 40.0, min_separation = 20.0, yaw_deg = [0.0, 0.0] }
    goals = { kind = "random", count = 3, distance = [40.0, 100.0], margin = 40.0, radius = 4.0, off_road = true, path = true }
    obs = [{ term = "road" }, { term = "route" }]
"#;

#[test]
fn path_goals_plan_legs_through_the_goals() {
    let sc = compile(CROSS_COUNTRY);
    let mut w = WorldInstance::new(sc.clone(), Seed::from_u64(4));
    let (mut goals, mut on_road) = (0, 0);
    for episode in 0..6 {
        if episode > 0 {
            w.reset(None);
        }
        let map = w.map().clone();
        let grid = &sc.groups[0].drive[w.map_index()];
        for a in w.agents() {
            let p = a.vehicle.position().truncate();
            assert_eq!(a.legs.len(), a.goals.len(), "{}: one leg per goal", a.id);
            let mut from = p;
            for (leg, goal) in a.legs.iter().zip(&a.goals) {
                let pts = leg.points();
                // From the previous goal (the spawn first: turned to face the path, the vehicle
                // stands a little off where the path was planned from) to the goal, over drivable
                // ground.
                let d0 = pts[0].truncate().distance(from);
                assert!(d0 < 0.3, "{}: leg starts {d0} m from {from}", a.id);
                assert!(pts.last().unwrap().truncate().distance(goal.position.truncate()) < 1e-6);
                let off = pts.iter().filter(|q| !grid.is_drivable(q.truncate())).count();
                assert!(off * 20 <= pts.len(), "{}: {off} of {} path points blocked", a.id, pts.len());
                from = goal.position.truncate();
                goals += 1;
                on_road += usize::from(map.roads().on_road(from).is_some());
            }
            // The route is the first leg, and the vehicle faces along it (`yaw_deg` 0).
            assert!(Arc::ptr_eq(a.route.as_ref().unwrap(), &a.legs[0]));
            let ahead = a.legs[0].points().iter().map(|q| q.truncate()).find(|q| q.distance(p) >= 8.0);
            if let Some(q) = ahead {
                let d = q - p;
                let err = wrap_angle(d.y.atan2(d.x) - yaw(a.vehicle.orientation()));
                assert!(err.abs() < 0.1, "{}: heading off the path by {err}", a.id);
            }
        }
    }
    assert!(on_road * 10 <= goals, "{on_road} of {goals} goals on a road");

    // Reaching a goal switches the route to the next leg; the last one stays.
    let legs = w.agent(0).legs.clone();
    for k in 1..legs.len() {
        assert!(w.advance_goal(0));
        assert!(Arc::ptr_eq(w.agent(0).route.as_ref().unwrap(), &legs[k]));
    }
    assert!(!w.advance_goal(0));
    assert!(Arc::ptr_eq(w.agent(0).route.as_ref().unwrap(), legs.last().unwrap()));
}

#[test]
fn path_goals_need_random_goals_of_ground_vehicles() {
    let err = |toml: String| Scenario::from_toml(&toml).unwrap().compile().unwrap_err().to_string();
    let drone = CROSS_COUNTRY
        .replace(r#"vehicle = "tracked_apc""#, r#"vehicle = "cf2x""#)
        .replace(r#"action_mode = "vw""#, "")
        .replace(
            "drivable = { max_slope_deg = 30.0, spawn_slope_deg = 10.0, margin = 2.0, resistance_cost = 30.0 }",
            "",
        );
    let e = err(drone);
    assert!(e.contains("goals.path"), "{e}");
    let spawn = CROSS_COUNTRY.replace(r#"kind = "random", count = 3"#, r#"kind = "spawn", count = 1"#);
    assert!(err(spawn).contains("goals.path"));
}

#[test]
fn recordings_keep_the_whole_path() {
    use autonomousim_sim::record::{Recorder, RecorderConfig, Recording};

    let sc = compile(CROSS_COUNTRY);
    let path = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("paths.mcap");
    let mut b = BatchSim::from_compiled(sc.clone(), 1, 2, 1).unwrap();
    b.attach_recorder(0, Recorder::create(&path, RecorderConfig::default()).unwrap());
    let whole: Vec<Vec<_>> = b
        .world(0)
        .agents()
        .iter()
        .map(|a| {
            a.legs.iter().enumerate().flat_map(|(k, l)| l.points()[usize::from(k > 0)..].iter().copied()).collect()
        })
        .collect();
    b.step(&[&[0.0; 4]]);
    b.detach_recorder(0).unwrap().finish().unwrap();
    let rec = Recording::read(&path).unwrap();
    let ep = &rec.episodes[0];
    for (recorded, points) in ep.routes.iter().zip(&whole) {
        assert_eq!(recorded.as_deref().unwrap().points(), &points[..]);
    }
}
