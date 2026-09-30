//! The `traffic` driver (M8b step 2): IDM car following on a ring road (stop-and-go waves as
//! in Sugiyama et al. 2008, the equilibrium gaps), MOBIL lane changes, and a long run of
//! multi-lane traffic without collisions.

use autonomousim_core::math::Pose;
use autonomousim_core::math::quat::yaw;
use autonomousim_core::rng::Seed;
use autonomousim_sim::driver::Driver;
use autonomousim_sim::events::Events;
use autonomousim_sim::traffic_driver::{Elem, TrafficDriver};
use autonomousim_sim::{CompiledScenario, Scenario, WorldInstance};
use glam::{DQuat, DVec3};
use std::f64::consts::{PI, TAU};
use std::sync::Arc;

fn compile(toml: &str) -> Arc<CompiledScenario> {
    Arc::new(Scenario::from_toml(toml).unwrap().compile().unwrap())
}

/// Scenario of kinematic sedans driven by `traffic` drivers with the driver settings `driver`
/// on a ring road of `radius` with `lanes` (one-way: the lanes are centred on the road).
fn ring(radius: f64, lanes: &str, class: &str, groups: &[(&str, usize, &str)]) -> String {
    let mut s = format!(
        r#"
        physics_hz = 200
        policy_hz = 25
        map = {{ type = "testworld", kind = "ring", radius = {radius}, lanes = {lanes}, class = "{class}" }}
        "#
    );
    for (name, count, driver) in groups {
        s += &format!(
            r#"
            [[groups]]
            name = "{name}"
            count = {count}
            vehicle = "sedan_like"
            physics = "kinematic"
            driver = {{ type = "traffic", {driver} }}
            spawn = {{ on_ground = true, on_road = true, min_separation = 8.0 }}
            disable_on_terminal = false
            "#
        );
    }
    s
}

fn traffic(w: &WorldInstance, i: usize) -> &TrafficDriver {
    match w.agent(i).driver.as_ref() {
        Some(Driver::Traffic(d)) => d,
        _ => panic!("a traffic driver"),
    }
}

/// Angle of agent `i` round the ring (rad, counter-clockwise from +x).
fn angle(w: &WorldInstance, i: usize) -> f64 {
    let p = w.agent(i).vehicle.position();
    p.y.atan2(p.x)
}

fn speed(w: &WorldInstance, i: usize) -> f64 {
    w.agent(i).vehicle.lin_vel_world().truncate().length()
}

/// Put agent `i` on the ring at angle `a` and `radius`, facing counter-clockwise, at `v`.
fn put(w: &mut WorldInstance, i: usize, radius: f64, a: f64, v: f64) {
    let pose = w.agent(i).vehicle.pose();
    let heading = a + 0.5 * PI;
    let rot = DQuat::from_rotation_z(heading - yaw(pose.rot)) * pose.rot;
    let pos = DVec3::new(radius * a.cos(), radius * a.sin(), pose.pos.z);
    let vel = DVec3::new(-a.sin(), a.cos(), 0.0) * v;
    w.place_agent(i, Pose::new(pos, rot), vel, DVec3::ZERO);
}

/// Bumper-to-bumper gap from each of the first `n` cars to the one ahead along the lane
/// graph of a one-lane ring (lanes and connectors in a loop), as the drivers measure it (m).
fn gaps(w: &WorldInstance, n: usize) -> Vec<f64> {
    let g = w.map().roads().lanes();
    // The loop from lane 0: lane, connector, lane, connector.
    let mut start = std::collections::BTreeMap::new();
    let (mut lane, mut at) = (0u32, 0.0);
    for _ in 0..2 {
        start.insert(Elem::Lane(lane), at);
        at += g.lanes()[lane as usize].line.length();
        let c = g.lanes()[lane as usize].successors[0];
        start.insert(Elem::Connector(c), at);
        at += g.connectors()[c as usize].line.length();
        lane = g.connectors()[c as usize].to;
    }
    let total = at;
    let geo = traffic(w, 0).geometry();
    let length = geo.front - geo.rear;
    let s = |i: usize| {
        let p = traffic(w, i).place.expect("on the ring");
        start[&p.elem] + p.station
    };
    let mut order: Vec<(f64, usize)> = (0..n).map(|i| (s(i), i)).collect();
    order.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut out = vec![0.0; n];
    for k in 0..n {
        let (a, i) = order[k];
        let (b, _) = order[(k + 1) % n];
        out[i] = (b - a).rem_euclid(total) - length;
    }
    out
}

/// Assert that no two cars touched.
fn no_crashes(w: &WorldInstance, n: usize) {
    for i in 0..n {
        assert!(!w.agent(i).events.contains(Events::CRASH_AGENT), "car {i} crashed");
        assert!(w.agent_contacts()[i].forces.is_empty(), "car {i} touches another");
    }
}

#[test]
fn ring_road_shows_stop_and_go_waves() {
    // Sugiyama et al. (2008): 22 cars on a 230 m ring. With a = 1 m/s² and T = 1 s the IDM
    // (behind the kinematic cars' 0.5 s speed response) is string-unstable at this density:
    // uniform flow breaks up into a jam that travels backwards.
    let radius = 230.0 / TAU;
    let driver = "speed_factor = [1.0, 1.0], headway = [1.0, 1.0], accel = [1.0, 1.0], decel = [2.5, 2.5]";
    let sc = compile(&ring(radius, "[1, 0]", "local", &[("npc", 22, driver)]));
    let mut w = WorldInstance::new(sc, Seed::from_u64(1));
    // Evenly spaced at rest, one car 1 m off.
    for i in 0..22 {
        let a = TAU * i as f64 / 22.0 + if i == 0 { 1.0 / radius } else { 0.0 };
        put(&mut w, i, radius, a, 0.0);
    }
    let (mut lo, mut hi) = (f64::INFINITY, 0.0f64);
    let mut jam: Vec<f64> = Vec::new();
    for step in 0..25 * 300 {
        w.step();
        no_crashes(&w, 22);
        if step >= 25 * 200 {
            for i in 0..22 {
                lo = lo.min(speed(&w, i));
                hi = hi.max(speed(&w, i));
            }
            if step % 25 == 0 {
                // Where the slowest car is.
                let slowest = (0..22).min_by(|&a, &b| speed(&w, a).total_cmp(&speed(&w, b))).unwrap();
                jam.push(angle(&w, slowest));
            }
        }
    }
    eprintln!("speeds over the last 100 s: {lo:.2}..{hi:.2} m/s");
    assert!(lo < 1.0 && hi > 4.0, "stop and go: {lo}..{hi}");
    // The jam moves against the traffic (clockwise).
    let mut travel = 0.0;
    for k in 1..jam.len() {
        let d = (jam[k] - jam[k - 1] + PI).rem_euclid(TAU) - PI;
        travel += d;
    }
    let wave = travel * radius / (jam.len() - 1) as f64;
    eprintln!("jam moves at {wave:.2} m/s");
    assert!(wave < -1.0, "backwards: {wave}");
}

#[test]
fn steady_gaps_match_the_idm_equilibrium() {
    // A string-stable setting: uniform flow at the equilibrium gap for its speed.
    let radius = 60.0;
    let driver = "speed_factor = [1.0, 1.0], headway = [1.5, 1.5], accel = [1.5, 1.5], decel = [2.5, 2.5]";
    let sc = compile(&ring(radius, "[1, 0]", "local", &[("npc", 12, driver)]));
    let mut w = WorldInstance::new(sc, Seed::from_u64(2));
    for i in 0..12 {
        put(&mut w, i, radius, TAU * i as f64 / 12.0 + if i == 3 { 0.05 } else { 0.0 }, 3.0);
    }
    for _ in 0..25 * 200 {
        w.step();
    }
    no_crashes(&w, 12);
    // Speed along the chassis x axis, as the drivers see it.
    let forward = |i: usize| {
        let v = &w.agent(i).vehicle;
        v.lin_vel_world().truncate().dot(glam::DVec2::from_angle(yaw(v.pose().rot)))
    };
    let v = (0..12).map(forward).sum::<f64>() / 12.0;
    for (i, gap) in gaps(&w, 12).into_iter().enumerate() {
        let expected = traffic(&w, i).idm.equilibrium_gap(forward(i));
        eprintln!("car {i}: speed {:.3} gap {gap:.3} (equilibrium {expected:.3})", forward(i));
        assert!((forward(i) - v).abs() < 0.05, "uniform speed");
        assert!((gap - expected).abs() < 0.1, "car {i}: gap {gap} vs {expected}");
    }
}

#[test]
fn mobil_overtakes_only_above_the_threshold() {
    // Two lanes one way; a slow car ahead of a fast one in the right lane.
    let radius = 120.0;
    let run = |threshold: f64| {
        let slow = "speed_factor = [0.4, 0.4], politeness = [0.0, 0.0]".to_string();
        let fast = format!("speed_factor = [1.0, 1.0], politeness = [0.0, 0.0], threshold = {threshold}");
        let sc = compile(&ring(radius, "[2, 0]", "arterial", &[("slow", 1, &slow), ("fast", 1, &fast)]));
        let mut w = WorldInstance::new(sc, Seed::from_u64(3));
        // Right (outer) lane: the lanes are centred, 3.5 m wide.
        put(&mut w, 0, radius + 1.75, 0.5, 5.0);
        put(&mut w, 1, radius + 1.75, 0.0, 5.0);
        let mut passed = false;
        for _ in 0..25 * 90 {
            w.step();
            no_crashes(&w, 2);
            let ahead = (angle(&w, 1) - angle(&w, 0) + PI).rem_euclid(TAU) - PI;
            passed |= ahead > 0.1;
        }
        (traffic(&w, 1).changes, traffic(&w, 0).changes, passed)
    };
    let (changes, slow_changes, passed) = run(0.1);
    eprintln!("threshold 0.1: {changes} changes, passed {passed}");
    assert!(changes >= 2 && passed, "out and back in");
    assert_eq!(slow_changes, 0, "the slow car keeps right");
    // The gain from overtaking (at most the fast car's a = 1–2 m/s²) is below this threshold.
    let (changes, _, passed) = run(3.0);
    assert!(changes == 0 && !passed, "{changes} changes");
}

#[test]
fn multi_lane_traffic_runs_without_collisions() {
    // 30 minutes of 40 cars with drawn parameters on a two-lane each way arterial ring.
    let radius = 150.0;
    let sc = compile(&ring(radius, "[2, 2]", "arterial", &[("npc", 40, "")]));
    let mut w = WorldInstance::new(sc, Seed::from_u64(4));
    let minutes = 30;
    let mut slow = vec![0usize; 40];
    for step in 0..25 * 60 * minutes {
        w.step();
        no_crashes(&w, 40);
        for (i, s) in slow.iter_mut().enumerate() {
            *s = if speed(&w, i) < 0.5 { *s + 1 } else { 0 };
            assert!(*s < 25 * 60, "car {i} stood for a minute at step {step}");
        }
    }
    let changes: u32 = (0..40).map(|i| traffic(&w, i).changes).sum();
    let mean = (0..40).map(|i| speed(&w, i)).sum::<f64>() / 40.0;
    eprintln!("{changes} lane changes, mean speed {mean:.2} m/s");
    assert!(changes > 0);
    for i in 0..40 {
        assert!(traffic(&w, i).place.is_some(), "car {i} lost its lane");
    }
}

#[test]
fn traffic_drives_through_an_urban_map() {
    // Junction rules come in step 3; here the drivers keep to the lane graph through
    // junctions, take turns and keep moving (collisions at junctions are not yet avoided).
    let toml = r#"
        physics_hz = 200
        policy_hz = 25
        map = { type = "urban", seed = 2, count = 1 }
        [[groups]]
        name = "npc"
        count = 20
        vehicle = "sedan_like"
        physics = "kinematic"
        driver = { type = "traffic" }
        spawn = { on_ground = true, on_road = true, min_separation = 20.0 }
        disable_on_terminal = false
    "#;
    let mut w = WorldInstance::new(compile(toml), Seed::from_u64(5));
    let start: Vec<DVec3> = (0..20).map(|i| w.agent(i).vehicle.position()).collect();
    let mut connectors = 0;
    for _ in 0..25 * 120 {
        w.step();
        for i in 0..20 {
            let p = traffic(&w, i).place.expect("on the lanes");
            connectors += usize::from(matches!(p.elem, Elem::Connector(_)));
            // Inside its lane (or connector): close to the line, or between two while changing.
            let line = p.elem.line(w.map().roads().lanes());
            let off = line.point_at(p.station).truncate().distance(w.agent(i).vehicle.position().truncate());
            let allowed = if traffic(&w, i).change.is_some() { 5.0 } else { 2.0 };
            assert!(off < allowed, "car {i} {off} m off its line at step {}: {:?}", w.steps(), traffic(&w, i));
        }
    }
    let moved = (0..20).filter(|&i| w.agent(i).vehicle.position().distance(start[i]) > 100.0).count();
    eprintln!("{moved} of 20 moved over 100 m; {connectors} car-steps on connectors");
    assert!(moved >= 15 && connectors > 0);
    for i in 0..20 {
        assert!(speed(&w, i) > 0.1 || traffic(&w, i).stuck < 10.0, "car {i} stuck");
    }
}
