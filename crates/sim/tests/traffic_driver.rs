//! The `traffic` driver (M8b steps 2 and 3): IDM car following on a ring road (stop-and-go
//! waves as in Sugiyama et al. 2008, the equilibrium gaps), MOBIL lane changes, a long run of
//! multi-lane traffic without collisions, and traffic with parked cars on urban maps
//! (junction rules: no collisions, no red lights crossed, no lasting gridlock).

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

/// What `urban_traffic` saw.
#[derive(Debug, Default)]
struct UrbanStats {
    crashes: usize,
    red_lights: usize,
    respawns: u32,
    /// Longest time a moving NPC stood still (s).
    longest_stand: f64,
    /// Mean speed of the moving NPCs in each 5-minute window (m/s).
    window_speeds: Vec<f64>,
    /// Entries into junctions of each kind (`JunctionKind` order), and the kinds on the map.
    entries: [usize; 7],
    kinds: [bool; 7],
    /// Largest distance a parked car moved (m).
    parked_moved: f64,
}

/// `n` traffic NPCs and `parked` parked cars on urban training map `seed`.
fn urban_toml(seed: u64, n: usize, parked: usize) -> String {
    format!(
        r#"
        physics_hz = 200
        policy_hz = 25
        map = {{ type = "urban", seed = {seed}, count = 1 }}
        [[groups]]
        name = "npc"
        count = {n}
        vehicle = "sedan_like"
        physics = "kinematic"
        driver = {{ type = "traffic" }}
        spawn = {{ on_ground = true, on_road = true, min_separation = 20.0 }}
        disable_on_terminal = false
        [[groups]]
        name = "parked"
        count = {parked}
        vehicle = "sedan_like"
        physics = "kinematic"
        driver = {{ type = "parked" }}
        spawn = {{ on_ground = true, in_bays = true }}
        disable_on_terminal = false
        "#
    )
}

/// `n` kinematic traffic NPCs and `parked` parked cars on urban training map `seed` for
/// `minutes`; prints the first few crashes.
fn urban_traffic(seed: u64, n: usize, parked: usize, minutes: f64) -> UrbanStats {
    let mut w = WorldInstance::new(compile(&urban_toml(seed, n, parked)), Seed::from_u64(seed));
    let g = w.map().roads().lanes().clone();
    let kind = |c: u32| g.junctions()[g.connectors()[c as usize].node as usize].kind as usize;
    let mut st = UrbanStats::default();
    for c in 0..g.connectors().len() as u32 {
        st.kinds[kind(c)] = true;
    }
    let parked_at: Vec<DVec3> = (n..n + parked).map(|i| w.agent(i).vehicle.position()).collect();
    let mut stand = vec![0.0f64; n];
    let mut last = vec![None; n];
    let steps = (25.0 * 60.0 * minutes) as usize;
    let window = 25 * 300;
    let mut speed_sum = 0.0;
    for step in 1..=steps {
        w.step();
        for i in 0..n + parked {
            let e = w.agent(i).events;
            if e.contains(Events::CRASH_AGENT) {
                st.crashes += 1;
                if st.crashes <= 4 {
                    let d = w.agent(i).driver.as_ref().and_then(Driver::as_traffic);
                    eprintln!(
                        "crash at step {}: car {i} at {:.1?} place {:?}",
                        w.steps(),
                        w.agent(i).vehicle.position().truncate(),
                        d.and_then(|d| d.place)
                    );
                }
            }
            if i >= n {
                continue;
            }
            if e.contains(Events::RED_LIGHT) {
                st.red_lights += 1;
                if st.red_lights <= 4 {
                    let d = traffic(&w, i);
                    let lights: Vec<_> = d.plan.iter().map(|&c| (c, w.signals().light_left(&g, c, w.time()))).collect();
                    eprintln!(
                        "red light at step {}: car {i} v {:.2} place {:?} plan {lights:?} granted {:?} waiting {:.1}",
                        w.steps(),
                        speed(&w, i),
                        d.place,
                        d.granted,
                        d.waiting
                    );
                }
            }
            let v = speed(&w, i);
            speed_sum += v;
            stand[i] = if v < 0.5 { stand[i] + 0.04 } else { 0.0 };
            st.longest_stand = st.longest_stand.max(stand[i]);
            let elem = traffic(&w, i).place.map(|p| p.elem);
            if let (Some(Elem::Lane(_)), Some(Elem::Connector(c))) = (last[i], elem) {
                st.entries[kind(c)] += 1;
            }
            last[i] = elem;
        }
        if step % window == 0 {
            st.window_speeds.push(speed_sum / (window * n) as f64);
            speed_sum = 0.0;
        }
    }
    st.respawns = (0..n).map(|i| traffic(&w, i).respawns).sum();
    st.parked_moved = (0..parked).map(|k| w.agent(n + k).vehicle.position().distance(parked_at[k])).fold(0.0, f64::max);
    st
}

/// The step 3 acceptance checks on `st` of `minutes` of 50 NPCs.
fn check_urban(seed: u64, st: &UrbanStats, minutes: f64) {
    eprintln!("seed {seed}: {st:?}");
    assert_eq!(st.crashes, 0, "seed {seed}: NPC collisions");
    assert_eq!(st.red_lights, 0, "seed {seed}: red lights crossed");
    // Gridlocks are rare and broken by respawning (after 120 s standing).
    assert!(f64::from(st.respawns) <= 2.0 + 0.3 * minutes, "seed {seed}: {} respawns", st.respawns);
    assert!(st.parked_moved < 0.05, "seed {seed}: a parked car moved {} m", st.parked_moved);
    // Stable flow: no window much slower than the first.
    let first = st.window_speeds[0];
    assert!(st.window_speeds.iter().all(|&v| v > 0.6 * first), "seed {seed}: speeds {:?}", st.window_speeds);
    // Every kind of junction on the map keeps flowing (entries at signals, stop and yield
    // signs, roundabouts, uncontrolled junctions, through and dead ends).
    for k in 0..7 {
        assert!(!st.kinds[k] || st.entries[k] > 0, "seed {seed}: no entries into junctions of kind {k}");
    }
}

#[test]
fn urban_traffic_keeps_the_rules() {
    for seed in [1, 2] {
        let st = urban_traffic(seed, 50, 20, 5.0);
        check_urban(seed, &st, 5.0);
    }
}

/// The step 3 acceptance run: 30 minutes on each of the first eight training maps
/// (about 10 min of CPU in release).
#[test]
#[ignore]
fn urban_traffic_for_half_an_hour() {
    for seed in 1..=8 {
        let st = urban_traffic(seed, 50, 20, 30.0);
        check_urban(seed, &st, 30.0);
    }
}

#[test]
fn parked_cars_need_bays() {
    let parked = |spawn: &str, map: &str| {
        let toml = format!(
            r#"
            map = {map}
            [[groups]]
            name = "parked"
            count = 3
            vehicle = "sedan_like"
            physics = "kinematic"
            driver = {{ type = "parked" }}
            spawn = {spawn}
            "#
        );
        Scenario::from_toml(&toml).unwrap().compile().map(|_| ())
    };
    let urban = r#"{ type = "urban", seed = 1, count = 1 }"#;
    let ring = r#"{ type = "testworld", kind = "ring", radius = 60.0, lanes = [1, 1], class = "local" }"#;
    assert!(parked("{ on_ground = true, in_bays = true }", urban).is_ok());
    // On the road, or on a map without bays (wheeled vehicles are always on the ground).
    assert!(parked("{ on_ground = true, on_road = true }", urban).is_err());
    assert!(parked("{ on_ground = true, in_bays = true }", ring).is_err());
}

/// An NPC closing in on a learning car standing in its lane: whether it touched it, and its
/// speed at the end.
fn meet_learner(attention: f64) -> (bool, f64) {
    let radius = 80.0;
    let mut toml = ring(radius, "[1, 0]", "local", &[("npc", 1, &format!("attention = {attention}"))]);
    toml += r#"
        [[groups]]
        name = "ego"
        vehicle = "sedan_like"
        physics = "kinematic"
        action_mode = "vk"
        spawn = { on_ground = true, on_road = true }
        disable_on_terminal = false
    "#;
    let mut w = WorldInstance::new(compile(&toml), Seed::from_u64(4));
    put(&mut w, 0, radius, 0.0, 8.0);
    put(&mut w, 1, radius, 1.0, 0.0);
    let mut touched = false;
    for _ in 0..25 * 30 {
        w.set_action(1, &[0.0, 0.0]);
        w.step();
        touched |= w.agent(0).events.contains(Events::CRASH_AGENT);
    }
    assert!(speed(&w, 1) < 0.5 || touched, "the learning car stands: {}", speed(&w, 1));
    (touched, speed(&w, 0))
}

#[test]
fn npcs_yield_to_learners_they_notice() {
    let (touched, v) = meet_learner(1.0);
    assert!(!touched && v < 0.5, "attention 1: stops behind ({touched}, {v})");
    let (touched, _) = meet_learner(0.0);
    assert!(touched, "attention 0: runs into it");
}

/// The `traffic` term sees an NPC in the lane to the left, and `lane_route` the way straight on.
#[test]
fn traffic_and_lane_route_terms() {
    let radius = 80.0;
    let mut toml = ring(radius, "[2, 0]", "arterial", &[("npc", 1, "speed_factor = [0.5, 0.5]")]);
    toml += r#"
        [[groups]]
        name = "ego"
        vehicle = "sedan_like"
        physics = "kinematic"
        action_mode = "vk"
        spawn = { on_ground = true, on_road = true }
        obs = [{ term = "traffic", count = 2 }, { term = "lane_route" }]
        disable_on_terminal = false
    "#;
    let sc = compile(&toml);
    assert_eq!(sc.groups[1].obs_dim(), 2 * 13 + 6);
    let mut w = WorldInstance::new(sc, Seed::from_u64(5));
    // Ego in the right (outer) lane, the NPC 8 m ahead in the left one (clear of the ring's
    // nodes at angles 0 and π).
    put(&mut w, 1, radius + 1.75, 0.5, 0.0);
    put(&mut w, 0, radius - 1.75, 0.5 + 8.0 / radius, 0.0);
    w.set_action(1, &[0.0, 0.0]);
    w.step();
    let mut o = vec![0.0f32; 32];
    w.observe(1, &mut o);
    let geo = traffic(&w, 0).geometry();
    let (x, y) = (f64::from(o[0]), f64::from(o[1]));
    assert!((6.0..10.0).contains(&x) && (2.5..4.5).contains(&y), "{o:?}");
    assert!(o[4].abs() < 0.2 && o[5] > 0.98, "heading {o:?}");
    assert!(
        (f64::from(o[6]) - (geo.front - geo.rear)).abs() < 1e-5
            && (f64::from(o[7]) - 2.0 * geo.half_width).abs() < 1e-5
    );
    assert_eq!(&o[8..13], &[0.0, 1.0, 0.0, 0.0, 1.0], "left lane, present");
    assert!(o[13..26].iter().all(|&v| v == 0.0), "one vehicle only");
    // No route: straight on from its own lane.
    assert_eq!(o[26], 0.0);
    assert!(o[27] > 0.0, "distance to the lane's end {o:?}");
    assert_eq!(&o[28..32], &[0.0, 1.0, 0.0, 0.0]);
}

/// NPC traffic with a learning car on an urban map, recorded: the NPCs go into `/npcs` (no
/// channels of their own), read back as states; the same seed and actions give the same run
/// (also with drivers that notice the learner only at random).
#[test]
fn npcs_are_recorded_packed_and_runs_repeat() {
    let toml = format!(
        "{}{}",
        urban_toml(2, 20, 2).replace("type = \"traffic\"", "type = \"traffic\", attention = 0.5"),
        r#"
        [[groups]]
        name = "ego"
        vehicle = "sedan_like"
        physics = "kinematic"
        action_mode = "vk"
        spawn = { on_ground = true, on_road = true }
        disable_on_terminal = false
    "#
    );
    let sc = compile(&toml);
    let n = sc.num_agents();
    let run = |record: bool| {
        let mut w = WorldInstance::new(sc.clone(), Seed::from_u64(9));
        let sink = Arc::new(std::sync::Mutex::new(autonomousim_sim::record::MemorySink::default()));
        let mut rec = autonomousim_sim::record::Recorder::new(
            Box::new(sink.clone()),
            autonomousim_sim::record::RecorderConfig::default(),
        );
        if record {
            rec.on_reset(&w);
        }
        for k in 0..25 * 20 {
            w.set_action(n - 1, &[0.4, if (k / 50) % 2 == 0 { 0.1 } else { -0.1 }]);
            if record {
                rec.on_actions(&w);
                w.step_with(&mut |w| rec.on_tick(w));
            } else {
                w.step();
            }
        }
        rec.finish().unwrap();
        let sink = Arc::try_unwrap(sink).ok().unwrap().into_inner().unwrap();
        (w.state_hash(), sink, w)
    };
    let (a, sink, w) = run(true);
    let (b, _, _) = run(false);
    assert_eq!(a, b, "the same run");
    // The NPCs are packed: no channels of their own, the learner has its own.
    let npcs = sink.topic("/npcs");
    assert!(!npcs.is_empty());
    assert!(sink.topic("/agent/0/state").is_empty() && sink.topic("/agent/0/action").is_empty());
    assert_eq!(sink.topic(&format!("/agent/{}/state", n - 1)).len(), npcs.len());
    let last: autonomousim_sim::record::RecordedNpcs = serde_json::from_value(npcs.last().unwrap().1.clone()).unwrap();
    assert_eq!(last.agents.len(), n - 1);
    assert!(last.kinematic.iter().all(|&k| k));
    for (id, s) in last.states() {
        let v = &w.agent(id).vehicle;
        assert_eq!((s.position, s.orientation), (v.position(), v.orientation()), "agent {id}");
        assert_eq!(s.wheels.len(), 4, "agent {id}");
    }
    let bytes: usize = npcs.iter().map(|m| serde_json::to_vec(&m.1).unwrap().len()).sum::<usize>() / npcs.len();
    eprintln!("/npcs: {bytes} bytes per message for {} NPCs", n - 1);
}
