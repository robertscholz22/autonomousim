//! Urban sites in the simulation: drones land on rooftop pads, cars get bay goals in parking
//! lots and parking lanes.

use autonomousim_control::ground::GroundSetpoint;
use autonomousim_core::math::Pose;
use autonomousim_core::math::quat::wrap_angle;
use autonomousim_core::rng::Seed;
use autonomousim_core::terrain::Terrain;
use autonomousim_sim::drive::ground_pose;
use autonomousim_sim::record::{MemorySink, Recorder, RecorderConfig, Recording};
use autonomousim_sim::{CompiledScenario, Events, STATE_DIM, STATE_FIELDS, Scenario, WorldInstance};
use autonomousim_world::BayKind;
use glam::{DQuat, DVec2, DVec3};
use std::sync::{Arc, Mutex};

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

/// Rooftop goals: the goal is a pad (at its surface, its heading); the drone starts on its
/// gear on a sidewalk clear of obstacles or on another pad, mostly within the distance range,
/// and rests there with motors idle without events. Both kinds of start occur; the draws are
/// deterministic.
#[test]
fn rooftop_goals_start_on_sidewalks_and_pads() {
    let sc = compile(
        r#"
        map = { type = "urban", seed = 2, count = 1 }
        [[groups]]
        vehicle = "iris_like"
        spawn = { on_ground = true }
        goals = { kind = "rooftop", distance = [200.0, 600.0], agl = [0.0, 0.0], rooftop = { roof_start = 0.4 } }
        "#,
    );
    let mut w = WorldInstance::new(sc.clone(), Seed::from_u64(3));
    let map = w.map().clone();
    let pads = &map.sites().pads;
    let bottom = sc.groups[0].bottom;
    let (mut roofs, mut streets, mut in_range) = (0, 0, 0);
    let mut first = Vec::new();
    for episode in 0..24u64 {
        w.reset(Some(episode));
        let goal = w.agent(0).goal();
        let pad = pads.iter().find(|p| p.centre.distance(goal.position) < 1e-9).expect("goal on a pad");
        assert!(wrap_angle(goal.yaw - pad.yaw).abs() < 1e-9);
        let p = w.agent(0).vehicle.position();
        first.push(p);
        let start = pads.iter().find(|q| q.centre.truncate().distance(p.truncate()) < 1e-9);
        match start {
            Some(q) => {
                roofs += 1;
                assert!((p.z - bottom - q.centre.z).abs() < 0.01, "{episode}: {p}");
            }
            None => {
                streets += 1;
                assert_eq!(map.roads().area(p.truncate()), autonomousim_world::Area::Sidewalk, "{episode}: {p}");
                let ground = map.terrain().height(p.x, p.y);
                assert!((p.z - bottom - ground).abs() < 0.01, "{episode}: {p}");
            }
        }
        let d = p.truncate().distance(goal.position.truncate());
        in_range += usize::from((200.0..=600.0).contains(&d));
        for _ in 0..(1.0 / w.scenario().policy_dt()) as usize {
            w.set_action(0, &[0.0, 0.0, 0.0, -1.0]);
            w.step();
            let q = w.agent(0).vehicle.position();
            assert!(
                !w.agent(0).events.is_terminal(),
                "{episode}: {:?} at {q}, from {p} ({start:?})",
                w.agent(0).events
            );
        }
        assert!(w.agent(0).events.contains(Events::LANDED), "{episode}");
    }
    assert!(roofs >= 4 && streets >= 8, "{roofs} roof and {streets} street starts");
    assert!(in_range >= 20, "{in_range} of 24 in range");
    let mut again = WorldInstance::new(sc, Seed::from_u64(3));
    for (episode, p) in first.iter().enumerate() {
        again.reset(Some(episode as u64));
        assert_eq!(again.agent(0).vehicle.position(), *p);
    }
}

/// Rooftop goals need an aircraft and a map with two pads.
#[test]
fn rooftop_goals_are_checked() {
    let err = |toml: &str| Scenario::from_toml(toml).unwrap().compile().err().map(|e| e.to_string());
    let flat = err(r#"
        map = { type = "testworld", kind = "flat", size = 200.0 }
        [[groups]]
        vehicle = "iris_like"
        goals = { kind = "rooftop" }
        "#);
    assert!(flat.as_deref().is_some_and(|e| e.contains("two rooftop pads")), "{flat:?}");
    let car = err(r#"
        map = { type = "urban", seed = 1, count = 1 }
        [[groups]]
        vehicle = "sedan_like"
        action_mode = "vk"
        goals = { kind = "rooftop" }
        "#);
    assert!(car.as_deref().is_some_and(|e| e.contains("aerial vehicle")), "{car:?}");
    let share = err(r#"
        map = { type = "urban", seed = 1, count = 1 }
        [[groups]]
        vehicle = "iris_like"
        goals = { kind = "rooftop", rooftop = { roof_start = 1.5 } }
        "#);
    assert!(share.as_deref().is_some_and(|e| e.contains("roof_start")), "{share:?}");
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

/// Parking between parked cars: nobody parks in the goal bay, and `near_bay_goals` parks the
/// cars in the bays nearest to it.
#[test]
fn parked_cars_leave_the_goal_bay_and_gather_around_it() {
    for kind in ["lot", "street"] {
        let sc = compile(&format!(
            r#"
            map = {{ type = "urban", seed = 2, count = 1 }}
            [[groups]]
            name = "ego"
            vehicle = "sedan_like"
            action_mode = "vk"
            goals = {{ kind = "bay", distance = [6.0, 12.0], bay = {{ kinds = ["{kind}"] }} }}
            [[groups]]
            name = "parked"
            count = 6
            vehicle = "sedan_like"
            physics = "kinematic"
            driver = {{ type = "parked" }}
            spawn = {{ on_ground = true, in_bays = true, near_bay_goals = true }}
            "#
        ));
        let mut w = WorldInstance::new(sc, Seed::from_u64(0));
        for episode in 0..10u64 {
            w.reset(Some(episode));
            let goal = w.agent(0).goal().position.truncate();
            let bays = &w.map().sites().bays;
            let near = |p: DVec2| {
                (0..bays.len())
                    .min_by(|&a, &b| {
                        bays[a].centre.truncate().distance(p).total_cmp(&bays[b].centre.truncate().distance(p))
                    })
                    .unwrap()
            };
            let at = near(goal);
            let gap = |k: usize| bays[k].centre.truncate().distance(bays[at].centre.truncate());
            let mut gaps: Vec<f64> = (0..bays.len()).filter(|&k| k != at).map(gap).collect();
            gaps.sort_by(f64::total_cmp);
            for i in 1..7 {
                let p = w.agent(i).vehicle.position().truncate();
                let k = near(p);
                assert_ne!(k, at, "{kind} {episode}: car {i} parks in the goal bay");
                assert!(gap(k) <= gaps[11] + 1e-9, "{kind} {episode}: car {i} parks far from the goal");
            }
        }
    }
    let far = r#"
        map = { type = "urban", seed = 2, count = 1 }
        [[groups]]
        vehicle = "sedan_like"
        spawn = { on_ground = true, near_bay_goals = true }
        "#;
    assert!(Scenario::from_toml(far).unwrap().compile().is_err());
}

/// Traffic signals: per-episode offsets drawn deterministically within each cycle; the
/// `signal` and `lanes` terms read the lane a car stands in before a stop line; recordings
/// carry the offsets and the light changes.
#[test]
fn signals_offsets_terms_and_recording() {
    let sc = compile(
        r#"
        map = { type = "urban", seed = 2, count = 1 }
        [[groups]]
        vehicle = "sedan_like"
        action_mode = "vk"
        obs = [{ term = "signal" }, { term = "lanes" }]
        "#,
    );
    let mut a = WorldInstance::new(sc.clone(), Seed::from_u64(7));
    let b = WorldInstance::new(sc.clone(), Seed::from_u64(7));
    let lanes = a.map().roads().lanes();
    let n = lanes.controllers().len();
    assert!(n > 0);
    assert_eq!(a.signals(), b.signals());
    assert_eq!(a.signals().offsets().len(), n);
    for (o, c) in a.signals().offsets().iter().zip(lanes.controllers()) {
        assert!((0.0..c.cycle()).contains(o));
    }
    let first = a.signals().clone();
    a.reset(None);
    assert_ne!(a.signals(), &first);
    a.reset(Some(7));
    assert_eq!(a.signals(), &first);

    // A lane into a signalized junction, long enough to stand 15 m before its stop line.
    let map = a.map().clone();
    let lanes = map.roads().lanes();
    let (l, lane) = lanes
        .lanes()
        .iter()
        .enumerate()
        .find(|(_, l)| lanes.junction_controller(l.to_node).is_some() && l.line.length() > 30.0)
        .unwrap();
    let s = lane.line.length() - 15.0;
    let p = lane.line.point_at(s);
    let pose = ground_pose(
        &map,
        sc.groups[0].def.as_wheeled().unwrap(),
        &sc.groups[0].rest,
        p.truncate(),
        lane.line.heading_at(s),
    );
    let sink = Arc::new(Mutex::new(MemorySink::default()));
    let mut rec = Recorder::new(Box::new(sink.clone()), RecorderConfig::default());
    rec.on_reset(&a);
    a.place_agent(0, pose, DVec3::ZERO, DVec3::ZERO);
    a.set_command(0, GroundSetpoint::SpeedCurvature { speed: 0.0, curvature: 0.0 });
    for _ in 0..(90.0 / a.scenario().policy_dt()) as usize {
        a.step_with(&mut |w| rec.on_tick(w));
        let e = a.agent(0).events;
        assert!(!e.intersects(Events::RED_LIGHT | Events::WRONG_WAY | Events::OFF_ROAD), "{e:?}");
    }
    assert_eq!(a.agent(0).track.lane, Some(l as u32));
    let mut o = vec![0.0f32; 16];
    a.observe(0, &mut o);
    assert_eq!(o[0] + o[1] + o[2], 1.0, "{o:?}");
    assert!((o[3] - 15.0).abs() < 1.5, "{o:?}");
    assert!((o[4] - 5.0).abs() < 0.5 && o[5].abs() < 0.5, "{o:?}");
    assert!((o[8] - 20.0).abs() < 1.0 && o[9].abs() < 1.0, "{o:?}");
    assert!((f64::from(o[12]) - lane.speed).abs() < 1e-5);
    assert_eq!(o[13], f32::from(u8::from(lane.left.is_some())));
    assert!(o[15].abs() < 0.5);
    rec.finish().unwrap();

    let sink = sink.lock().unwrap();
    let ep = &sink.topic("/episode")[0].1;
    let offsets: Vec<f64> = serde_json::from_value(ep["signals"].clone()).unwrap();
    assert_eq!(offsets, first.offsets());
    let changes = sink.topic("/signals");
    // All controllers at the reset, then over 90 s every one changes at least twice.
    assert!(changes.len() >= 3 * n, "{} messages", changes.len());
    let k = changes[0].1["controller"].as_u64().unwrap() as usize;
    let t = changes.iter().filter(|m| m.1["controller"] == k).nth(1).unwrap().1["time"].as_f64().unwrap();
    let (phase, light) = a.signals().state(lanes, k, t);
    let m = changes.iter().filter(|m| m.1["controller"] == k).nth(1).unwrap();
    assert_eq!(m.1["phase"].as_u64().unwrap() as usize, phase);
    assert_eq!(m.1["light"], format!("{light:?}").to_lowercase());
    drop(sink);

    // Read back from a file.
    let path = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("signals.mcap");
    let mut rec = Recorder::create(&path, RecorderConfig::default()).unwrap();
    a.reset(Some(7));
    rec.on_reset(&a);
    for _ in 0..5 {
        a.step_with(&mut |w| rec.on_tick(w));
    }
    rec.finish().unwrap();
    let r = Recording::read(&path).unwrap();
    assert_eq!(r.episodes[0].signal_offsets, first.offsets());
}

/// Junction goals: the agents spawn on different arms of one junction (or roundabout) in a
/// lane facing along it, each with a route through the junction to a goal on another road; a
/// traffic driver flying the group follows the routes and, giving way to the others, brings
/// (nearly) every car to its goal without a crash. Sites leave out signals.
#[test]
fn junction_goals_cross_one_junction() {
    let sc = compile(
        r#"
        policy_hz = 20
        map = { type = "urban", seed = 2, count = 2 }
        [[groups]]
        name = "cars"
        count = 3
        vehicle = "sedan_like"
        driver = { type = "traffic", respawn = 0.0 }
        goals = { kind = "junction", distance = [15.0, 35.0], radius = 3.0 }
        "#,
    );
    let maps = &sc.maps;
    let sites: Vec<_> =
        maps.iter().map(|m| autonomousim_sim::junction::sites(m, &sc.groups[0].spec.goals.junction.kinds)).collect();
    assert!(sites.iter().all(|s| !s.is_empty()));
    for s in sites.iter().flatten() {
        assert!(s.arms.len() >= 2 && !s.exits.is_empty());
    }
    let (mut reached, mut total) = (0, 0);
    for seed in 0..8 {
        let mut w = WorldInstance::new(sc.clone(), Seed::from_u64(seed));
        for k in 0..3 {
            let a = w.agent(k);
            let route = a.route.as_ref().expect("a route through the junction");
            let p = a.vehicle.position().truncate();
            let goal = a.goals[0].position.truncate();
            // Starts on its route, facing along it; the goal ends it.
            assert!(route.project(p).distance < 0.5, "seed {seed} agent {k}");
            let heading = route.heading_at(0.0);
            let yaw = autonomousim_core::math::quat::yaw(a.vehicle.pose().rot);
            assert!(wrap_angle(yaw - heading).abs() < 0.1);
            assert!(route.point_at(route.length()).truncate().distance(goal) < 1e-6);
            assert!((15.0..120.0).contains(&p.distance(goal)), "seed {seed} agent {k}: {}", p.distance(goal));
        }
        // Apart, on different arms.
        for i in 0..3 {
            for j in i + 1..3 {
                let (pi, pj) = (w.agent(i).vehicle.position(), w.agent(j).vehicle.position());
                assert!(pi.distance(pj) > 5.0, "seed {seed}: agents {i} and {j} start {:.1} m apart", pi.distance(pj));
            }
        }
        let mut done = [false; 3];
        for _ in 0..800 {
            w.step();
            for (k, d) in done.iter_mut().enumerate() {
                let e = w.agent(k).events;
                assert!(*d || !e.is_terminal(), "seed {seed} agent {k}: {e:?}");
                if e.contains(Events::FINISHED) && !*d {
                    *d = true;
                    reached += 1;
                }
            }
            if done.iter().all(|&d| d) {
                break;
            }
        }
        total += 3;
    }
    assert!(reached * 10 >= total * 9, "{reached} of {total} reached their goals");
}
