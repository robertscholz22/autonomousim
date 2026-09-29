//! Scripted `road` drivers: random routes on rural roads, determinism, and scripted groups in
//! batches.

use autonomousim_core::rng::Seed;
use autonomousim_sim::record::{Recorder, RecorderConfig, Recording};
use autonomousim_sim::{BatchSim, CompiledScenario, Events, Scenario, WorldInstance};
use std::sync::Arc;

fn compile(toml: &str) -> Arc<CompiledScenario> {
    Arc::new(Scenario::from_toml(toml).unwrap().compile().unwrap())
}

const TRAFFIC: &str = r#"
    name = "traffic"
    map = { type = "rural", seed = 3, count = 2, cache = false }
    [[groups]]
    name = "cars"
    count = 2
    vehicle = "sedan_like"
    spawn = { on_road = true, min_separation = 40.0 }
    driver = { type = "road" }
"#;

/// Ten minutes of random routes on each map of the pool: the cars stay on the roads (turning
/// round at dead ends and before the map's edges), never meet a terminal event, keep driving
/// (well past their first leg) and do not stop for long.
#[test]
fn cars_drive_random_routes_on_the_roads() {
    let sc = compile(TRAFFIC);
    let steps_per_s = (1.0 / (sc.dt() * f64::from(sc.decimation))).round() as usize;
    let mut seen = [false; 2];
    let mut seed = 0;
    while seen.iter().any(|s| !s) {
        let mut w = WorldInstance::new(sc.clone(), Seed::from_u64(seed));
        seed += 1;
        w.reset(None);
        if seen[w.map_index()] {
            continue;
        }
        seen[w.map_index()] = true;
        let map = w.map().clone();
        let mut travelled = [0.0; 2];
        let mut turns = [0; 2];
        let mut still = [0usize; 2];
        let mut last: Vec<_> = w.agents().iter().map(|a| a.vehicle.position()).collect();
        let mut was_turning = [false; 2];
        for step in 0..600 * steps_per_s {
            w.step();
            for (k, a) in w.agents().iter().enumerate() {
                let p = a.vehicle.position();
                assert!(!a.events.is_terminal() && !a.disabled, "car {k} at {p} after {step} steps: {:?}", a.events);
                assert!(
                    map.roads().on_road(p.truncate()).is_some(),
                    "car {k} left the road at {p} after {step} steps (map {})",
                    w.map_index()
                );
                let d = p.distance(last[k]);
                travelled[k] += d;
                still[k] = if d < 0.01 { still[k] + 1 } else { 0 };
                assert!(still[k] < 60 * steps_per_s, "car {k} stands still at {p} after {step} steps");
                last[k] = p;
                let turning = a.driver.as_ref().unwrap().turn.is_some();
                if turning && !was_turning[k] {
                    turns[k] += 1;
                }
                was_turning[k] = turning;
            }
        }
        eprintln!("map {}: travelled {travelled:?}, turns {turns:?}", w.map_index());
        for (k, d) in travelled.iter().enumerate() {
            assert!(*d > 2000.0, "car {k} drove only {d:.0} m in 10 min (map {})", w.map_index());
            assert!(turns[k] > 0, "car {k} never turned round (map {})", w.map_index());
        }
    }
}

#[test]
fn driving_is_deterministic() {
    let sc = compile(TRAFFIC);
    let run = |seed| {
        let mut w = WorldInstance::new(sc.clone(), Seed::from_u64(seed));
        w.reset(None);
        let mut hashes = Vec::new();
        for _ in 0..40 {
            for _ in 0..50 {
                w.step();
            }
            hashes.push(w.state_hash());
        }
        let route = w.agents()[0].route.clone().unwrap();
        (hashes, route.points().to_vec())
    };
    let (a, b, c) = (run(5), run(5), run(6));
    assert_eq!(a, b);
    assert_ne!(a.0, c.0);
}

/// Recordings carry the routes as they extend, so replays draw the route a car followed at
/// any time.
#[test]
fn recordings_carry_the_routes() {
    let sc = compile(TRAFFIC);
    let path = std::env::temp_dir().join(format!("autonomousim-routes-{}.mcap", std::process::id()));
    let mut rec = Recorder::create(&path, RecorderConfig::default()).unwrap();
    let mut w = WorldInstance::new(sc, Seed::from_u64(4));
    w.reset(None);
    rec.on_reset(&w);
    let route = |w: &WorldInstance, k: usize| w.agents()[k].route.clone().unwrap();
    let mut seen: Vec<Vec<(f64, Arc<_>)>> = (0..2).map(|k| vec![(w.time(), route(&w, k))]).collect();
    for _ in 0..12000 {
        rec.on_actions(&w);
        w.step_with(&mut |w| rec.on_tick(w));
        for (k, s) in seen.iter_mut().enumerate() {
            if !Arc::ptr_eq(&s.last().unwrap().1, &route(&w, k)) {
                s.push((w.time(), route(&w, k)));
            }
        }
    }
    rec.finish().unwrap();
    let recording = Recording::read(&path).unwrap();
    std::fs::remove_file(&path).ok();
    let ep = &recording.episodes[0];
    for (k, s) in seen.iter().enumerate() {
        assert!(s.len() > 2, "car {k}'s route never extended");
        for (t, r) in s {
            assert_eq!(ep.route_at(k, *t).unwrap().points(), r.points(), "car {k} at {t}");
        }
    }
}

/// A drone group learns among scripted cars: batches take no actions for the cars (an
/// empty array or a full one, ignored), and env 0 does not depend on the thread count.
#[test]
fn scripted_groups_in_batches() {
    let toml = format!(
        "{TRAFFIC}\n{}",
        r#"
        [[groups]]
        name = "drones"
        count = 1
        vehicle = "cf2x"
        spawn = { agl = [2.0, 3.0] }
        "#
    );
    let sc = compile(&toml);
    assert!(sc.groups[0].scripted() && !sc.groups[1].scripted());
    let run = |threads| {
        let mut b = BatchSim::from_compiled(sc.clone(), 3, 9, threads).unwrap();
        let drones = vec![0.0f32; 3 * sc.groups[1].act_dim()];
        let ignored = vec![1.0f32; 3 * 2 * sc.groups[0].act_dim()];
        for k in 0..100 {
            let cars: &[f32] = if k % 2 == 0 { &[] } else { &ignored };
            b.step(&[cars, &drones]);
        }
        b.world(0).state_hash()
    };
    assert_eq!(run(1), run(3));

    // The cars drive although no actions reach them.
    let mut b = BatchSim::from_compiled(sc.clone(), 1, 9, 1).unwrap();
    let start = b.world(0).agents()[0].vehicle.position();
    for _ in 0..250 {
        b.step(&[&[], &[0.0; 4]]);
    }
    let a = &b.world(0).agents()[0];
    assert!(a.vehicle.position().distance(start) > 10.0 && !a.events.intersects(Events::DISABLED));
}

#[test]
fn drivers_need_cars_on_roads() {
    let err = |toml: String| Scenario::from_toml(&toml).unwrap().compile().unwrap_err().to_string();
    let e = err(TRAFFIC.replace("on_road = true, ", ""));
    assert!(e.contains("driver"), "{e}");
    let e = err(TRAFFIC.replace(r#"driver = { type = "road" }"#, r#"driver = { type = "road", speed = [5.0, 1.0] }"#));
    assert!(e.contains("invalid road driver"), "{e}");
    let e = err(TRAFFIC.replace(
        r#"type = "road" }"#,
        r#"type = "road" }
    goals = { kind = "route", distance = [100.0, 200.0] }"#,
    ));
    assert!(e.contains("driver"), "{e}");
}
