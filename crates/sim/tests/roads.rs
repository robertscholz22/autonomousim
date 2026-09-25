//! Driving on rural roads: spawns in the lane, routes along the roads, and the `road`, `route`
//! and `on_road` observation terms.

use autonomousim_core::math::quat::{wrap_angle, yaw};
use autonomousim_core::rng::Seed;
use autonomousim_core::terrain::Terrain;
use autonomousim_sim::lane::{Follow, ROAD_REACH, lane_offset, road_state};
use autonomousim_sim::{CompiledScenario, STATE_DIM, Scenario, WorldInstance};
use autonomousim_world::NodeKind;
use glam::DVec2;
use std::f64::consts::FRAC_PI_2;
use std::sync::Arc;

fn compile(toml: &str) -> Arc<CompiledScenario> {
    Arc::new(Scenario::from_toml(toml).unwrap().compile().unwrap())
}

const RURAL: &str = r#"
    name = "rural"
    map = { type = "rural", seed = 3, count = 2, cache = false }
    [[groups]]
    name = "cars"
    count = 4
    vehicle = "sedan_like"
    spawn = { on_road = true, min_separation = 10.0 }
    goals = { kind = "route", distance = [150.0, 400.0], radius = 5.0, route = { step = 20.0 } }
    obs = [{ term = "road" }, { term = "route" }, { term = "on_road" }]
"#;

#[test]
fn road_spawns_start_in_the_lane_along_their_route() {
    let sc = compile(RURAL);
    let ride = sc.groups[0].rest.pos.z;
    let mut w = WorldInstance::new(sc.clone(), Seed::from_u64(5));
    for episode in 0..4 {
        if episode > 0 {
            w.reset(None);
        }
        let map = w.map().clone();
        let net = map.roads();
        for a in w.agents() {
            let p = a.vehicle.position();
            let rp = net.on_road(p.truncate()).unwrap_or_else(|| panic!("spawn {p} off the road"));
            let road = &net.roads()[rp.road as usize];
            // In the right-hand lane, facing along the road.
            let h = wrap_angle(rp.projection.heading - yaw(a.vehicle.orientation()));
            let along = if h.cos() > 0.0 { 1.0 } else { -1.0 };
            assert!(h.cos().abs() > 0.9, "{}: heading off the road by {h}", a.id);
            let lane = -along * lane_offset(road);
            assert!((rp.projection.offset - lane).abs() < 0.6, "{}: offset {} not {lane}", a.id, rp.projection.offset);
            // The route starts here and its goals lie on it, every 20 m, the last at its end.
            let route = a.route.as_ref().expect("route goals keep the route");
            let len = route.length();
            // Few yards on a small map: the closest length to the range when none is in it.
            assert!((100.0..=600.0).contains(&len), "{}: route of {len} m", a.id);
            assert!(route.point_at(0.0).truncate().distance(p.truncate()) < 0.6);
            assert_eq!(a.goals.len(), (len / 20.0).ceil() as usize);
            for (k, g) in a.goals.iter().enumerate() {
                let s = (20.0 * (k + 1) as f64).min(len);
                let q = route.point_at(s).truncate();
                assert!(g.position.truncate().distance(q) < 1e-9);
                assert!((g.position.z - map.terrain().height(q.x, q.y) - ride).abs() < 1e-9);
            }
            // Every point of the route is on a road; it ends at a yard.
            for i in 0..=(len as usize) {
                let q = route.point_at(i as f64).truncate();
                assert!(net.nearest(q, 2.5).is_some(), "{}: route point {q} off the roads", a.id);
            }
            let end = route.point_at(len).truncate();
            let yard = net.nodes().iter().filter(|n| n.kind == NodeKind::Yard);
            assert!(yard.map(|n| n.position.truncate().distance(end)).fold(f64::INFINITY, f64::min) < 30.0);
        }
    }
}

#[test]
fn road_terms_follow_the_route() {
    let sc = compile(RURAL);
    let w = WorldInstance::new(sc.clone(), Seed::from_u64(9));
    let dim = sc.groups[0].obs.dim();
    assert_eq!(dim, 6 + 8 + 1);
    let mut obs = vec![0.0f32; 4 * dim];
    w.observe(0, &mut obs);
    for (a, o) in w.agents().iter().zip(obs.chunks_exact(dim)) {
        let p = a.vehicle.position();
        let y = yaw(a.vehicle.orientation());
        // At the start of the route: in the lane, heading along it.
        assert!(o[0].abs() < 0.6, "{}: offset {}", a.id, o[0]);
        assert!(o[1].abs() < 0.3, "{}: heading error {}", a.id, o[1]);
        let f = Follow::new(a.route.as_deref(), w.map(), p.truncate(), y).unwrap();
        for (k, ahead) in [5.0, 10.0, 20.0, 40.0].into_iter().enumerate() {
            assert!((f64::from(o[2 + k]) - f.curvature(ahead)).abs() < 1e-5);
            // Route points lie ahead, at about their distance.
            let (x, yy) = (f64::from(o[6 + 2 * k]), f64::from(o[7 + 2 * k]));
            assert!(x > 0.0, "{}: point {ahead} m ahead behind ({x}, {yy})", a.id);
            assert!(((x * x + yy * yy).sqrt() - ahead).abs() < 0.2 * ahead + 1.0, "{}: ({x}, {yy}) for {ahead}", a.id);
        }
        assert_eq!(o[14], 1.0);
    }
    // The state column: the same offset and heading error, on the road.
    let mut state = vec![0.0; 4 * STATE_DIM];
    w.write_state(0, &mut state);
    for (row, o) in state.as_chunks::<STATE_DIM>().0.iter().zip(obs.chunks_exact(dim)) {
        assert!((row[21] - f64::from(o[0])).abs() < 1e-5 && (row[22] - f64::from(o[1])).abs() < 1e-5);
        assert_eq!(row[23], 0.0);
    }
}

#[test]
fn road_terms_without_a_route_follow_the_nearest_road() {
    let sc = compile(&RURAL.replace(r#"kind = "route""#, r#"kind = "spawn""#));
    let mut w = WorldInstance::new(sc.clone(), Seed::from_u64(2));
    let dim = sc.groups[0].obs.dim();
    let mut obs = vec![0.0f32; 4 * dim];
    for _ in 0..3 {
        w.observe(0, &mut obs);
        for (a, o) in w.agents().iter().zip(obs.chunks_exact(dim)) {
            assert!(a.route.is_none());
            // Spawned in the lane in either direction: the terms follow the travel direction.
            assert!(o[0].abs() < 0.6 && o[1].abs() < 0.3, "{}: {:?}", a.id, &o[..2]);
            assert!(o[6] > 3.0 && o[6] < 6.0, "{}: {:?}", a.id, &o[6..8]);
            assert_eq!(o[14], 1.0);
        }
        w.reset(None);
    }
    // Far from any road, the terms read 0.
    let map = w.map().clone();
    let (lo, hi) = map.extent();
    let mut far = None;
    'search: for i in 0..40 {
        for j in 0..40 {
            let p = lo + (hi - lo) * DVec2::new(i as f64 + 0.5, j as f64 + 0.5) / 40.0;
            if map.roads().nearest(p, 40.0).is_none() {
                far = Some(p);
                break 'search;
            }
        }
    }
    let p = far.expect("a point far from the roads");
    assert!(Follow::new(None, &map, p, 0.0).is_none());
    assert_eq!(road_state(None, &map, p, 0.0), [0.0, 0.0, ROAD_REACH]);
    // Beside a road: the distance to its surface.
    let road = &map.roads().roads()[0];
    let pr = road.line.project(road.line.point_at(0.5 * road.line.length()).truncate());
    let beside = pr.point.truncate() + DVec2::from_angle(pr.heading + FRAC_PI_2) * (0.5 * road.width + 4.0);
    let d = road_state(None, &map, beside, pr.heading)[2];
    assert!((d - 4.0).abs() < 0.3, "{d}");
}

#[test]
fn road_maps_are_required() {
    let wild = RURAL
        .replace(r#"type = "rural""#, r#"type = "wild""#)
        .replace(", count = 2", ", count = 1, config = { size = 128.0 }");
    let err = Scenario::from_toml(&wild).unwrap().compile().unwrap_err().to_string();
    assert!(err.contains("need roads"), "{err}");
    let drone = RURAL.replace("sedan_like", "cf2x").replace("on_road = true", "on_road = true, on_ground = false");
    let err = Scenario::from_toml(&drone).unwrap().compile().unwrap_err().to_string();
    assert!(err.contains("on_ground"), "{err}");
}

#[test]
fn recordings_keep_the_routes() {
    use autonomousim_sim::BatchSim;
    use autonomousim_sim::record::{Recorder, RecorderConfig, Recording};

    let sc = compile(RURAL);
    let path = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("routes.mcap");
    let mut b = BatchSim::from_compiled(sc.clone(), 1, 4, 1).unwrap();
    b.attach_recorder(0, Recorder::create(&path, RecorderConfig::default()).unwrap());
    let routes: Vec<_> = b.world(0).agents().iter().map(|a| a.route.clone().unwrap()).collect();
    b.step(&[&[0.0; 8]]);
    b.detach_recorder(0).unwrap().finish().unwrap();
    let rec = Recording::read(&path).unwrap();
    let ep = &rec.episodes[0];
    assert_eq!(ep.routes.len(), 4);
    for (recorded, route) in ep.routes.iter().zip(&routes) {
        assert_eq!(recorded.as_deref().unwrap().points(), route.points());
    }
}

#[test]
fn maps_without_farms_route_to_a_road_point() {
    // Map 8 of this pool has roads but no farm.
    let sc = compile(&RURAL.replace("seed = 3, count = 2", "seed = 1000, count = 9"));
    assert!(!sc.maps[8].roads().nodes().iter().any(|n| n.kind == NodeKind::Yard));
    let mut w = WorldInstance::new(sc.clone(), Seed::from_u64(1));
    let mut episodes = 0;
    while episodes < 3 {
        w.reset(None);
        if w.map_index() != 8 {
            continue;
        }
        episodes += 1;
        for a in w.agents() {
            let route = a.route.as_ref().expect("a route to a road point");
            assert!(route.length() > 100.0 && a.goals.len() > 3, "{}: {} m", a.id, route.length());
        }
    }
}
