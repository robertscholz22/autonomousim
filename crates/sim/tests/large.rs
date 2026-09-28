//! Drones on a tiled (large) wild map: spawns sample the lazily generated tiles, agents hold
//! their spawn positions, ground vehicles are rejected (tiled maps have no drive grids), and
//! aircraft waypoints kilometres apart keep within a climb grade.

use autonomousim_core::rng::Seed;
use autonomousim_core::terrain::Terrain;
use autonomousim_sim::{Scenario, WorldInstance};
use std::sync::Arc;

const LARGE: &str = r#"
    name = "large"
    map = { type = "wild", preset = "large", seed = 3, cache = false, config = { size = 4096.0 } }
    [[groups]]
    name = "quads"
    count = 16
    vehicle = "iris_like"
    spawn = { agl = [20.0, 40.0], min_separation = 50.0 }
"#;

#[test]
fn drones_fly_on_a_tiled_map() {
    let sc = Arc::new(Scenario::from_toml(LARGE).unwrap().compile().unwrap());
    let mut w = WorldInstance::new(sc, Seed::from_u64(1));
    let map = w.map().clone();
    assert!(map.is_tiled());
    let tiles = map.terrain().tiled().unwrap().clone();
    let start: Vec<_> = w.agents().iter().map(|a| a.vehicle.position()).collect();
    for p in &start {
        let agl = p.z - map.terrain().height(p.x, p.y);
        assert!((19.0..41.0).contains(&agl), "spawn {p} at {agl} m above ground");
    }
    // Spawns spread over the map, so tiles far apart were generated.
    assert!(tiles.generated() >= 4, "{} tiles", tiles.generated());
    for _ in 0..250 {
        w.step();
    }
    for (a, p) in w.agents().iter().zip(&start) {
        assert!(a.vehicle.position().distance(*p) < 0.5, "{} drifted from {p}", a.vehicle.position());
    }
}

#[test]
fn ground_vehicles_are_rejected_on_tiled_maps() {
    let toml = LARGE.replace("vehicle = \"iris_like\"", "vehicle = \"offroad_4x4\"");
    let err = Scenario::from_toml(&toml).unwrap().compile().expect_err("rejected").to_string();
    assert!(err.contains("tiled"), "{err}");
}

/// Waypoints 1–1.5 km apart around the relief climb or descend at most `grade` per metre from
/// the previous point.
#[test]
fn aircraft_waypoints_keep_their_grade() {
    let toml = r#"
        name = "waypoints"
        map = { type = "wild", preset = "large", seed = 3, cache = false, config = { size = 4096.0 } }
        [[groups]]
        name = "uav"
        count = 4
        vehicle = "aerosonde_like"
        spawn = { agl = [150.0, 150.0], margin = 500.0, min_separation = 200.0 }
        goals = { kind = "random", count = 3, distance = [1000.0, 1500.0], agl = [100.0, 200.0], grade = 0.08, margin = 500.0, clearance = 20.0, radius = 50.0 }
    "#;
    let sc = Arc::new(Scenario::from_toml(toml).unwrap().compile().unwrap());
    let w = WorldInstance::new(sc, Seed::from_u64(2));
    let map = w.map().clone();
    for a in w.agents() {
        let mut prev = a.vehicle.position();
        assert_eq!(a.goals.len(), 3);
        for g in &a.goals {
            let p = g.position;
            let d = p.truncate().distance(prev.truncate());
            let agl = p.z - map.surface_height(p.x, p.y);
            assert!((999.0..1501.0).contains(&d), "goal {p} {d} m from {prev}");
            assert!((99.0..201.0).contains(&agl), "goal {p} {agl} m above ground");
            assert!((p.z - prev.z).abs() <= 0.08 * d, "goal {p} from {prev}: grade {}", (p.z - prev.z) / d);
            prev = p;
        }
    }
    // Validated before the map is built: a flat test world will do.
    let ground = toml.replace("aerosonde_like", "offroad_4x4").replace(
        r#"{ type = "wild", preset = "large", seed = 3, cache = false, config = { size = 4096.0 } }"#,
        r#"{ type = "testworld", kind = "flat", size = 200.0 }"#,
    );
    let err = Scenario::from_toml(&ground).unwrap().compile().expect_err("rejected").to_string();
    assert!(err.contains("grade"), "{err}");
}
