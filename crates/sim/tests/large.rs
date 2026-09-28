//! Drones on a tiled (large) wild map: spawns sample the lazily generated tiles, agents hold
//! their spawn positions, and ground vehicles are rejected (tiled maps have no drive grids).

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
