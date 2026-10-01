//! Step costs with pedestrians: the urban showcase (`assets/scenarios/urban_traffic.toml`) with
//! and without its 1,000, and a training world (a learning sedan among 50 hybrid NPC cars on
//! an urban training map) with and without 100.
//! `cargo run --release -p autonomousim-sim --example showcase_bench [seconds]`.

use autonomousim_core::rng::Seed;
use autonomousim_sim::{Scenario, WorldInstance};
use std::sync::Arc;
use std::time::Instant;

/// Seconds of stepping (and observing the first group) per simulated second.
fn cost(toml: &str, seconds: f64) -> f64 {
    let sc = Arc::new(Scenario::from_toml(toml).unwrap().compile().unwrap());
    let mut w = WorldInstance::new(sc.clone(), Seed::from_u64(1));
    let steps = (seconds * f64::from(sc.spec.policy_hz)) as usize;
    let mut obs = vec![0.0; sc.groups[0].obs_dim() * sc.groups[0].spec.count];
    let start = Instant::now();
    for _ in 0..steps {
        w.step();
        w.observe(0, &mut obs);
    }
    start.elapsed().as_secs_f64() / seconds
}

fn training(peds: usize) -> String {
    format!(
        r#"
        physics_hz = 1000
        policy_hz = 50
        map = {{ type = "urban", seed = 2, count = 1 }}
        [pedestrians]
        count = {peds}
        [[groups]]
        name = "ego"
        vehicle = "sedan_like"
        spawn = {{ on_ground = true, on_road = true }}
        obs = [{{ term = "traffic" }}, {{ term = "pedestrians" }}, {{ term = "lanes" }}]
        [[groups]]
        name = "cars"
        count = 50
        vehicle = "sedan_like"
        physics = "hybrid"
        driver = {{ type = "traffic" }}
        spawn = {{ on_ground = true, on_road = true, min_separation = 20.0 }}
        disable_on_terminal = false
        "#
    )
}

fn main() {
    let seconds: f64 = std::env::args().nth(1).map_or(20.0, |s| s.parse().unwrap());
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../assets/scenarios/urban_traffic.toml");
    let showcase = std::fs::read_to_string(path).unwrap();
    let (with, without) = (cost(&showcase, seconds), cost(&showcase.replace("count = 1000", "count = 0"), seconds));
    println!("showcase: {with:.3} s per simulated second, {without:.3} without its pedestrians");
    let (with, without) = (cost(&training(100), seconds), cost(&training(0), seconds));
    println!(
        "training world: {:.0} µs per policy step, {:.0} without its 100 pedestrians (+{:.0} %)",
        with * 1e6 / 50.0,
        without * 1e6 / 50.0,
        100.0 * (with / without - 1.0)
    );
}
