//! What hybrid NPCs cost a learning car (M8b step 5). The full model of a car is cheap (about
//! 1 µs per physics tick), so 50 NPCs cost several times a lone learning car's step whatever
//! their model; what is bounded is the cost of one NPC: less than the learning car's own.
//! Timed, so only in release builds.

use autonomousim_control::ground::GroundSetpoint;
use autonomousim_core::rng::Seed;
use autonomousim_sim::{Scenario, WorldInstance};
use std::sync::Arc;
use std::time::Instant;

/// A learning car on urban training map 2 at the ground preset's 1 kHz, with `npcs` hybrid
/// traffic NPCs.
fn world(npcs: usize) -> WorldInstance {
    let mut toml = r#"
        physics_hz = 1000
        policy_hz = 50
        map = { type = "urban", seed = 2, count = 1 }
        [[groups]]
        name = "ego"
        vehicle = "sedan_like"
        spawn = { on_ground = true, on_road = true }
    "#
    .to_string();
    if npcs > 0 {
        toml += &format!(
            r#"
        [[groups]]
        name = "npc"
        count = {npcs}
        vehicle = "sedan_like"
        physics = "hybrid"
        driver = {{ type = "traffic" }}
        spawn = {{ on_ground = true, on_road = true, min_separation = 20.0 }}
        disable_on_terminal = false
        "#
        );
    }
    let sc = Arc::new(Scenario::from_toml(&toml).unwrap().compile().unwrap());
    WorldInstance::new(sc, Seed::from_u64(2))
}

/// Seconds per policy step over `steps` steps (after a second's warm-up), and the most NPCs
/// promoted at once.
fn time(npcs: usize, steps: usize) -> (f64, usize) {
    let mut w = world(npcs);
    let mut promoted = 0;
    let mut run = |w: &mut WorldInstance, n: usize| {
        for _ in 0..n {
            w.set_command(0, GroundSetpoint::SpeedCurvature { speed: 6.0, curvature: 0.0 });
            w.step();
            promoted = promoted.max((1..w.agents().len()).filter(|&i| !w.agent(i).is_kinematic()).count());
        }
    };
    run(&mut w, 50);
    let start = Instant::now();
    run(&mut w, steps);
    (start.elapsed().as_secs_f64() / steps as f64, promoted)
}

#[test]
#[cfg_attr(debug_assertions, ignore = "timed: release builds only")]
fn a_hybrid_npc_costs_less_than_a_learning_car() {
    let (alone, _) = time(0, 1000);
    let (among, promoted) = time(50, 1000);
    let per_npc = (among - alone) / 50.0;
    eprintln!(
        "learning car alone {:.1} µs per step, among 50 hybrid NPCs {:.1} µs: {:.1} µs per NPC ({:.0} % of the car), \
         up to {promoted} promoted",
        1e6 * alone,
        1e6 * among,
        1e6 * per_npc,
        100.0 * per_npc / alone
    );
    assert!(per_npc < alone, "{:.1} µs per NPC", 1e6 * per_npc);
}
