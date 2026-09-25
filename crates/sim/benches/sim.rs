//! Policy steps of single worlds and of batches (throughput in env-steps/s: one env-step is
//! one policy step of one world, 10 physics ticks at 500/50 Hz for multirotors, 20 at
//! 1000/50 Hz for ground vehicles). Multirotors hold zero velocity and cars drive circles on
//! flat ground or stand on off-road maps, so that they stay active however long criterion
//! runs; each benchmark checks that afterwards.

use autonomousim_core::rng::Seed;
use autonomousim_sim::{BatchSim, CompiledScenario, Scenario, WorldInstance};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use std::sync::Arc;

fn compile(toml: &str) -> Arc<CompiledScenario> {
    Arc::new(Scenario::from_toml(toml).unwrap().compile().unwrap())
}

const HOVER: &str = r#"
    map = { type = "testworld", kind = "flat", size = 200.0 }
    [[groups]]
    action_mode = "velocity"
    spawn = { agl = [2.0, 2.0] }
"#;

const SENSORS: &str = r#"
    map = { type = "testworld", kind = "flat", size = 200.0 }
    [randomize_environment]
    wind_speed = [2.0, 2.0]
    turbulence_w20 = [7.7, 7.7]
    [[groups]]
    action_mode = "velocity"
    spawn = { agl = [2.0, 2.0] }
    sensors = [
        { name = "imu", type = "imu" }, { name = "gps", type = "gps" },
        { name = "baro", type = "baro" }, { name = "mag", type = "mag" },
    ]
"#;

const FOREST_LIDAR: &str = r#"
    map = { type = "testworld", kind = "forest_patch", size = 400.0, density = 150.0, seed = 1 }
    [[groups]]
    action_mode = "velocity"
    spawn = { agl = [2.0, 3.0] }
    sensors = [ { name = "imu", type = "imu" }, { name = "lidar", type = "lidar" } ]
    obs = [ { term = "goal_rel_body" }, { term = "rot6d" }, { term = "lin_vel_body" }, { term = "ang_vel_body" },
            { term = "lidar_log", sensor = "lidar" } ]
"#;

const SWARM: &str = r#"
    map = { type = "testworld", kind = "flat", size = 200.0 }
    [[groups]]
    count = 128
    action_mode = "velocity"
    spawn = { layout = { type = "grid", spacing = 1.5 }, agl = [2.0, 2.0] }
"#;

const CAR: &str = r#"
    map = { type = "testworld", kind = "flat", size = 400.0 }
    [[groups]]
    vehicle = "sedan_like"
    action_mode = "vk"
    spawn = { margin = 60.0 }
"#;

const TRUCK_OFFROAD_LIDAR: &str = r#"
    map = { type = "wild", preset = "offroad", seed = 1, cache = false, config = { size = 256.0 } }
    [[groups]]
    vehicle = "offroad_4x4"
    action_mode = "vk"
    sensors = [ { name = "imu", type = "imu" }, { name = "lidar", type = "lidar" } ]
    obs = [ { term = "goal_rel_heading" }, { term = "speed" }, { term = "pitch_roll" }, { term = "wheel_speeds" },
            { term = "wheel_slip" }, { term = "gear_rpm" }, { term = "lidar_log", sensor = "lidar" } ]
"#;

/// Cars in a 8 × 8 grid, standing on their brakes.
const CARS: &str = r#"
    map = { type = "testworld", kind = "flat", size = 400.0 }
    [[groups]]
    count = 64
    vehicle = "sedan_like"
    action_mode = "vk"
    spawn = { layout = { type = "grid", spacing = 8.0 } }
"#;

/// A slow left circle (radius 20 m for the sedan).
fn circle_action(sc: &CompiledScenario) -> Vec<f32> {
    let g = &sc.groups[0];
    [0.3f32, 0.3].repeat(g.spec.count)
}

fn hover_action(sc: &CompiledScenario) -> Vec<f32> {
    let g = &sc.groups[0];
    vec![0.0; g.spec.count * g.act_dim()]
}

fn assert_active(w: &WorldInstance, name: &str) {
    let off = w.agents().iter().filter(|a| a.disabled).count();
    assert_eq!(off, 0, "{name}: {off} agents were disabled during the benchmark");
}

fn world(c: &mut Criterion) {
    let mut g = c.benchmark_group("world_step");
    let quads = [("quad", HOVER), ("quad_sensors_wind", SENSORS), ("quad_forest_lidar", FOREST_LIDAR)];
    let cars = [("car", CAR), ("truck_offroad_lidar", TRUCK_OFFROAD_LIDAR)];
    for ((name, toml), driving) in quads.into_iter().zip([false; 3]).chain(cars.into_iter().zip([true, false])) {
        let sc = compile(toml);
        let a = if driving { circle_action(&sc) } else { hover_action(&sc) };
        let mut w = WorldInstance::new(sc, Seed::from_u64(0));
        let mut obs = vec![0.0; w.scenario().groups[0].obs_dim()];
        g.throughput(Throughput::Elements(1));
        g.bench_function(name, |b| {
            b.iter(|| {
                w.set_actions(0, &a);
                w.step();
                w.observe(0, &mut obs);
                black_box(&obs);
            })
        });
        assert_active(&w, name);
    }
    g.finish();

    let mut g = c.benchmark_group("swarm");
    g.sample_size(20);
    let threads = std::env::var("AUTONOMOUSIM_BENCH_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(10);
    for n in [128, 256] {
        let sc = compile(&SWARM.replace("count = 128", &format!("count = {n}")));
        let mut w = WorldInstance::new(sc, Seed::from_u64(0));
        // Agent-steps: n agents per policy step, hovering at their spawn.
        g.throughput(Throughput::Elements(n));
        g.bench_function(format!("{n}_agents"), |b| b.iter(|| w.step()));
        assert_active(&w, "swarm");
        let a = hover_action(w.scenario());
        let mut b = BatchSim::from_compiled(w.scenario().clone(), 1, 0, threads).unwrap();
        g.bench_function(format!("{n}_agents_in_pool"), |bench| bench.iter(|| b.step(&[&a])));
        assert_active(b.world(0), "swarm in pool");
    }
    let mut w = WorldInstance::new(compile(CARS), Seed::from_u64(0));
    // 64 cars per policy step, standing.
    g.throughput(Throughput::Elements(64));
    g.bench_function("64_cars", |b| b.iter(|| w.step()));
    assert_active(&w, "cars");
    g.finish();
}

fn batch(c: &mut Criterion) {
    let threads = std::env::var("AUTONOMOUSIM_BENCH_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(10);
    let mut g = c.benchmark_group(format!("batch_{threads}t"));
    g.sample_size(20);
    for (name, toml) in [("quad", HOVER), ("quad_forest_lidar", FOREST_LIDAR), ("car", CAR)] {
        let sc = compile(toml);
        let n = 256;
        let a = if name == "car" { circle_action(&sc) } else { hover_action(&sc) }.repeat(n);
        let mut b = BatchSim::from_compiled(sc, n, 0, threads).unwrap();
        g.throughput(Throughput::Elements(n as u64));
        g.bench_function(format!("{name}_x{n}"), |bench| {
            bench.iter(|| {
                b.step(&[&a]);
                black_box(b.obs(0));
            })
        });
        (0..n).for_each(|i| assert_active(b.world(i), name));
    }
    g.finish();
}

criterion_group!(benches, world, batch);
criterion_main!(benches);
