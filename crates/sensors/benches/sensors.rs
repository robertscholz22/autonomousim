//! Cost per update of each sensor (IMU every tick; the others per measurement) and of LiDAR
//! scans in a forest.

use autonomousim_core::rng::Seed;
use autonomousim_core::time::Clock;
use autonomousim_sensors::*;
use autonomousim_world::environment::{Atmosphere, MagneticField};
use autonomousim_world::{GeoOrigin, testworlds};
use criterion::{Criterion, criterion_group, criterion_main};
use glam::{DQuat, DVec3};
use std::hint::black_box;

fn bench(c: &mut Criterion) {
    let world = testworlds::forest_patch(400.0, 150.0, 1);
    let geo = GeoOrigin::default();
    let atmosphere = Atmosphere::default();
    let magnetic = MagneticField::dipole(geo.origin.lat_deg, geo.origin.lon_deg);
    let env = SensorEnv::new(&world, &geo, &atmosphere, &magnetic, 9.80665);
    let clock = Clock::new(500);
    let kin = BodyKinematics {
        position: DVec3::new(3.0, -7.0, world.surface_height(3.0, -7.0) + 3.0),
        attitude: DQuat::from_rotation_z(0.4),
        velocity: DVec3::new(2.0, 0.0, 0.0),
        specific_force: DVec3::new(0.0, 0.0, 9.8),
        ..Default::default()
    };
    let seed = Seed::from_u64(0);
    let mut tick = 0u64;
    let mut sensor = |name: &str, config: SensorConfig, c: &mut Criterion| {
        let mut s = Sensor::new(&config, &clock, seed).unwrap();
        // Measure on every iteration: step the tick by the sensor's divider.
        let divider = match &config {
            SensorConfig::Imu(c) => 500 / c.rate_hz,
            SensorConfig::Gps(c) => 500 / c.rate_hz,
            SensorConfig::Baro(c) => 500 / c.rate_hz,
            SensorConfig::Mag(c) => 500 / c.rate_hz,
            SensorConfig::Rangefinder(c) => 500 / c.rate_hz,
            SensorConfig::Lidar(c) => 500 / c.rate_hz,
            SensorConfig::GroundTruth(c) => 500 / c.rate_hz,
        };
        c.bench_function(&format!("sensor/{name}"), |b| {
            b.iter(|| {
                tick = (tick / u64::from(divider) + 1) * u64::from(divider);
                black_box(s.update(tick, 0.0, black_box(&kin), &env))
            })
        });
    };
    sensor("imu_ideal", SensorConfig::Imu(ImuConfig::ideal()), c);
    sensor("imu_adis16448", SensorConfig::Imu(ImuConfig::adis16448()), c);
    sensor("gps", SensorConfig::Gps(GpsConfig::default()), c);
    sensor("baro", SensorConfig::Baro(BaroConfig::default()), c);
    sensor("mag", SensorConfig::Mag(MagConfig::default()), c);
    sensor("rangefinder", SensorConfig::Rangefinder(RangefinderConfig::default()), c);
    sensor("ground_truth", SensorConfig::GroundTruth(GroundTruthConfig::default()), c);
    sensor("lidar_rl64_forest", SensorConfig::Lidar(LidarConfig::rl64()), c);
    let mut g = c.benchmark_group("slow");
    g.sample_size(10);
    let mut vlp = Lidar::new(LidarConfig::vlp16_like(), &clock, seed).unwrap();
    g.bench_function("sensor/lidar_vlp16_forest", |b| b.iter(|| vlp.scan(0, 0.0, black_box(&kin), &env)));
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
