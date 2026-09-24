//! Sensor models against analytic expectations: a resting and a spinning IMU, Allan
//! variance, GPS latency and statistics, barometer, magnetometer, rangefinder and LiDAR in
//! scenes with known geometry.

use autonomousim_core::contact::StaticScene;
use autonomousim_core::material::MaterialId;
use autonomousim_core::math::Pose;
use autonomousim_core::rng::Seed;
use autonomousim_core::time::Clock;
use autonomousim_sensors::noise::allan_deviation;
use autonomousim_sensors::*;
use autonomousim_vehicles::multirotor::{AirData, GroundPlane, InitialState, MotorInit, Multirotor, StepEnv};
use autonomousim_vehicles::presets;
use autonomousim_world::environment::{Atmosphere, MagneticField};
use autonomousim_world::{GeoOrigin, StaticWorld, testworlds};
use glam::{DQuat, DVec3};
use std::sync::Arc;

const G: f64 = 9.80665;
const HZ: u32 = 500;

struct Scene {
    world: StaticWorld,
    geo: GeoOrigin,
    atmosphere: Atmosphere,
    magnetic: MagneticField,
}

impl Scene {
    fn new(world: StaticWorld) -> Self {
        let geo = GeoOrigin::default();
        let magnetic = MagneticField::dipole(geo.origin.lat_deg, geo.origin.lon_deg);
        Self { world, geo, atmosphere: Atmosphere::default(), magnetic }
    }

    fn env(&self) -> SensorEnv<'_> {
        SensorEnv::new(&self.world, &self.geo, &self.atmosphere, &self.magnetic, G)
    }
}

fn at(position: DVec3, attitude: DQuat) -> BodyKinematics {
    BodyKinematics { position, attitude, ..Default::default() }
}

fn seed(label: &str) -> Seed {
    Seed::from_u64(7).child(label)
}

/// A quadrotor resting on a 10° gravel slope: the ideal accelerometer reads the reaction to
/// gravity, `R⁻¹·(0, 0, g)`, and the gyroscope zero.
#[test]
fn static_imu_on_a_slope() {
    let angle = 10f64.to_radians();
    let scene = Scene::new(testworlds::incline(40.0, angle, MaterialId::GRAVEL));
    let def = Arc::new(presets::multirotor("iris_like").unwrap());
    let mut quad = Multirotor::new(def, 1.0 / f64::from(HZ));
    let start = DQuat::from_rotation_y(-angle);
    quad.reset(&InitialState::at(Pose::new(DVec3::new(0.0, 0.0, 0.3), start), MotorInit::Idle));
    let n = quad.num_rotors();
    let clock = Clock::new(HZ);
    let mut imu = Imu::new(ImuConfig::ideal(), &clock, seed("imu")).unwrap();
    let kin = |q: &Multirotor| BodyKinematics {
        position: q.position(),
        attitude: q.orientation(),
        velocity: q.lin_vel_world(),
        rates: q.ang_vel_body(),
        specific_force: q.specific_force_body(),
        ang_acc: q.ang_acc_body(),
    };
    for tick in 0..3 * u64::from(HZ) {
        let scene_ref = StaticScene {
            terrain: scene.world.terrain(),
            obstacles: scene.world.obstacles(),
            materials: scene.world.materials(),
        };
        let ground = GroundPlane::below(scene.world.terrain(), quad.position(), 2.0);
        let env =
            StepEnv { scene: Some(scene_ref), gravity: DVec3::new(0.0, 0.0, -G), air: AirData::default(), ground };
        quad.step(&vec![0.0; n], &env).unwrap();
        imu.update(tick, tick as f64 / f64::from(HZ), &kin(&quad), &scene.env());
    }
    let r = imu.latest().unwrap().value;
    let expected = quad.orientation().inverse() * DVec3::new(0.0, 0.0, G);
    println!("accel {} expected {}, gyro {}", r.accel, expected, r.gyro);
    assert!((r.accel - expected).length() < 1e-6 * G && r.gyro.length() < 1e-6);
    // Resting on the slope means tilted by it.
    assert!((r.accel.x / r.accel.z - angle.tan()).abs() < 0.02, "{}", r.accel);
}

/// Spinning at a constant rate with the IMU off the axis and rotated: centripetal
/// acceleration, rates in the sensor frame, and averaging over the sample interval.
#[test]
fn imu_lever_arm_and_mount() {
    let scene = Scene::new(testworlds::flat(20.0));
    let clock = Clock::new(HZ);
    let mount =
        Mount { position: DVec3::new(0.1, 0.0, 0.0), rotation: DVec3::new(0.0, 0.0, std::f64::consts::FRAC_PI_2) };
    let config = ImuConfig { rate_hz: 100, mount, ..ImuConfig::ideal() };
    let mut imu = Imu::new(config, &clock, seed("imu")).unwrap();
    let w = 3.0;
    let mut kin = at(DVec3::new(0.0, 0.0, 5.0), DQuat::IDENTITY);
    kin.rates = DVec3::new(0.0, 0.0, w);
    kin.specific_force = DVec3::new(0.0, 0.0, G);
    for tick in 0..10 {
        imu.update(tick, 0.0, &kin, &scene.env());
    }
    let r = imu.latest().unwrap();
    assert_eq!(r.tick, 5);
    // Body: (−ω²·0.1, 0, g). The sensor x-axis is the body y-axis, sensor y is body −x.
    let expected = DVec3::new(0.0, w * w * 0.1, G);
    assert!((r.value.accel - expected).length() < 1e-12, "{}", r.value.accel);
    assert!((r.value.gyro - DVec3::new(0.0, 0.0, w)).length() < 1e-12);
    // Angular acceleration adds the tangential term α×r (body +y, sensor +x).
    kin.ang_acc = DVec3::new(0.0, 0.0, 2.0);
    let ideal = imu.ideal(&kin);
    assert!((ideal.accel - expected - DVec3::new(0.2, 0.0, 0.0)).length() < 1e-12);
}

/// White noise and bias random walk recovered from the Allan deviation of a static IMU
/// within 10 %; turn-on biases differ per episode and repeat for the same seed.
#[test]
fn imu_allan_variance() {
    let scene = Scene::new(testworlds::flat(20.0));
    let clock = Clock::new(HZ);
    let (n, k) = (0.01, 0.0316);
    let noise = InertialNoise { noise_density: n, bias_random_walk: k, bias_tau: None, ..InertialNoise::IDEAL };
    let config = ImuConfig { accel: noise.clone(), gyro: noise, ..ImuConfig::ideal() };
    let mut imu = Imu::new(config, &clock, seed("imu")).unwrap();
    let kin = at(DVec3::ZERO, DQuat::IDENTITY);
    let samples = 2000 * HZ as usize;
    let mut series = (0..6).map(|_| Vec::with_capacity(samples)).collect::<Vec<_>>();
    for tick in 0..samples as u64 {
        imu.update(tick, 0.0, &kin, &scene.env());
        let r = imu.latest().unwrap().value;
        for (i, v) in r.accel.to_array().into_iter().chain(r.gyro.to_array()).enumerate() {
            series[i].push(v);
        }
    }
    let dt = 1.0 / f64::from(HZ);
    let mean_var = |m: usize| series.iter().map(|y| allan_deviation(y, dt, m).powi(2)).sum::<f64>() / 6.0;
    // τ = 0.01 s: white noise dominates (σ = N/√τ); τ = 10 s: random walk (σ = K·√(τ/3)).
    let n_est = (mean_var(5) * 0.01).sqrt();
    let k_est = (mean_var(5000) * 3.0 / 10.0).sqrt();
    println!("N {n_est:.5} (true {n}), K {k_est:.5} (true {k})");
    assert!((n_est / n - 1.0).abs() < 0.1 && (k_est / k - 1.0).abs() < 0.1);

    // Per-episode turn-on bias: reproducible per seed, different across seeds.
    let config = ImuConfig::adis16448();
    let mut imu = Imu::new(config, &clock, seed("a")).unwrap();
    let b1 = imu.biases();
    imu.reset(seed("b"));
    let b2 = imu.biases();
    imu.reset(seed("a"));
    assert!(imu.biases() == b1 && b1 != b2);
}

/// Fixes appear exactly `latency` after they were measured and describe the state at the
/// measurement; latitude/longitude round-trip to the map frame.
#[test]
fn gps_latency_is_exact() {
    let scene = Scene::new(testworlds::flat(2000.0));
    let clock = Clock::new(HZ);
    let mount = Mount { position: DVec3::new(0.0, 0.0, 0.1), rotation: DVec3::ZERO };
    let config = GpsConfig { rate_hz: 10, latency: 0.1, mount, ..GpsConfig::ideal() };
    let mut gps = Gps::new(config, &clock, seed("gps")).unwrap();
    let v = DVec3::new(3.0, -1.0, 0.5);
    let q = DQuat::from_rotation_x(0.3);
    let mut updates = Vec::new();
    for tick in 0..500u64 {
        let t = tick as f64 / f64::from(HZ);
        let mut kin = at(DVec3::new(0.0, 0.0, 10.0) + v * t, q);
        kin.velocity = v;
        if gps.update(tick, t, &kin, &scene.env()) {
            updates.push(tick);
        }
        if let Some(fix) = gps.latest() {
            // Newest fix whose delay has passed.
            assert_eq!(fix.tick, (tick - 50) / 50 * 50);
            let truth = DVec3::new(0.0, 0.0, 10.0) + v * fix.time + q * mount.position;
            assert!((fix.value.position - truth).length() < 1e-9);
            assert!((scene.geo.geodetic_to_enu(fix.value.geodetic) - truth).length() < 1e-6);
            assert!((fix.value.velocity - v).length() < 1e-12);
        } else {
            assert!(tick < 50);
        }
    }
    assert_eq!(updates, (1..10).map(|k| k * 50).collect::<Vec<_>>());
}

/// Default GPS errors: slowly wandering with the configured spread and correlation time.
#[test]
fn gps_error_statistics() {
    let scene = Scene::new(testworlds::flat(200.0));
    let clock = Clock::new(10);
    let config = GpsConfig { latency: 0.0, ..GpsConfig::default() };
    let mut gps = Gps::new(config.clone(), &clock, seed("gps")).unwrap();
    let kin = at(DVec3::ZERO, DQuat::IDENTITY);
    let (mut sum2, mut n) = (DVec3::ZERO, 0.0);
    for tick in 0..200_000u64 {
        gps.update(tick, 0.0, &kin, &scene.env());
        let e = gps.latest().unwrap().value.position;
        sum2 += e * e;
        n += 1.0;
    }
    let std = (sum2 / n).to_array().map(f64::sqrt);
    let h = config.drift_horizontal.hypot(config.noise_horizontal);
    let v = config.drift_vertical.hypot(config.noise_vertical);
    println!("GPS error std {std:?} (expected {h:.2}, {h:.2}, {v:.2})");
    assert!((std[0] / h - 1.0).abs() < 0.15 && (std[1] / h - 1.0).abs() < 0.15 && (std[2] / v - 1.0).abs() < 0.15);
}

#[test]
fn barometer_and_magnetometer() {
    let scene = Scene::new(testworlds::flat(200.0));
    let clock = Clock::new(HZ);
    let q = DQuat::from_euler(glam::EulerRot::ZYX, 1.0, 0.2, -0.4);
    let kin = at(DVec3::new(10.0, -20.0, 150.0), q);

    let mut baro = Barometer::new(BaroConfig::ideal(), &clock, seed("baro")).unwrap();
    baro.update(0, 0.0, &kin, &scene.env());
    let r = baro.latest().unwrap().value;
    let alt = scene.geo.altitude(kin.position);
    assert!((r.pressure - scene.atmosphere.at_altitude(alt).pressure).abs() < 1e-9);
    assert!((r.altitude - alt).abs() < 1e-6, "{} vs {alt}", r.altitude);
    // Noise: 1.5 Pa ≈ 12.5 cm at sea level.
    let mut baro =
        Barometer::new(BaroConfig { turn_on_bias: 0.0, drift: 0.0, ..BaroConfig::default() }, &clock, seed("b"))
            .unwrap();
    let mut sum2 = 0.0;
    for k in 0..5000u64 {
        baro.update(k * 10, 0.0, &kin, &scene.env());
        sum2 += (baro.latest().unwrap().value.altitude - alt).powi(2);
    }
    let std = (sum2 / 5000.0).sqrt();
    assert!((0.1..0.16).contains(&std), "{std}");

    let mut mag = Magnetometer::new(MagConfig::ideal(), &clock, seed("mag")).unwrap();
    mag.update(0, 0.0, &kin, &scene.env());
    let field = mag.latest().unwrap().value.field;
    assert!((field - q.inverse() * scene.magnetic.enu).length() < 1e-15);
    // Level: the horizontal field points to magnetic north.
    let mut mag = Magnetometer::new(MagConfig::ideal(), &clock, seed("mag")).unwrap();
    let yaw = 0.7;
    mag.update(0, 0.0, &at(DVec3::ZERO, DQuat::from_rotation_z(yaw)), &scene.env());
    let f = mag.latest().unwrap().value.field;
    let heading = std::f64::consts::FRAC_PI_2 - scene.magnetic.declination() - f.y.atan2(f.x);
    assert!((autonomousim_core::math::quat::wrap_angle(heading - yaw)).abs() < 1e-12, "{heading}");
}

#[test]
fn rangefinder_over_flat_ground_and_water() {
    let scene = Scene::new(testworlds::flat(200.0));
    let clock = Clock::new(HZ);
    let config = RangefinderConfig { noise: 0.0, noise_relative: 0.0, ..RangefinderConfig::default() };
    let mut rf = Rangefinder::new(config.clone(), &clock, seed("rf")).unwrap();
    let tilt = 0.3;
    rf.update(0, 0.0, &at(DVec3::new(0.0, 0.0, 3.0), DQuat::from_rotation_x(tilt)), &scene.env());
    let range = rf.latest().unwrap().value.range.unwrap();
    assert!((range - 3.0 / tilt.cos()).abs() < 1e-9);
    rf.update(10, 0.0, &at(DVec3::new(0.0, 0.0, 45.0), DQuat::IDENTITY), &scene.env());
    assert_eq!(rf.latest().unwrap().value.range, None);

    // Over a lake: absorbed by the water unless water returns.
    let lake = Scene::new(testworlds::lake(200.0, 5.0, -1.0));
    let kin = at(DVec3::new(0.0, 0.0, 2.0), DQuat::IDENTITY);
    rf.update(20, 0.0, &kin, &lake.env());
    assert_eq!(rf.latest().unwrap().value.range, None);
    let targets = Targets { water: true, ..Targets::default() };
    let mut rf = Rangefinder::new(RangefinderConfig { targets, ..config }, &clock, seed("rf")).unwrap();
    rf.update(0, 0.0, &kin, &lake.env());
    assert!((rf.latest().unwrap().value.range.unwrap() - 3.0).abs() < 1e-9);
}

/// Distance along `d` from `o` to the first of: ground z = 0, the arena's inner wall faces
/// x, y = ±half (height `h`).
fn arena_range(o: DVec3, d: DVec3, half: f64, h: f64) -> Option<f64> {
    let mut best = f64::INFINITY;
    if d.z < 0.0 {
        best = -o.z / d.z;
    }
    for (axis, sign) in [(0, 1.0), (0, -1.0), (1, 1.0), (1, -1.0)] {
        if d[axis] * sign > 0.0 {
            let t = (sign * half - o[axis]) / d[axis];
            let z = o.z + t * d.z;
            if (0.0..=h).contains(&z) && t < best {
                best = t;
            }
        }
    }
    best.is_finite().then_some(best)
}

/// Every beam of a tilted LiDAR in a walled arena matches the analytic intersection.
#[test]
fn lidar_in_walled_arena_is_exact() {
    let (half, height) = (10.0, 5.0);
    let scene = Scene::new(testworlds::walled_arena(half, height));
    let clock = Clock::new(HZ);
    let mount = Mount { position: DVec3::new(0.1, 0.0, 0.05), rotation: DVec3::new(0.0, 0.1, 0.0) };
    let config = LidarConfig {
        mount,
        pattern: BeamPattern::rings(8, -40.0, 30.0, 90, 360.0),
        noise: 0.0,
        max_range: 60.0,
        ..LidarConfig::rl64()
    };
    let mut lidar = Lidar::new(config, &clock, seed("lidar")).unwrap();
    let q = DQuat::from_euler(glam::EulerRot::ZYX, 0.5, 0.15, -0.1);
    let kin = at(DVec3::new(2.0, -3.0, 1.5), q);
    assert!(lidar.update(0, 0.0, &kin, &scene.env()));
    let scan = lidar.latest().unwrap();
    let pose = mount.world_pose(&kin);
    let (mut walls, mut ground, mut open) = (0, 0, 0);
    for (i, d) in lidar.directions().iter().enumerate() {
        let want = arena_range(pose.pos, pose.rot * *d, half, height);
        let got = scan.ranges[i];
        match want {
            Some(t) => {
                assert!((f64::from(got) - t).abs() < 1e-5 * t.max(1.0), "beam {i}: {got} vs {t}");
                match scan.kinds[i] {
                    ReturnKind::Terrain => ground += 1,
                    ReturnKind::Solid => walls += 1,
                    k => panic!("beam {i}: {k:?}"),
                }
            }
            None => {
                assert!(got.is_infinite() && scan.kinds[i] == ReturnKind::None, "beam {i}: {got}");
                open += 1;
            }
        }
    }
    println!("{walls} wall, {ground} ground, {open} open beams");
    assert!(walls > 100 && ground > 100 && open > 50);
    // World points lie on the surfaces.
    for p in scan.points(lidar.directions()) {
        let on_wall = (p.x.abs() - half).abs() < 1e-4 || (p.y.abs() - half).abs() < 1e-4;
        assert!(on_wall || p.z.abs() < 1e-4, "{p}");
    }
}

/// Canopies return the beam unless foliage is excluded; noise and dropout behave.
#[test]
fn lidar_foliage_noise_and_dropout() {
    let scene = Scene::new(testworlds::single_tree());
    let clock = Clock::new(HZ);
    let down = BeamPattern::Beams { directions: vec![DVec3::NEG_Z] };
    let kin = at(DVec3::new(0.5, 0.0, 20.0), DQuat::IDENTITY);
    let config = LidarConfig { pattern: down, noise: 0.0, ..LidarConfig::rl64() };
    let mut lidar = Lidar::new(config.clone(), &clock, seed("l")).unwrap();
    lidar.update(0, 0.0, &kin, &scene.env());
    assert_eq!(lidar.latest().unwrap().kinds[0], ReturnKind::Foliage);
    let targets = Targets { foliage: false, ..Targets::default() };
    let mut lidar = Lidar::new(LidarConfig { targets, ..config.clone() }, &clock, seed("l")).unwrap();
    lidar.update(0, 0.0, &kin, &scene.env());
    // The trunk (radius 0.25 m) is not below x = 0.5, so the beam reaches the ground.
    let s = lidar.latest().unwrap();
    assert!(s.kinds[0] == ReturnKind::Terrain && (s.ranges[0] - 20.0).abs() < 1e-5);

    let flat = Scene::new(testworlds::flat(100.0));
    let beams = BeamPattern::Beams { directions: vec![DVec3::NEG_Z; 20_000] };
    let config = LidarConfig { pattern: beams, noise: 0.05, dropout: 0.1, ..LidarConfig::rl64() };
    let mut lidar = Lidar::new(config, &clock, seed("l")).unwrap();
    lidar.update(0, 0.0, &at(DVec3::new(0.0, 0.0, 10.0), DQuat::IDENTITY), &flat.env());
    let s = lidar.latest().unwrap();
    let hits: Vec<f64> = s.ranges.iter().filter(|r| r.is_finite()).map(|&r| f64::from(r)).collect();
    let lost = 1.0 - hits.len() as f64 / 20_000.0;
    let mean = hits.iter().sum::<f64>() / hits.len() as f64;
    let std = (hits.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / hits.len() as f64).sqrt();
    println!("dropout {lost:.3}, mean {mean:.4}, std {std:.4}");
    assert!((lost - 0.1).abs() < 0.01 && (mean - 10.0).abs() < 0.002 && (std / 0.05 - 1.0).abs() < 0.03);
}

/// Same seed, same readings; the sensors' streams do not depend on each other.
#[test]
fn readings_are_deterministic() {
    let scene = Scene::new(testworlds::forest_patch(200.0, 150.0, 3));
    let clock = Clock::new(HZ);
    let specs = [
        SensorConfig::Imu(ImuConfig::adis16448()),
        SensorConfig::Gps(GpsConfig::default()),
        SensorConfig::Baro(BaroConfig::default()),
        SensorConfig::Mag(MagConfig::default()),
        SensorConfig::Rangefinder(RangefinderConfig::default()),
        SensorConfig::Lidar(LidarConfig { dropout: 0.05, ..LidarConfig::rl64() }),
        SensorConfig::GroundTruth(GroundTruthConfig::default()),
    ];
    let run = |order: &[usize]| {
        let mut sensors: Vec<(usize, Sensor)> =
            order.iter().map(|&i| (i, Sensor::new(&specs[i], &clock, seed(specs[i].kind())).unwrap())).collect();
        let mut log = vec![String::new(); specs.len()];
        for tick in 0..1000u64 {
            let t = tick as f64 / f64::from(HZ);
            let mut kin = at(DVec3::new(t, 0.5 * t, 5.0 + 0.2 * t), DQuat::from_rotation_z(0.3 * t));
            kin.velocity = DVec3::new(1.0, 0.5, 0.2);
            kin.rates = DVec3::new(0.0, 0.0, 0.3);
            kin.specific_force = DVec3::new(0.0, 0.0, G);
            for (i, s) in &mut sensors {
                if s.update(tick, t, &kin, &scene.env()) {
                    log[*i] += &match s {
                        Sensor::Imu(s) => format!("{:?}", s.latest()),
                        Sensor::Gps(s) => format!("{:?}", s.latest()),
                        Sensor::Baro(s) => format!("{:?}", s.latest()),
                        Sensor::Mag(s) => format!("{:?}", s.latest()),
                        Sensor::Rangefinder(s) => format!("{:?}", s.latest()),
                        Sensor::Lidar(s) => format!("{:?}", s.latest()),
                        Sensor::GroundTruth(s) => format!("{:?}", s.latest()),
                    };
                }
            }
        }
        log
    };
    let a = run(&[0, 1, 2, 3, 4, 5, 6]);
    assert_eq!(a, run(&[6, 5, 4, 3, 2, 1, 0]));
    assert!(a.iter().all(|l| !l.is_empty()));
}
