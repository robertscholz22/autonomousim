//! Configuration and dispatch over all sensor kinds, for vehicles that carry a list of named
//! sensors.
//!
//! ```toml
//! [[sensors]]
//! name = "imu"
//! type = "imu"
//! rate_hz = 500
//!
//! [[sensors]]
//! name = "lidar"
//! type = "lidar"
//! max_range = 30.0
//! pattern = { type = "rings", elevations = [-10.0, 0.0, 10.0], azimuths = 32, azimuth_fov = 360.0 }
//! ```

use crate::{
    BaroConfig, Barometer, BodyKinematics, Gps, GpsConfig, GroundTruthConfig, GroundTruthSensor, Imu, ImuConfig, Lidar,
    LidarConfig, MagConfig, Magnetometer, Rangefinder, RangefinderConfig, SensorEnv, SensorError,
};
use autonomousim_core::rng::Seed;
use autonomousim_core::time::Clock;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SensorConfig {
    Imu(ImuConfig),
    Gps(GpsConfig),
    Baro(BaroConfig),
    Mag(MagConfig),
    Rangefinder(RangefinderConfig),
    Lidar(LidarConfig),
    GroundTruth(GroundTruthConfig),
}

impl SensorConfig {
    pub fn kind(&self) -> &'static str {
        match self {
            SensorConfig::Imu(_) => "imu",
            SensorConfig::Gps(_) => "gps",
            SensorConfig::Baro(_) => "baro",
            SensorConfig::Mag(_) => "mag",
            SensorConfig::Rangefinder(_) => "rangefinder",
            SensorConfig::Lidar(_) => "lidar",
            SensorConfig::GroundTruth(_) => "ground_truth",
        }
    }
}

/// A named sensor; the name selects its random stream and its observation terms.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SensorSpec {
    pub name: String,
    /// Unit carrying the sensor on a ground vehicle with trailers (0: the towing unit; its
    /// mount is in that unit's frame). Only rangefinders and LiDARs ride on the units behind.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub unit: usize,
    #[serde(flatten)]
    pub config: SensorConfig,
}

fn is_zero(x: &usize) -> bool {
    *x == 0
}

/// One sensor of any kind. Not boxed: vehicles keep their sensors in a `Vec` and update them
/// in place every tick, so the size spread costs less than a pointer chase would.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug)]
pub enum Sensor {
    Imu(Imu),
    Gps(Gps),
    Baro(Barometer),
    Mag(Magnetometer),
    Rangefinder(Rangefinder),
    Lidar(Lidar),
    GroundTruth(GroundTruthSensor),
}

impl Sensor {
    pub fn new(config: &SensorConfig, clock: &Clock, seed: Seed) -> Result<Self, SensorError> {
        Ok(match config {
            SensorConfig::Imu(c) => Sensor::Imu(Imu::new(c.clone(), clock, seed)?),
            SensorConfig::Gps(c) => Sensor::Gps(Gps::new(c.clone(), clock, seed)?),
            SensorConfig::Baro(c) => Sensor::Baro(Barometer::new(c.clone(), clock, seed)?),
            SensorConfig::Mag(c) => Sensor::Mag(Magnetometer::new(c.clone(), clock, seed)?),
            SensorConfig::Rangefinder(c) => Sensor::Rangefinder(Rangefinder::new(c.clone(), clock, seed)?),
            SensorConfig::Lidar(c) => Sensor::Lidar(Lidar::new(c.clone(), clock, seed)?),
            SensorConfig::GroundTruth(c) => Sensor::GroundTruth(GroundTruthSensor::new(c.clone(), clock)?),
        })
    }

    /// New episode with the sensor's random stream `seed`.
    pub fn reset(&mut self, seed: Seed) {
        match self {
            Sensor::Imu(s) => s.reset(seed),
            Sensor::Gps(s) => s.reset(seed),
            Sensor::Baro(s) => s.reset(seed),
            Sensor::Mag(s) => s.reset(seed),
            Sensor::Rangefinder(s) => s.reset(seed),
            Sensor::Lidar(s) => s.reset(seed),
            Sensor::GroundTruth(s) => s.reset(),
        }
    }

    /// Advance one physics tick; true if a new reading became visible.
    pub fn update(&mut self, tick: u64, time: f64, kin: &BodyKinematics, env: &SensorEnv) -> bool {
        match self {
            Sensor::Imu(s) => s.update(tick, time, kin, env),
            Sensor::Gps(s) => s.update(tick, time, kin, env),
            Sensor::Baro(s) => s.update(tick, time, kin, env),
            Sensor::Mag(s) => s.update(tick, time, kin, env),
            Sensor::Rangefinder(s) => s.update(tick, time, kin, env),
            Sensor::Lidar(s) => s.update(tick, time, kin, env),
            Sensor::GroundTruth(s) => s.update(tick, time, kin, env),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specs_parse_from_toml() {
        #[derive(Deserialize)]
        struct File {
            sensors: Vec<SensorSpec>,
        }
        let f: File = toml::from_str(
            r#"
            [[sensors]]
            name = "imu"
            type = "imu"
            rate_hz = 250
            gyro = { noise_density = 1e-4 }

            [[sensors]]
            name = "lidar"
            type = "lidar"
            max_range = 30.0
            pattern = { type = "rings", elevations = [-10.0, 0.0, 10.0], azimuths = 32, azimuth_fov = 360.0 }

            [[sensors]]
            name = "truth"
            type = "ground_truth"
            "#,
        )
        .unwrap();
        let [imu, lidar, truth] = &f.sensors[..] else { panic!() };
        let SensorConfig::Imu(c) = &imu.config else { panic!() };
        assert_eq!((c.rate_hz, c.gyro.noise_density, c.gyro.range), (250, 1e-4, 0.0));
        let SensorConfig::Lidar(c) = &lidar.config else { panic!() };
        assert_eq!((c.max_range, c.pattern.directions().len()), (30.0, 96));
        assert_eq!(truth.config.kind(), "ground_truth");
        let clock = Clock::new(500);
        for s in &f.sensors {
            Sensor::new(&s.config, &clock, Seed::from_u64(0)).unwrap();
        }
        // Rates must divide the physics rate; latencies must be whole steps.
        let bad = SensorConfig::Gps(GpsConfig { rate_hz: 7, ..GpsConfig::default() });
        assert!(Sensor::new(&bad, &clock, Seed::from_u64(0)).is_err());
        let bad = SensorConfig::Gps(GpsConfig { latency: 0.0031, ..GpsConfig::default() });
        assert!(Sensor::new(&bad, &clock, Seed::from_u64(0)).is_err());
        // Unknown fields are rejected.
        assert!(toml::from_str::<File>("[[sensors]]\nname = \"b\"\ntype = \"baro\"\nnoize = 1.0").is_err());
    }
}
