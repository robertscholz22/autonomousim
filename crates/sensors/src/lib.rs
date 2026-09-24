//! Sensor models: IMU, GPS, barometer, magnetometer, rangefinder, raycast LiDAR and ground
//! truth.
//!
//! Every sensor is updated once per physics tick with the vehicle's [`BodyKinematics`] and the
//! [`SensorEnv`]. It measures on the ticks its rate divides, and a reading becomes visible
//! after its latency (a whole number of ticks), so timing is exact and deterministic.
//! Randomness comes from the sensor's own [`Seed`](autonomousim_core::rng::Seed) stream.
//! Per-episode errors (turn-on biases, scale factors, hard-iron offsets) are drawn again on
//! every reset.
//!
//! Frames: world ENU, body FLU. Each sensor has a mount pose in the body frame, and vector
//! readings are in the sensor frame.

pub mod baro;
pub mod gps;
pub mod ground_truth;
pub mod imu;
pub mod latency;
pub mod lidar;
pub mod mag;
pub mod noise;
pub mod rangefinder;
pub mod suite;

pub use baro::{BaroConfig, BaroReading, Barometer};
pub use gps::{Gps, GpsConfig, GpsFix};
pub use ground_truth::{GroundTruth, GroundTruthConfig, GroundTruthSensor};
pub use imu::{Imu, ImuConfig, ImuReading, InertialNoise};
pub use latency::{DelayLine, Stamped};
pub use lidar::{BeamPattern, Lidar, LidarConfig, LidarScan, ReturnKind};
pub use mag::{MagConfig, MagReading, Magnetometer};
pub use rangefinder::{RangeReading, Rangefinder, RangefinderConfig};
pub use suite::{Sensor, SensorConfig, SensorSpec};

use autonomousim_core::geometry::{HitKind, HitMask, Ray, RayHit};
use autonomousim_core::math::Pose;
use autonomousim_core::time::{Clock, RateError};
use autonomousim_world::environment::{Atmosphere, MagneticField};
use autonomousim_world::{GeoOrigin, StaticWorld};
use glam::{DQuat, DVec3};
use serde::{Deserialize, Serialize};

#[derive(Debug, thiserror::Error)]
pub enum SensorError {
    #[error(transparent)]
    Rate(#[from] RateError),
    #[error("latency {latency} s of `{what}` is not a whole number of physics steps of {dt} s")]
    Latency { what: String, latency: f64, dt: f64 },
    #[error("invalid sensor configuration: {0}")]
    InvalidConfig(String),
}

/// Rigid-body motion of the vehicle carrying the sensors, at its centre of mass.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BodyKinematics {
    /// World position (m).
    pub position: DVec3,
    /// Body → world rotation.
    pub attitude: DQuat,
    /// World velocity (m/s).
    pub velocity: DVec3,
    /// Body angular velocity (rad/s).
    pub rates: DVec3,
    /// Specific force (all non-gravitational forces over mass; body frame, m/s²). This is
    /// what an ideal accelerometer at the centre of mass reads.
    pub specific_force: DVec3,
    /// Body angular acceleration (rad/s²).
    pub ang_acc: DVec3,
}

impl BodyKinematics {
    pub fn pose(&self) -> Pose {
        Pose::new(self.position, self.attitude)
    }
}

/// Geometry a ray-based sensor sees: the static world plus, optionally, other agents.
pub trait RayScene {
    /// First hit among the classes in `mask` within `max_toi`.
    fn raycast(&self, ray: &Ray, max_toi: f64, mask: HitMask) -> Option<RayHit>;
}

impl RayScene for StaticWorld {
    fn raycast(&self, ray: &Ray, max_toi: f64, mask: HitMask) -> Option<RayHit> {
        StaticWorld::raycast(self, ray, max_toi, mask)
    }
}

/// Everything outside the vehicle that the sensors need.
#[derive(Clone, Copy)]
pub struct SensorEnv<'a> {
    pub world: &'a StaticWorld,
    /// Ray targets for LiDAR and rangefinders (usually the world, or the world plus agents).
    pub rays: &'a dyn RayScene,
    pub geo: &'a GeoOrigin,
    pub atmosphere: &'a Atmosphere,
    pub magnetic: &'a MagneticField,
    /// Gravity magnitude (m/s²).
    pub gravity: f64,
}

impl<'a> SensorEnv<'a> {
    /// Environment with the static world as the only ray target.
    pub fn new(
        world: &'a StaticWorld,
        geo: &'a GeoOrigin,
        atmosphere: &'a Atmosphere,
        magnetic: &'a MagneticField,
        gravity: f64,
    ) -> Self {
        Self { world, rays: world, geo, atmosphere, magnetic, gravity }
    }
}

/// What a ray-based sensor responds to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Targets {
    pub terrain: bool,
    /// Water surfaces return the beam; otherwise they absorb it (no return), as near-infrared
    /// laser light mostly does.
    pub water: bool,
    pub solid: bool,
    /// Tree canopies and bushes return the beam; otherwise it passes through them.
    pub foliage: bool,
    /// Other agents return the beam; otherwise it passes through them.
    pub agents: bool,
}

impl Default for Targets {
    fn default() -> Self {
        Self { terrain: true, water: false, solid: true, foliage: true, agents: true }
    }
}

impl Targets {
    /// Mask for the ray query: water is always included, because it blocks the beam either way.
    pub fn query_mask(&self) -> HitMask {
        let bit = |on: bool, m: HitMask| if on { m } else { HitMask::NONE };
        bit(self.terrain, HitMask::TERRAIN)
            | HitMask::WATER
            | bit(self.solid, HitMask::SOLID)
            | bit(self.foliage, HitMask::FOLIAGE)
            | bit(self.agents, HitMask::AGENTS)
    }

    /// Whether a hit produces a return.
    pub fn returns(&self, kind: HitKind) -> bool {
        !matches!(kind, HitKind::Water) || self.water
    }
}

/// Mount of a sensor on the body: position (m) and rotation sensor → body.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Mount {
    pub position: DVec3,
    /// Rotation vector (rad) of the sensor frame relative to the body frame.
    pub rotation: DVec3,
}

impl Mount {
    pub fn quat(&self) -> DQuat {
        DQuat::from_scaled_axis(self.rotation)
    }

    /// World pose of the sensor frame.
    pub fn world_pose(&self, kin: &BodyKinematics) -> Pose {
        Pose::new(kin.position + kin.attitude * self.position, kin.attitude * self.quat())
    }
}

/// When a sensor measures and when its readings appear, in physics ticks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Timing {
    /// Measure on ticks divisible by this.
    pub divider: u32,
    /// Readings become visible this many ticks after the measurement.
    pub latency: u64,
}

impl Timing {
    /// Timing for a sensor at `rate_hz` with `latency` seconds on `clock`; both must be exact
    /// in physics ticks.
    pub fn new(what: &str, clock: &Clock, rate_hz: u32, latency: f64) -> Result<Self, SensorError> {
        let divider = clock.divider(what, rate_hz)?;
        let dt = clock.dt();
        let ticks = (latency / dt).round();
        if !(latency >= 0.0 && (latency / dt - ticks).abs() < 1e-6) {
            return Err(SensorError::Latency { what: what.to_string(), latency, dt });
        }
        Ok(Self { divider, latency: ticks as u64 })
    }

    #[inline]
    pub fn is_due(&self, tick: u64) -> bool {
        tick.is_multiple_of(u64::from(self.divider))
    }
}
