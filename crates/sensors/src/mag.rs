//! Three-axis magnetometer: the map's Earth field in the sensor frame, distorted by a
//! per-episode hard-iron offset and scale error (a diagonal soft-iron model), plus white
//! noise.

use crate::latency::{DelayLine, Stamped};
use crate::noise::quantize3;
use crate::{BodyKinematics, Mount, SensorEnv, SensorError, Timing};
use autonomousim_core::rng::{Seed, SimRng};
use autonomousim_core::time::Clock;
use glam::{DQuat, DVec3};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MagConfig {
    pub rate_hz: u32,
    pub latency: f64,
    pub mount: Mount,
    /// White noise per axis (T).
    pub noise: f64,
    /// Standard deviation of the per-episode hard-iron offset per axis (T).
    pub hard_iron: f64,
    /// Standard deviation of the per-episode scale error per axis (fraction).
    pub scale: f64,
    /// Resolution (T; 0: continuous).
    pub resolution: f64,
}

impl Default for MagConfig {
    /// Consumer 3-axis magnetometer (IST8310 / HMC5883 class) after a coarse calibration.
    fn default() -> Self {
        Self {
            rate_hz: 50,
            latency: 0.0,
            mount: Mount::default(),
            noise: 3e-7,
            hard_iron: 1e-6,
            scale: 0.02,
            resolution: 0.0,
        }
    }
}

impl MagConfig {
    pub fn ideal() -> Self {
        Self { noise: 0.0, hard_iron: 0.0, scale: 0.0, ..Self::default() }
    }

    pub fn validate(&self) -> Result<(), SensorError> {
        let ok = [self.noise, self.hard_iron, self.scale, self.resolution].iter().all(|x| x.is_finite() && *x >= 0.0)
            && self.mount.rotation.is_finite();
        if ok { Ok(()) } else { Err(SensorError::InvalidConfig(format!("{self:?}"))) }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MagReading {
    /// Magnetic flux density in the sensor frame (T).
    pub field: DVec3,
}

#[derive(Clone, Debug)]
pub struct Magnetometer {
    config: MagConfig,
    timing: Timing,
    mount_inv: DQuat,
    offset: DVec3,
    scale: DVec3,
    rng: SimRng,
    out: DelayLine<MagReading>,
}

impl Magnetometer {
    pub fn new(config: MagConfig, clock: &Clock, seed: Seed) -> Result<Self, SensorError> {
        config.validate()?;
        let timing = Timing::new("mag", clock, config.rate_hz, config.latency)?;
        let mut mag = Self {
            timing,
            mount_inv: config.mount.quat().inverse(),
            offset: DVec3::ZERO,
            scale: DVec3::ONE,
            rng: seed.rng(),
            out: DelayLine::new(timing.latency),
            config,
        };
        mag.reset(seed);
        Ok(mag)
    }

    pub fn config(&self) -> &MagConfig {
        &self.config
    }

    pub fn reset(&mut self, seed: Seed) {
        self.rng = seed.rng();
        self.offset = self.config.hard_iron * self.rng.normal3();
        self.scale = DVec3::ONE + self.config.scale * self.rng.normal3();
        self.out.clear();
    }

    /// Per-episode hard-iron offset (sensor frame, T).
    pub fn hard_iron(&self) -> DVec3 {
        self.offset
    }

    pub fn update(&mut self, tick: u64, time: f64, kin: &BodyKinematics, env: &SensorEnv) -> bool {
        if self.timing.is_due(tick) {
            let b = self.mount_inv * (kin.attitude.inverse() * env.magnetic.enu);
            let mut field = self.scale * b + self.offset;
            if self.config.noise > 0.0 {
                field += self.config.noise * self.rng.normal3();
            }
            let value = MagReading { field: quantize3(field, self.config.resolution) };
            self.out.push(Stamped { tick, time, value });
        }
        self.out.poll(tick)
    }

    pub fn latest(&self) -> Option<&Stamped<MagReading>> {
        self.out.latest()
    }
}
