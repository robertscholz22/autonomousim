//! Pitot-static airspeed sensor: the differential (dynamic) pressure of the flow along the probe,
//! with a per-episode offset and white noise, and the indicated and true airspeeds derived from
//! it.
//!
//! The probe points along the x axis of its mount and reads `½ρ·max(0, v·x̂)²` with `v` the
//! air-relative velocity at the probe (vehicle motion, rotation about the centre of mass, and
//! the wind of [`BodyKinematics::wind`]). Only the axial component counts: the reading falls
//! with cos² of the flow angle, a little faster than a real probe's.

use crate::latency::{DelayLine, Stamped};
use crate::noise::quantize;
use crate::{BodyKinematics, Mount, SensorEnv, SensorError, Timing};
use autonomousim_core::rng::{Seed, SimRng};
use autonomousim_core::time::Clock;
use glam::DVec3;
use serde::{Deserialize, Serialize};

/// Sea-level ISA density (kg/m³), the reference of indicated airspeed.
const RHO0: f64 = 1.225;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PitotConfig {
    pub rate_hz: u32,
    pub latency: f64,
    /// Probe position and orientation (the probe points along the mount's x axis).
    pub mount: Mount,
    /// White differential-pressure noise (Pa).
    pub noise: f64,
    /// Standard deviation of the per-episode pressure offset (Pa).
    pub turn_on_bias: f64,
    /// Resolution (Pa; 0: continuous).
    pub resolution: f64,
}

impl Default for PitotConfig {
    /// MS4525DO-class digital differential pressure sensor: ~1 Pa noise and a few pascals of
    /// zero offset (≈ 2 m/s of indicated airspeed near zero, negligible at cruise).
    fn default() -> Self {
        Self { rate_hz: 50, latency: 0.0, mount: Mount::default(), noise: 1.0, turn_on_bias: 2.0, resolution: 0.0 }
    }
}

impl PitotConfig {
    pub fn ideal() -> Self {
        Self { noise: 0.0, turn_on_bias: 0.0, ..Self::default() }
    }

    pub fn validate(&self) -> Result<(), SensorError> {
        let ok = [self.noise, self.turn_on_bias, self.resolution].iter().all(|x| x.is_finite() && *x >= 0.0);
        if ok { Ok(()) } else { Err(SensorError::InvalidConfig(format!("{self:?}"))) }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PitotReading {
    /// Differential pressure (Pa; negative readings come from the offset and noise).
    pub differential_pressure: f64,
    /// Indicated airspeed `√(2·q/ρ₀)` (m/s; 0 for negative pressure).
    pub indicated_airspeed: f64,
    /// True airspeed `√(2·q/ρ)` with the atmosphere's density at the probe (m/s).
    pub true_airspeed: f64,
}

#[derive(Clone, Debug)]
pub struct Pitot {
    config: PitotConfig,
    timing: Timing,
    bias: f64,
    rng: SimRng,
    out: DelayLine<PitotReading>,
}

impl Pitot {
    pub fn new(config: PitotConfig, clock: &Clock, seed: Seed) -> Result<Self, SensorError> {
        config.validate()?;
        let timing = Timing::new("pitot", clock, config.rate_hz, config.latency)?;
        let mut pitot = Self { timing, bias: 0.0, rng: seed.rng(), out: DelayLine::new(timing.latency), config };
        pitot.reset(seed);
        Ok(pitot)
    }

    pub fn config(&self) -> &PitotConfig {
        &self.config
    }

    pub fn reset(&mut self, seed: Seed) {
        self.rng = seed.rng();
        self.bias = self.config.turn_on_bias * self.rng.normal();
        self.out.clear();
    }

    /// Pressure offset of this episode (Pa).
    pub fn bias(&self) -> f64 {
        self.bias
    }

    pub fn update(&mut self, tick: u64, time: f64, kin: &BodyKinematics, env: &SensorEnv) -> bool {
        if self.timing.is_due(tick) {
            let m = &self.config.mount;
            let r = kin.attitude * m.position;
            let v = kin.velocity + kin.attitude * kin.rates.cross(m.position) - kin.wind;
            let axial = v.dot(kin.attitude * m.quat() * DVec3::X).max(0.0);
            let density = env.atmosphere.at_altitude(env.geo.altitude(kin.position + r)).density;
            let noise = if self.config.noise > 0.0 { self.config.noise * self.rng.normal() } else { 0.0 };
            let q = quantize(0.5 * density * axial * axial + self.bias + noise, self.config.resolution);
            let speed = |rho: f64| (2.0 * q.max(0.0) / rho).sqrt();
            let value = PitotReading {
                differential_pressure: q,
                indicated_airspeed: speed(RHO0),
                true_airspeed: speed(density),
            };
            self.out.push(Stamped { tick, time, value });
        }
        self.out.poll(tick)
    }

    pub fn latest(&self) -> Option<&Stamped<PitotReading>> {
        self.out.latest()
    }
}
