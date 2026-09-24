//! Barometer: static pressure of the ISA atmosphere at the sensor's altitude, with a
//! per-episode offset, slow drift and white noise, and the pressure altitude derived from it.

use crate::latency::{DelayLine, Stamped};
use crate::noise::{GaussMarkov, quantize};
use crate::{BodyKinematics, Mount, SensorEnv, SensorError, Timing};
use autonomousim_core::rng::{Seed, SimRng};
use autonomousim_core::time::Clock;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BaroConfig {
    pub rate_hz: u32,
    pub latency: f64,
    /// Sensor position on the body (the rotation is unused).
    pub mount: Mount,
    /// White pressure noise (Pa).
    pub noise: f64,
    /// Standard deviation of the per-episode pressure offset (Pa).
    pub turn_on_bias: f64,
    /// Stationary standard deviation (Pa) and correlation time (s) of the drift.
    pub drift: f64,
    pub drift_tau: f64,
    /// Resolution (Pa; 0: continuous).
    pub resolution: f64,
}

impl Default for BaroConfig {
    /// MS5611-class sensor at high oversampling: ~1.5 Pa (≈ 12 cm) noise; the offset stands for
    /// the unknown reference pressure of the day.
    fn default() -> Self {
        Self {
            rate_hz: 50,
            latency: 0.0,
            mount: Mount::default(),
            noise: 1.5,
            turn_on_bias: 20.0,
            drift: 3.0,
            drift_tau: 300.0,
            resolution: 0.0,
        }
    }
}

impl BaroConfig {
    pub fn ideal() -> Self {
        Self { noise: 0.0, turn_on_bias: 0.0, drift: 0.0, ..Self::default() }
    }

    pub fn validate(&self) -> Result<(), SensorError> {
        let ok =
            [self.noise, self.turn_on_bias, self.drift, self.resolution].iter().all(|x| x.is_finite() && *x >= 0.0)
                && self.drift_tau > 0.0;
        if ok { Ok(()) } else { Err(SensorError::InvalidConfig(format!("{self:?}"))) }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BaroReading {
    /// Static pressure (Pa).
    pub pressure: f64,
    /// Air temperature (K).
    pub temperature: f64,
    /// Pressure altitude (m) for the atmosphere's sea-level pressure setting.
    pub altitude: f64,
}

#[derive(Clone, Debug)]
pub struct Barometer {
    config: BaroConfig,
    timing: Timing,
    drift_gm: GaussMarkov,
    bias: f64,
    drift: f64,
    rng: SimRng,
    out: DelayLine<BaroReading>,
}

impl Barometer {
    pub fn new(config: BaroConfig, clock: &Clock, seed: Seed) -> Result<Self, SensorError> {
        config.validate()?;
        let timing = Timing::new("baro", clock, config.rate_hz, config.latency)?;
        let dt = clock.dt() * f64::from(timing.divider);
        let mut baro = Self {
            timing,
            drift_gm: GaussMarkov::from_sigma(config.drift, config.drift_tau, dt),
            bias: 0.0,
            drift: 0.0,
            rng: seed.rng(),
            out: DelayLine::new(timing.latency),
            config,
        };
        baro.reset(seed);
        Ok(baro)
    }

    pub fn config(&self) -> &BaroConfig {
        &self.config
    }

    pub fn reset(&mut self, seed: Seed) {
        self.rng = seed.rng();
        self.bias = self.config.turn_on_bias * self.rng.normal();
        self.drift = self.config.drift * self.rng.normal();
        self.out.clear();
    }

    /// Current pressure error from offset and drift (Pa).
    pub fn bias(&self) -> f64 {
        self.bias + self.drift
    }

    pub fn update(&mut self, tick: u64, time: f64, kin: &BodyKinematics, env: &SensorEnv) -> bool {
        if self.timing.is_due(tick) {
            let p = kin.position + kin.attitude * self.config.mount.position;
            let air = env.atmosphere.at_altitude(env.geo.altitude(p));
            self.drift = self.drift_gm.step(self.drift, &mut self.rng);
            let noise = if self.config.noise > 0.0 { self.config.noise * self.rng.normal() } else { 0.0 };
            let pressure = quantize(air.pressure + self.bias + self.drift + noise, self.config.resolution);
            let value = BaroReading {
                pressure,
                temperature: air.temperature,
                altitude: env.atmosphere.baro_altitude(pressure),
            };
            self.out.push(Stamped { tick, time, value });
        }
        self.out.poll(tick)
    }

    pub fn latest(&self) -> Option<&Stamped<BaroReading>> {
        self.out.latest()
    }
}
