//! Single-beam rangefinder (laser or sonar altimeter).

use crate::latency::{DelayLine, Stamped};
use crate::{BodyKinematics, Mount, SensorEnv, SensorError, Targets, Timing};
use autonomousim_core::geometry::Ray;
use autonomousim_core::rng::{Seed, SimRng};
use autonomousim_core::time::Clock;
use glam::DVec3;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RangefinderConfig {
    pub rate_hz: u32,
    pub latency: f64,
    /// Beam origin on the body (the rotation is unused).
    pub mount: Mount,
    /// Beam direction in the body frame.
    pub direction: DVec3,
    /// Valid range (m); hits outside it give no reading.
    pub min_range: f64,
    pub max_range: f64,
    /// Noise standard deviation: absolute (m) plus a fraction of the range.
    pub noise: f64,
    pub noise_relative: f64,
    /// Probability that a measurement is lost.
    pub dropout: f64,
    pub targets: Targets,
}

impl Default for RangefinderConfig {
    /// Downward-looking laser altimeter (LightWare / TFmini class).
    fn default() -> Self {
        Self {
            rate_hz: 50,
            latency: 0.0,
            mount: Mount::default(),
            direction: DVec3::NEG_Z,
            min_range: 0.05,
            max_range: 40.0,
            noise: 0.02,
            noise_relative: 0.005,
            dropout: 0.0,
            targets: Targets::default(),
        }
    }
}

impl RangefinderConfig {
    pub fn validate(&self) -> Result<(), SensorError> {
        let ok = self.direction.length() > 1e-9
            && self.direction.is_finite()
            && self.min_range >= 0.0
            && self.max_range > self.min_range
            && self.max_range.is_finite()
            && [self.noise, self.noise_relative].iter().all(|x| x.is_finite() && *x >= 0.0)
            && (0.0..=1.0).contains(&self.dropout);
        if ok { Ok(()) } else { Err(SensorError::InvalidConfig(format!("{self:?}"))) }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RangeReading {
    /// Distance along the beam (m), if something within range returned it.
    pub range: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct Rangefinder {
    config: RangefinderConfig,
    timing: Timing,
    direction: DVec3,
    rng: SimRng,
    out: DelayLine<RangeReading>,
}

impl Rangefinder {
    pub fn new(config: RangefinderConfig, clock: &Clock, seed: Seed) -> Result<Self, SensorError> {
        config.validate()?;
        let timing = Timing::new("rangefinder", clock, config.rate_hz, config.latency)?;
        Ok(Self {
            timing,
            direction: config.direction.normalize(),
            rng: seed.rng(),
            out: DelayLine::new(timing.latency),
            config,
        })
    }

    pub fn config(&self) -> &RangefinderConfig {
        &self.config
    }

    pub fn reset(&mut self, seed: Seed) {
        self.rng = seed.rng();
        self.out.clear();
    }

    /// Noise-free range for the current pose.
    pub fn ideal(&self, kin: &BodyKinematics, env: &SensorEnv) -> Option<f64> {
        let c = &self.config;
        let ray = Ray::new(kin.position + kin.attitude * c.mount.position, kin.attitude * self.direction);
        let hit = env.rays.raycast(&ray, c.max_range, c.targets.query_mask())?;
        (c.targets.returns(hit.kind) && hit.toi >= c.min_range).then_some(hit.toi)
    }

    pub fn update(&mut self, tick: u64, time: f64, kin: &BodyKinematics, env: &SensorEnv) -> bool {
        if self.timing.is_due(tick) {
            let c = &self.config;
            let mut range = self.ideal(kin, env);
            if c.dropout > 0.0 && self.rng.chance(c.dropout) {
                range = None;
            }
            if let Some(r) = range {
                let sigma = c.noise + c.noise_relative * r;
                let noisy = if sigma > 0.0 { r + sigma * self.rng.normal() } else { r };
                range = Some(noisy.clamp(c.min_range, c.max_range));
            }
            self.out.push(Stamped { tick, time, value: RangeReading { range } });
        }
        self.out.poll(tick)
    }

    pub fn latest(&self) -> Option<&Stamped<RangeReading>> {
        self.out.latest()
    }
}
