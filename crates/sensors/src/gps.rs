//! GNSS receiver: position and velocity of the antenna with a slowly wandering error
//! (first-order Gauss–Markov, standing in for atmospheric and orbit errors), white noise, and
//! a transport delay. Positions are reported both in the map's ENU frame and as WGS-84
//! latitude/longitude/altitude through the map's geodetic origin.

use crate::latency::{DelayLine, Stamped};
use crate::noise::GaussMarkov;
use crate::{BodyKinematics, Mount, SensorEnv, SensorError, Timing};
use autonomousim_core::rng::{Seed, SimRng};
use autonomousim_core::time::Clock;
use autonomousim_world::Geodetic;
use glam::DVec3;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GpsConfig {
    pub rate_hz: u32,
    /// Age of a fix when it becomes available (s; whole physics steps).
    pub latency: f64,
    /// Antenna position on the body (the rotation is unused).
    pub mount: Mount,
    /// Standard deviation of the horizontal / vertical Gauss–Markov error (m).
    pub drift_horizontal: f64,
    pub drift_vertical: f64,
    /// Correlation time of that error (s).
    pub drift_tau: f64,
    /// White position noise per fix, horizontal / vertical (m).
    pub noise_horizontal: f64,
    pub noise_vertical: f64,
    /// White velocity noise per axis (m/s).
    pub noise_velocity: f64,
}

impl Default for GpsConfig {
    /// A single-frequency consumer receiver (u-blox M8/M9 class, open sky): about 2 m CEP.
    fn default() -> Self {
        Self {
            rate_hz: 10,
            latency: 0.1,
            mount: Mount::default(),
            drift_horizontal: 1.5,
            drift_vertical: 3.0,
            drift_tau: 60.0,
            noise_horizontal: 0.3,
            noise_vertical: 0.6,
            noise_velocity: 0.05,
        }
    }
}

impl GpsConfig {
    /// Error-free fixes (still at the configured rate and latency).
    pub fn ideal() -> Self {
        Self {
            drift_horizontal: 0.0,
            drift_vertical: 0.0,
            noise_horizontal: 0.0,
            noise_vertical: 0.0,
            noise_velocity: 0.0,
            ..Self::default()
        }
    }

    pub fn validate(&self) -> Result<(), SensorError> {
        let ok = [
            self.drift_horizontal,
            self.drift_vertical,
            self.noise_horizontal,
            self.noise_vertical,
            self.noise_velocity,
        ]
        .iter()
        .all(|x| x.is_finite() && *x >= 0.0)
            && self.drift_tau > 0.0
            && self.mount.position.is_finite();
        if ok { Ok(()) } else { Err(SensorError::InvalidConfig(format!("{self:?}"))) }
    }
}

/// One fix, valid at its measurement time.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct GpsFix {
    /// Antenna position in the map frame (ENU, m).
    pub position: DVec3,
    /// The same position as latitude/longitude (degrees) and ellipsoidal altitude (m).
    pub geodetic: Geodetic,
    /// Antenna velocity (ENU, m/s).
    pub velocity: DVec3,
    /// Reported 1σ horizontal and vertical position accuracy (m).
    pub eph: f64,
    pub epv: f64,
}

#[derive(Clone, Debug)]
pub struct Gps {
    config: GpsConfig,
    timing: Timing,
    horizontal: GaussMarkov,
    vertical: GaussMarkov,
    drift: DVec3,
    rng: SimRng,
    out: DelayLine<GpsFix>,
}

impl Gps {
    pub fn new(config: GpsConfig, clock: &Clock, seed: Seed) -> Result<Self, SensorError> {
        config.validate()?;
        let timing = Timing::new("gps", clock, config.rate_hz, config.latency)?;
        let dt = clock.dt() * f64::from(timing.divider);
        let mut gps = Self {
            timing,
            horizontal: GaussMarkov::from_sigma(config.drift_horizontal, config.drift_tau, dt),
            vertical: GaussMarkov::from_sigma(config.drift_vertical, config.drift_tau, dt),
            drift: DVec3::ZERO,
            rng: seed.rng(),
            out: DelayLine::new(timing.latency),
            config,
        };
        gps.reset(seed);
        Ok(gps)
    }

    pub fn config(&self) -> &GpsConfig {
        &self.config
    }

    pub fn timing(&self) -> Timing {
        self.timing
    }

    /// New episode: the slow error starts from its stationary distribution.
    pub fn reset(&mut self, seed: Seed) {
        self.rng = seed.rng();
        let n = self.rng.normal3();
        self.drift = DVec3::new(
            n.x * self.config.drift_horizontal,
            n.y * self.config.drift_horizontal,
            n.z * self.config.drift_vertical,
        );
        self.out.clear();
    }

    /// Current slow position error (m).
    pub fn drift(&self) -> DVec3 {
        self.drift
    }

    /// Advance one physics tick; true if a new fix became available.
    pub fn update(&mut self, tick: u64, time: f64, kin: &BodyKinematics, env: &SensorEnv) -> bool {
        if self.timing.is_due(tick) {
            let c = &self.config;
            let r = kin.attitude * c.mount.position;
            let antenna = kin.position + r;
            let velocity = kin.velocity + kin.attitude * kin.rates.cross(c.mount.position);
            let d = self.drift;
            self.drift = DVec3::new(
                self.horizontal.step(d.x, &mut self.rng),
                self.horizontal.step(d.y, &mut self.rng),
                self.vertical.step(d.z, &mut self.rng),
            );
            let n = self.rng.normal3();
            let white = DVec3::new(n.x * c.noise_horizontal, n.y * c.noise_horizontal, n.z * c.noise_vertical);
            let position = antenna + self.drift + white;
            let value = GpsFix {
                position,
                geodetic: env.geo.enu_to_geodetic(position),
                velocity: velocity + c.noise_velocity * self.rng.normal3(),
                eph: c.drift_horizontal.hypot(c.noise_horizontal),
                epv: c.drift_vertical.hypot(c.noise_vertical),
            };
            self.out.push(Stamped { tick, time, value });
        }
        self.out.poll(tick)
    }

    pub fn latest(&self) -> Option<&Stamped<GpsFix>> {
        self.out.latest()
    }
}
