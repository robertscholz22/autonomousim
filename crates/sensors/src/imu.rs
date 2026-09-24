//! Inertial measurement unit: 3-axis accelerometer and gyroscope.
//!
//! Each axis reads `y = (1 + s)·x + b₀ + b(t) + n`, clipped to the range and quantised, with
//!
//! - `x`: the true value in the sensor frame, averaged over the sample interval (as an IMU with
//!   an anti-aliasing filter or delta-velocity/delta-angle integration does). The accelerometer
//!   sees the specific force at its mount point, `f + α×r + ω×(ω×r)`;
//! - `s`, `b₀`: per-episode scale-factor error and turn-on bias;
//! - `b(t)`: bias drift, a random walk or first-order Gauss–Markov process;
//! - `n`: white noise from a spectral density.
//!
//! Parameter names follow the Allan-variance convention (noise density N, bias random walk K),
//! as in Kalibr and the RotorS/PX4 Gazebo IMU plugin.

use crate::latency::{DelayLine, Stamped};
use crate::noise::{GaussMarkov, quantize3, white_sigma};
use crate::{BodyKinematics, Mount, SensorEnv, SensorError, Timing};
use autonomousim_core::rng::{Seed, SimRng};
use autonomousim_core::time::Clock;
use glam::{DQuat, DVec3};
use serde::{Deserialize, Serialize};

/// Error model of one triad (accelerometer: m/s², gyroscope: rad/s).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct InertialNoise {
    /// White-noise density N (units/√Hz).
    pub noise_density: f64,
    /// Bias driving-noise density K (units/s/√Hz).
    pub bias_random_walk: f64,
    /// Bias correlation time (s); none: pure random walk.
    pub bias_tau: Option<f64>,
    /// Standard deviation of the per-episode constant bias (units).
    pub turn_on_bias: f64,
    /// Standard deviation of the per-episode scale-factor error (fraction).
    pub scale: f64,
    /// Measurement range ± (units; 0: unlimited).
    pub range: f64,
    /// Resolution (units per LSB; 0: continuous).
    pub resolution: f64,
}

impl Default for InertialNoise {
    fn default() -> Self {
        Self::IDEAL
    }
}

impl InertialNoise {
    pub const IDEAL: Self = Self {
        noise_density: 0.0,
        bias_random_walk: 0.0,
        bias_tau: None,
        turn_on_bias: 0.0,
        scale: 0.0,
        range: 0.0,
        resolution: 0.0,
    };
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ImuConfig {
    pub rate_hz: u32,
    /// Delay until a reading is available (s; whole physics steps).
    pub latency: f64,
    pub mount: Mount,
    pub accel: InertialNoise,
    pub gyro: InertialNoise,
}

impl Default for ImuConfig {
    fn default() -> Self {
        Self::adis16448()
    }
}

impl ImuConfig {
    /// Noise-free, unlimited IMU at 500 Hz.
    pub fn ideal() -> Self {
        Self {
            rate_hz: 500,
            latency: 0.0,
            mount: Mount::default(),
            accel: InertialNoise::IDEAL,
            gyro: InertialNoise::IDEAL,
        }
    }

    /// Analog Devices ADIS16448 as parameterised by the RotorS / PX4 Gazebo IMU plugin
    /// (noise densities and random walks doubled from the datasheet there), at 500 Hz.
    pub fn adis16448() -> Self {
        let deg = std::f64::consts::PI / 180.0;
        Self {
            rate_hz: 500,
            latency: 0.0,
            mount: Mount::default(),
            accel: InertialNoise {
                noise_density: 2.0 * 2.0e-3,
                bias_random_walk: 2.0 * 3.0e-3,
                bias_tau: Some(300.0),
                turn_on_bias: 20.0e-3 * 9.8,
                scale: 0.0,
                range: 18.0 * 9.80665,
                resolution: 0.0,
            },
            gyro: InertialNoise {
                noise_density: 2.0 * 35.0 / 3600.0 * deg,
                bias_random_walk: 2.0 * 4.0 / 3600.0 * deg,
                bias_tau: Some(1000.0),
                turn_on_bias: 0.5 * deg,
                scale: 0.0,
                range: 1000.0 * deg,
                resolution: 0.0,
            },
        }
    }

    pub fn validate(&self) -> Result<(), SensorError> {
        let ok = |n: &InertialNoise| {
            [n.noise_density, n.bias_random_walk, n.turn_on_bias, n.scale, n.range, n.resolution]
                .iter()
                .all(|x| x.is_finite() && *x >= 0.0)
                && n.bias_tau.is_none_or(|t| t > 0.0)
        };
        if ok(&self.accel) && ok(&self.gyro) && self.mount.position.is_finite() && self.mount.rotation.is_finite() {
            Ok(())
        } else {
            Err(SensorError::InvalidConfig(format!("{self:?}")))
        }
    }
}

/// Accelerometer (m/s²) and gyroscope (rad/s) reading in the sensor frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ImuReading {
    pub accel: DVec3,
    pub gyro: DVec3,
}

/// Run-time state of one triad.
#[derive(Clone, Debug)]
struct Triad {
    white: f64,
    drift: GaussMarkov,
    bias0: DVec3,
    bias: DVec3,
    scale: DVec3,
}

impl Triad {
    fn new(n: &InertialNoise, dt: f64) -> Self {
        Self {
            white: white_sigma(n.noise_density, dt),
            drift: GaussMarkov::from_density(n.bias_random_walk, n.bias_tau, dt),
            bias0: DVec3::ZERO,
            bias: DVec3::ZERO,
            scale: DVec3::ONE,
        }
    }

    fn reset(&mut self, n: &InertialNoise, rng: &mut SimRng) {
        self.bias0 = n.turn_on_bias * rng.normal3();
        self.scale = DVec3::ONE + n.scale * rng.normal3();
        self.bias = self.drift.sample_stationary3(rng);
    }

    fn measure(&mut self, x: DVec3, n: &InertialNoise, rng: &mut SimRng) -> DVec3 {
        self.bias = self.drift.step3(self.bias, rng);
        let mut y = self.scale * x + self.bias0 + self.bias;
        if self.white > 0.0 {
            y += self.white * rng.normal3();
        }
        if n.range > 0.0 {
            y = y.clamp(DVec3::splat(-n.range), DVec3::splat(n.range));
        }
        quantize3(y, n.resolution)
    }
}

#[derive(Clone, Debug)]
pub struct Imu {
    config: ImuConfig,
    timing: Timing,
    dt: f64,
    mount_inv: DQuat,
    accel: Triad,
    gyro: Triad,
    rng: SimRng,
    // Sums over the current sample interval.
    sum_accel: DVec3,
    sum_gyro: DVec3,
    count: u32,
    out: DelayLine<ImuReading>,
}

impl Imu {
    pub fn new(config: ImuConfig, clock: &Clock, seed: Seed) -> Result<Self, SensorError> {
        config.validate()?;
        let timing = Timing::new("imu", clock, config.rate_hz, config.latency)?;
        let dt = clock.dt() * f64::from(timing.divider);
        let mut imu = Self {
            timing,
            dt,
            mount_inv: config.mount.quat().inverse(),
            accel: Triad::new(&config.accel, dt),
            gyro: Triad::new(&config.gyro, dt),
            rng: seed.rng(),
            sum_accel: DVec3::ZERO,
            sum_gyro: DVec3::ZERO,
            count: 0,
            out: DelayLine::new(timing.latency),
            config,
        };
        imu.reset(seed);
        Ok(imu)
    }

    pub fn config(&self) -> &ImuConfig {
        &self.config
    }

    pub fn timing(&self) -> Timing {
        self.timing
    }

    /// Sample interval (s).
    pub fn sample_dt(&self) -> f64 {
        self.dt
    }

    /// New episode: fresh random stream and per-episode errors, empty buffers.
    pub fn reset(&mut self, seed: Seed) {
        self.rng = seed.rng();
        self.accel.reset(&self.config.accel, &mut self.rng);
        self.gyro.reset(&self.config.gyro, &mut self.rng);
        self.sum_accel = DVec3::ZERO;
        self.sum_gyro = DVec3::ZERO;
        self.count = 0;
        self.out.clear();
    }

    /// Current accelerometer and gyroscope biases (turn-on + drift, sensor frame).
    pub fn biases(&self) -> (DVec3, DVec3) {
        (self.accel.bias0 + self.accel.bias, self.gyro.bias0 + self.gyro.bias)
    }

    /// True reading at the mount point (sensor frame) for the current kinematics.
    pub fn ideal(&self, kin: &BodyKinematics) -> ImuReading {
        let r = self.config.mount.position;
        let (w, alpha) = (kin.rates, kin.ang_acc);
        let f = kin.specific_force + alpha.cross(r) + w.cross(w.cross(r));
        ImuReading { accel: self.mount_inv * f, gyro: self.mount_inv * w }
    }

    /// Advance one physics tick; true if a new reading became visible.
    pub fn update(&mut self, tick: u64, time: f64, kin: &BodyKinematics, _env: &SensorEnv) -> bool {
        let x = self.ideal(kin);
        self.sum_accel += x.accel;
        self.sum_gyro += x.gyro;
        self.count += 1;
        if self.timing.is_due(tick) {
            let k = 1.0 / f64::from(self.count);
            let (a, g) = (self.sum_accel * k, self.sum_gyro * k);
            let value = ImuReading {
                accel: self.accel.measure(a, &self.config.accel, &mut self.rng),
                gyro: self.gyro.measure(g, &self.config.gyro, &mut self.rng),
            };
            self.out.push(Stamped { tick, time, value });
            self.sum_accel = DVec3::ZERO;
            self.sum_gyro = DVec3::ZERO;
            self.count = 0;
        }
        self.out.poll(tick)
    }

    pub fn latest(&self) -> Option<&Stamped<ImuReading>> {
        self.out.latest()
    }
}
