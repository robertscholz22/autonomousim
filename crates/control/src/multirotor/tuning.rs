//! Controller configuration, and gains derived from the vehicle model.
//!
//! The rate loop is designed by pole placement on the linearised roll/pitch/yaw dynamics with
//! a first-order motor lag τ_m (the mean of the spin-up and spin-down time constants), in
//! angular-acceleration units:
//!
//! ```text
//! ω/α_cmd = 1/(s(τ_m·s + 1)),   α_cmd = K_p·e + K_i∫e − K_d·ω̇
//! τ_m·s³ + (1 + K_d)·s² + K_p·s + K_i = τ_m·(s² + 2ζω_n·s + ω_n²)(s + p)
//! ```
//!
//! The natural frequency is `ω_n = min(rate_omega_max, rate_lead/τ_m, rate_delay/dt)`: the
//! derivative term may lead the motor corner frequency by `rate_lead`, and the control delay of
//! one step costs at most `rate_delay` rad of phase. Every outer loop is a fixed ratio slower
//! than the one inside it, so one [`Tuning`] fits vehicles of any size.

use super::position::PositionGains;
use super::rate::RateGains;
use crate::ControlError;
use autonomousim_vehicles::multirotor::MultirotorDef;
use glam::DVec3;
use serde::{Deserialize, Serialize};

/// Everything configurable about [`MultirotorController`](super::MultirotorController).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ControllerConfig {
    pub tuning: Tuning,
    pub limits: Limits,
    /// Let the collective rise to keep roll and pitch authority at low thrust (PX4 airmode).
    pub airmode: bool,
    /// Air density the controller assumes (kg/m³); default: the rotors' reference density.
    pub design_density: Option<f64>,
    /// Gravity (m/s²).
    pub gravity: f64,
}

impl Default for ControllerConfig {
    fn default() -> Self {
        Self {
            tuning: Tuning::default(),
            limits: Limits::default(),
            airmode: false,
            design_density: None,
            gravity: 9.80665,
        }
    }
}

/// Loop bandwidths as ratios; see the module documentation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Tuning {
    /// Upper bound of the roll/pitch rate-loop natural frequency (rad/s).
    pub rate_omega_max: f64,
    /// Largest ratio of the rate-loop natural frequency to the motor corner frequency `1/τ_m`.
    pub rate_lead: f64,
    /// Largest phase (rad) that the one-step control delay may cost at the natural frequency.
    pub rate_delay: f64,
    /// Rate-loop damping ratio.
    pub rate_zeta: f64,
    /// Rate-integrator pole as a fraction of the natural frequency.
    pub rate_integral: f64,
    /// Rate-integrator limit as a fraction of each axis' torque authority.
    pub rate_i_limit: f64,
    /// Yaw-rate natural frequency relative to roll and pitch (yaw authority is small).
    pub yaw_ratio: f64,
    /// Cutoff of the derivative low-pass relative to the natural frequency (0: unfiltered).
    pub d_cutoff_ratio: f64,
    /// Rate natural frequency / attitude gain.
    pub attitude_ratio: f64,
    /// Share of the yaw error the attitude loop corrects together with the tilt (PX4
    /// `MC_YAW_WEIGHT`).
    pub yaw_weight: f64,
    /// Attitude gain / velocity gain.
    pub velocity_ratio: f64,
    /// Disturbance-observer bandwidth / velocity gain. Higher rejects gusts faster but passes
    /// more velocity noise: the observer differentiates the velocity.
    pub disturbance_ratio: f64,
    /// Velocity gain / position gain.
    pub position_ratio: f64,
    /// Natural frequency of the velocity reference / velocity gain (0: no setpoint shaping).
    pub reference_ratio: f64,
    /// Rate of the velocity and position loops (Hz).
    pub outer_rate: f64,
}

impl Default for Tuning {
    fn default() -> Self {
        Self {
            rate_omega_max: 60.0,
            rate_lead: 3.5,
            rate_delay: 0.15,
            rate_zeta: 0.7,
            rate_integral: 0.05,
            rate_i_limit: 0.3,
            yaw_ratio: 0.5,
            d_cutoff_ratio: 0.0,
            attitude_ratio: 3.5,
            yaw_weight: 0.4,
            velocity_ratio: 3.0,
            disturbance_ratio: 1.0,
            position_ratio: 3.0,
            reference_ratio: 0.5,
            outer_rate: 100.0,
        }
    }
}

/// Setpoint limits of the cascade.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Limits {
    /// Body-rate setpoints of the attitude loop (rad/s).
    pub rate: DVec3,
    /// Tilt in the velocity and position modes (rad).
    pub tilt: f64,
    /// Horizontal speed, climb and descent rate (m/s).
    pub vel_xy: f64,
    pub vel_up: f64,
    pub vel_down: f64,
    /// Smallest collective in the velocity and position modes, as a fraction of the maximum.
    pub thrust_min: f64,
    /// Horizontal thrust kept when the vertical demand saturates, as a fraction of the maximum.
    pub xy_margin: f64,
    /// Acceleration limits of the shaped velocity reference in the velocity and position modes
    /// (m/s²; see [`PositionController::shape`](super::PositionController::shape)).
    pub accel_xy: f64,
    pub accel_up: f64,
    pub accel_down: f64,
}

impl Default for Limits {
    fn default() -> Self {
        let deg = std::f64::consts::PI / 180.0;
        Self {
            rate: DVec3::new(220.0, 220.0, 200.0) * deg,
            tilt: 45.0 * deg,
            vel_xy: 12.0,
            vel_up: 3.0,
            vel_down: 1.5,
            thrust_min: 0.12,
            xy_margin: 0.3,
            accel_xy: 5.0,
            accel_up: 4.0,
            accel_down: 3.0,
        }
    }
}

/// Gains of all loops for one vehicle and control period.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gains {
    pub rate: RateGains,
    /// Attitude error → body rate (1/s), before the yaw-weight compensation.
    pub attitude: DVec3,
    pub position: PositionGains,
    /// The velocity and position loops run every this many control steps.
    pub outer_every: u32,
    /// Motor time constant the design assumed (s).
    pub motor_tau: f64,
}

impl ControllerConfig {
    pub fn validate(&self) -> Result<(), ControlError> {
        let t = &self.tuning;
        let l = &self.limits;
        let pos = |x: f64| x > 0.0;
        let ok = [
            t.rate_omega_max,
            t.rate_lead,
            t.rate_delay,
            t.rate_zeta,
            t.attitude_ratio,
            t.velocity_ratio,
            t.position_ratio,
            t.outer_rate,
            t.yaw_ratio,
            t.disturbance_ratio,
            l.tilt,
            l.vel_xy,
            l.vel_up,
            l.vel_down,
            l.accel_xy,
            l.accel_up,
            l.accel_down,
            self.gravity,
        ]
        .into_iter()
        .all(pos)
            && pos(l.rate.min_element())
            && [t.rate_integral, t.rate_i_limit, t.d_cutoff_ratio, t.reference_ratio].into_iter().all(|x| x >= 0.0)
            && (0.0..=1.0).contains(&t.yaw_weight)
            && l.tilt < std::f64::consts::FRAC_PI_2
            && (0.0..1.0).contains(&l.thrust_min)
            && (0.0..1.0).contains(&l.xy_margin)
            && self.design_density.is_none_or(pos);
        if ok { Ok(()) } else { Err(ControlError::InvalidConfig(format!("{self:?}"))) }
    }
}

impl Gains {
    /// Gains for `def` controlled every `dt` seconds. `torque_authority` (N·m per axis) scales
    /// the rate-integrator limit.
    pub fn derive(def: &MultirotorDef, tuning: &Tuning, dt: f64, torque_authority: DVec3) -> Self {
        let t = tuning;
        let tau = 0.5 * (def.rotor.tau_up + def.rotor.tau_down);
        let omega = t.rate_omega_max.min(t.rate_lead / tau).min(t.rate_delay / dt);
        let omega = DVec3::new(omega, omega, omega * t.yaw_ratio);
        let axis = |w: f64| {
            let zeta = t.rate_zeta;
            let p = t.rate_integral * w;
            // Motors faster than the design needs: treat them as just fast enough (K_d = 0).
            let tau = tau.max(1.0 / (2.0 * zeta * w + p));
            (tau * (w * w + 2.0 * zeta * w * p), tau * w * w * p, tau * (2.0 * zeta * w + p) - 1.0)
        };
        let (px, ix, dx) = axis(omega.x);
        let (pz, iz, dz) = axis(omega.z);
        let inertia = def.body.inertia;
        let rate = RateGains {
            p: DVec3::new(px, px, pz),
            i: DVec3::new(ix, ix, iz),
            d: DVec3::new(dx, dx, dz).max(DVec3::ZERO),
            i_limit: t.rate_i_limit * torque_authority / inertia,
            d_cutoff: t.d_cutoff_ratio * omega.x,
        };
        let attitude = omega / t.attitude_ratio;
        let vel = attitude.x / t.velocity_ratio;
        let position = PositionGains {
            pos_p: DVec3::splat(vel / t.position_ratio),
            vel_p: DVec3::splat(vel),
            disturbance: t.disturbance_ratio * vel,
            reference: t.reference_ratio * vel,
        };
        let outer_every = ((1.0 / (t.outer_rate * dt)).round() as u32).max(1);
        Self { rate, attitude, position, outer_every, motor_tau: tau }
    }
}
