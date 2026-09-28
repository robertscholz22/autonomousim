//! Controller configuration, the flight envelope and the control model derived from the
//! aircraft.
//!
//! At construction the controller trims the aircraft over its speed range (the trim table
//! gives the surface and throttle feedforward) and linearises it at several airspeeds across
//! the range ([`ControlDerivatives`]); at run time it interpolates that schedule in airspeed and
//! scales it by the density ratio.
//!
//! Rate loop, per pilot axis (roll right, pitch up, yaw right), in angular acceleration:
//!
//! ```text
//! ω̇ = a,   a = K_p·e + K_i∫e,   K_p = ω_b,   K_i = rate_integral·ω_b²
//! u = u_trim(V) + B(V)⁻¹·(J·a − D(V)·ω − k_α·M_α(V)·α̂),   α̂̇ = q_ref − L_α(V)/(m·V)·α̂
//! ```
//!
//! with the scheduled control matrix `B`, damping `D`, pitch stiffness `M_α` and lift slope
//! `L_α` (α̂ is the angle of attack the commanded pitching builds, `q_ref` the pitch-rate
//! setpoint through the loop's own first-order response, not the gusts; `k_α` is
//! `pitch_stiffness`), so the closed loop is nearly
//! first order with bandwidth `ω_b = min(rate_bandwidth_max, rate_servo_ratio/τ_servo)`. The
//! attitude loop is `attitude_ratio` times slower, TECS and the guidance slower again.

use crate::ControlError;
use autonomousim_vehicles::fixedwing::{ControlDerivatives, FixedWing, FixedWingDef};
use glam::{DMat3, DVec3};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Everything configurable about [`FixedWingController`](super::FixedWingController).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FixedWingConfig {
    pub tuning: FixedWingTuning,
    pub limits: FixedWingLimits,
    /// Air density of the design point (kg/m³).
    pub design_density: f64,
    /// Design airspeed (m/s); default: the middle of the trimmable speed range.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub design_airspeed: Option<f64>,
    /// Gravity (m/s²).
    pub gravity: f64,
}

impl Default for FixedWingConfig {
    fn default() -> Self {
        Self {
            tuning: FixedWingTuning::default(),
            limits: FixedWingLimits::default(),
            design_density: 1.225,
            design_airspeed: None,
            gravity: 9.80665,
        }
    }
}

/// Loop gains as bandwidths and ratios; see the module documentation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FixedWingTuning {
    /// Upper bound of the roll and pitch rate-loop bandwidth (rad/s).
    pub rate_bandwidth_max: f64,
    /// Largest rate-loop bandwidth relative to the servo corner frequency `1/τ`.
    pub rate_servo_ratio: f64,
    /// Rate-integrator zero as a fraction of the bandwidth.
    pub rate_integral: f64,
    /// Rate-integrator limit as a fraction of each axis' control power (angular acceleration at
    /// full deflection).
    pub rate_i_limit: f64,
    /// Yaw-rate bandwidth relative to roll and pitch.
    pub yaw_ratio: f64,
    /// Yaw-rate setpoint per radian of sideslip (1/s): the rudder turns the nose into the
    /// relative wind.
    pub sideslip_gain: f64,
    /// Share of the pitch stiffness `M_α·α̂` the rate loop cancels (0–1): more tracks pitch
    /// rates without a standing error, too much leaves the short period weakly stable where the
    /// linear model overestimates the stiffness.
    pub pitch_stiffness: f64,
    /// Rate bandwidth / attitude gain.
    pub attitude_ratio: f64,
    /// Airspeed error → acceleration demand (1/s).
    pub speed_gain: f64,
    /// Altitude error → climb-rate demand (1/s).
    pub altitude_gain: f64,
    /// Course error → course-rate demand (1/s).
    pub course_gain: f64,
    /// TECS: total energy rate error → thrust over weight (proportional, and integral in 1/s).
    pub energy_p: f64,
    pub energy_i: f64,
    /// TECS: energy balance rate error → pitch (proportional, and integral in 1/s).
    pub balance_p: f64,
    pub balance_i: f64,
    /// Cutoff of the airspeed-derivative filter (rad/s).
    pub speed_rate_cutoff: f64,
    /// L1 guidance: period (s) and damping of the path convergence (PX4 `FW_L1_PERIOD`,
    /// `FW_L1_DAMPING`).
    pub l1_period: f64,
    pub l1_damping: f64,
}

impl Default for FixedWingTuning {
    fn default() -> Self {
        Self {
            rate_bandwidth_max: 10.0,
            rate_servo_ratio: 0.25,
            rate_integral: 0.2,
            rate_i_limit: 0.3,
            yaw_ratio: 0.5,
            sideslip_gain: 2.0,
            pitch_stiffness: 1.0,
            attitude_ratio: 3.0,
            speed_gain: 0.3,
            altitude_gain: 0.4,
            course_gain: 0.5,
            energy_p: 1.0,
            energy_i: 0.3,
            balance_p: 1.0,
            balance_i: 0.2,
            speed_rate_cutoff: 2.0,
            l1_period: 20.0,
            l1_damping: 0.75,
        }
    }
}

/// Setpoint limits inside the controller.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FixedWingLimits {
    /// Body-rate setpoints: roll, pitch, yaw (rad/s).
    pub rate: DVec3,
    /// Bank angle (rad).
    pub roll: f64,
    /// Pitch attitude range (rad).
    pub pitch_min: f64,
    pub pitch_max: f64,
    /// Acceleration demand of the speed loop (m/s²).
    pub accel: f64,
}

impl Default for FixedWingLimits {
    fn default() -> Self {
        let deg = std::f64::consts::PI / 180.0;
        Self {
            rate: DVec3::new(90.0, 45.0, 45.0) * deg,
            roll: 50.0 * deg,
            pitch_min: -25.0 * deg,
            pitch_max: 25.0 * deg,
            accel: 2.0,
        }
    }
}

/// One row of the trim table.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrimPoint {
    pub airspeed: f64,
    pub alpha: f64,
    /// Normalised aileron, elevator, rudder and throttle.
    pub controls: [f64; 4],
}

/// What the controller knows about the aircraft.
#[derive(Clone, Debug)]
pub struct FlightModel {
    pub def: Arc<FixedWingDef>,
    pub mass: f64,
    pub gravity: f64,
    /// Straight-flight stall speed at the design density (m/s).
    pub stall_speed: f64,
    /// Level trims at the design density, by increasing airspeed.
    pub trims: Vec<TrimPoint>,
    /// Highest level-flight airspeed at full throttle (m/s).
    pub max_speed: f64,
    /// Steepest steady climb at the design airspeed (m/s).
    pub max_climb: f64,
    pub design: ControlDerivatives,
    pub design_density: f64,
    /// Inertia in pilot axes (roll right, pitch up, yaw right).
    pub inertia: DMat3,
    /// Linear models at the design density, by increasing airspeed.
    pub schedule: Vec<LinearPoint>,
    /// Servo time constants of aileron, elevator, rudder (s).
    pub servo_tau: DVec3,
}

/// The control model at one airspeed, in pilot axes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinearPoint {
    pub airspeed: f64,
    /// Moment per normalised aileron, elevator, rudder (columns).
    pub control: DMat3,
    /// Moment per rate (columns).
    pub damping: DMat3,
    /// Moment per radian of angle of attack, and the lift per radian (N).
    pub alpha_moment: DVec3,
    pub lift_alpha: f64,
    /// Steady thrust (N) per unit of throttle.
    pub thrust_per_throttle: f64,
}

impl LinearPoint {
    fn of(d: &ControlDerivatives) -> Self {
        Self {
            airspeed: d.trim.airspeed,
            control: DMat3::from_cols(to_pilot(d.control[0]), to_pilot(d.control[1]), to_pilot(d.control[2])),
            damping: DMat3::from_cols(to_pilot(d.damping[0]), -to_pilot(d.damping[1]), -to_pilot(d.damping[2])),
            alpha_moment: to_pilot(d.alpha),
            lift_alpha: d.lift_alpha,
            thrust_per_throttle: d.thrust_per_throttle,
        }
    }

    fn lerp(&self, o: &Self, w: f64) -> Self {
        Self {
            airspeed: self.airspeed + w * (o.airspeed - self.airspeed),
            control: self.control + (o.control - self.control) * w,
            damping: self.damping + (o.damping - self.damping) * w,
            alpha_moment: self.alpha_moment.lerp(o.alpha_moment, w),
            lift_alpha: self.lift_alpha + w * (o.lift_alpha - self.lift_alpha),
            thrust_per_throttle: self.thrust_per_throttle + w * (o.thrust_per_throttle - self.thrust_per_throttle),
        }
    }

    fn scaled(mut self, k: f64) -> Self {
        self.control *= k;
        self.damping *= k;
        self.alpha_moment *= k;
        self.lift_alpha *= k;
        self.thrust_per_throttle *= k;
        self
    }
}

/// Body FLU ↔ pilot axes (roll right, pitch up, yaw right).
pub fn to_pilot(v: DVec3) -> DVec3 {
    DVec3::new(v.x, -v.y, -v.z)
}

impl FlightModel {
    pub fn new(def: &Arc<FixedWingDef>, config: &FixedWingConfig) -> Result<Self, ControlError> {
        let err = |m: String| ControlError::InvalidConfig(format!("{}: {m}", def.name));
        let (rho, g) = (config.design_density, config.gravity);
        if !(rho > 0.0 && g > 0.0) {
            return Err(err("design density and gravity must be positive".into()));
        }
        let a = FixedWing::new(def.clone(), 0.01);
        let stall_speed = def.stall_speed(rho, g);
        // Level trims from just above the stall until the throttle runs out.
        let mut trims = Vec::new();
        let mut v = 1.05 * stall_speed;
        let mut max_speed = 0.0;
        while v < 10.0 * stall_speed {
            match a.trim(v, rho, 0.0, 0.0, g) {
                Ok(t) => {
                    let c = t.controls;
                    trims.push(TrimPoint {
                        airspeed: v,
                        alpha: t.alpha,
                        controls: [c.aileron, c.elevator, c.rudder, c.throttle],
                    });
                    max_speed = v;
                }
                Err(_) if !trims.is_empty() => break,
                Err(_) => {}
            }
            v *= 1.03;
        }
        if trims.len() < 3 {
            return Err(err("does not trim in level flight over a speed range".into()));
        }
        let lo = trims[0].airspeed;
        let design_airspeed = config.design_airspeed.unwrap_or(0.5 * (lo + max_speed));
        let design = a.control_derivatives(design_airspeed, rho, g).map_err(|e| err(format!("design point: {e}")))?;
        // Steepest climb at the design airspeed (bisection on the flight-path angle).
        let (mut ok, mut bad) = (0.0, 0.5);
        for _ in 0..30 {
            let gamma = 0.5 * (ok + bad);
            if a.trim(design_airspeed, rho, gamma, 0.0, g).is_ok() {
                ok = gamma;
            } else {
                bad = gamma;
            }
        }
        let max_climb = design_airspeed * f64::sin(ok);
        let s = DMat3::from_diagonal(DVec3::new(1.0, -1.0, -1.0));
        let inertia = s * a.inertia() * s;
        // Linear models across the speed range.
        const POINTS: usize = 6;
        let mut schedule = Vec::with_capacity(POINTS);
        for k in 0..POINTS {
            let v = lo + (max_speed - lo) * k as f64 / (POINTS - 1) as f64;
            let d = a.control_derivatives(v, rho, g).map_err(|e| err(format!("linearising: {e}")))?;
            let p = LinearPoint::of(&d);
            if p.control.determinant().abs() < 1e-12 {
                return Err(err(format!("no independent control authority about all three axes at {v:.1} m/s")));
            }
            schedule.push(p);
        }
        let k = &def.controls;
        Ok(Self {
            def: def.clone(),
            mass: a.mass(),
            gravity: g,
            stall_speed,
            trims,
            max_speed,
            max_climb,
            design,
            design_density: rho,
            inertia,
            schedule,
            servo_tau: DVec3::new(k.aileron.tau, k.elevator.tau, k.rudder.tau),
        })
    }

    /// Level trim interpolated at `airspeed` (clamped to the table).
    pub fn trim_at(&self, airspeed: f64) -> TrimPoint {
        let t = &self.trims;
        let i = t.partition_point(|p| p.airspeed < airspeed).clamp(1, t.len() - 1);
        let (a, b) = (&t[i - 1], &t[i]);
        let w = ((airspeed - a.airspeed) / (b.airspeed - a.airspeed)).clamp(0.0, 1.0);
        let lerp = |x: f64, y: f64| x + w * (y - x);
        TrimPoint {
            airspeed,
            alpha: lerp(a.alpha, b.alpha),
            controls: [0, 1, 2, 3].map(|k| lerp(a.controls[k], b.controls[k])),
        }
    }

    /// Lowest and highest level-flight trim airspeeds (m/s).
    pub fn speed_range(&self) -> (f64, f64) {
        (self.trims[0].airspeed, self.max_speed)
    }

    /// Normal operating speeds (m/s): from 1.3 × the stall speed (or 1.1 × the lowest level
    /// trim, where the elevator runs out first) to 95 % of the top level speed.
    pub fn normal_speeds(&self) -> [f64; 2] {
        let (lo, hi) = self.speed_range();
        [(1.3 * self.stall_speed).max(1.1 * lo), 0.95 * hi]
    }

    /// Design airspeed (m/s).
    pub fn design_airspeed(&self) -> f64 {
        self.design.trim.airspeed
    }

    /// Rate-loop bandwidths of roll, pitch and yaw (rad/s).
    pub fn rate_bandwidth(&self, t: &FixedWingTuning) -> DVec3 {
        let b = |tau: f64| {
            if tau > 0.0 { t.rate_bandwidth_max.min(t.rate_servo_ratio / tau) } else { t.rate_bandwidth_max }
        };
        DVec3::new(b(self.servo_tau.x), b(self.servo_tau.y), t.yaw_ratio * b(self.servo_tau.z))
    }

    /// Linear model at `airspeed` (clamped to the schedule) in air of density `rho`.
    pub fn linear_at(&self, airspeed: f64, rho: f64) -> LinearPoint {
        let t = &self.schedule;
        let i = t.partition_point(|p| p.airspeed < airspeed).clamp(1, t.len() - 1);
        let (a, b) = (&t[i - 1], &t[i]);
        let w = ((airspeed - a.airspeed) / (b.airspeed - a.airspeed)).clamp(0.0, 1.0);
        a.lerp(b, w).scaled(rho / self.design_density)
    }

    /// Angular acceleration at full deflection about each pilot axis at the design point
    /// (rad/s²).
    pub fn control_power(&self) -> DVec3 {
        let j = self.inertia.inverse();
        let c = self.linear_at(self.design_airspeed(), self.design_density).control;
        DVec3::new((j * c.col(0)).x.abs(), (j * c.col(1)).y.abs(), (j * c.col(2)).z.abs())
    }
}
