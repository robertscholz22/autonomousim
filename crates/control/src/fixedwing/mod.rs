//! Fixed-wing aircraft: normalised action modes and the controller between setpoints and the
//! aircraft's [`FixedWingInput`].
//!
//! | Mode | Components | Meaning at ±1 |
//! |---|---|---|
//! | `surfaces` | aileron, elevator, rudder, throttle | full deflection (+ roll right, pitch up, yaw right); throttle −1 → 0 (idle), +1 → full |
//! | `rates` | roll, pitch, yaw rate, throttle | ± the rate limits (+ roll right, pitch up, yaw right; the turn's yaw rate is added); throttle as in `surfaces` |
//! | `attitude` (default) | bank, pitch, airspeed | ± the bank and pitch limits (+ right, nose up); airspeed across its range, 0 the middle |
//! | `guidance` | course rate, climb rate, airspeed | ± the limits (+ counter-clockwise, i.e. turning left, as the ENU yaw; + climbing); airspeed as in `attitude` |
//!
//! Flaps stay up and the brakes released; scripted agents set them through a
//! [`FixedWingSetpoint::Surfaces`] command. Components outside `[−1, 1]` are clipped and
//! non-finite ones read as 0.
//!
//! The cascade (after PX4's fixed-wing controllers): the rate loop and its gains derived from
//! the aircraft ([`tuning`]), an attitude loop with turn coordination, and TECS and L1
//! guidance ([`guidance`]).

pub mod guidance;
pub mod tuning;

use crate::ControlError;
use autonomousim_core::math::quat::wrap_angle;
use autonomousim_vehicles::fixedwing::{FixedWing, FixedWingDef, FixedWingInput};
use glam::{DVec2, DVec3, EulerRot};
pub use guidance::{Path, Tecs};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;
use std::sync::Arc;
pub use tuning::{FixedWingConfig, FixedWingLimits, FixedWingTuning, FlightModel, LinearPoint, TrimPoint, to_pilot};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FixedWingActionMode {
    /// Control surfaces and throttle directly.
    Surfaces,
    /// Body rates and throttle.
    Rates,
    /// Bank, pitch and airspeed; TECS drives the throttle.
    #[default]
    Attitude,
    /// Course rate, climb rate and airspeed.
    Guidance,
}

impl FixedWingActionMode {
    pub const ALL: [FixedWingActionMode; 4] = [
        FixedWingActionMode::Surfaces,
        FixedWingActionMode::Rates,
        FixedWingActionMode::Attitude,
        FixedWingActionMode::Guidance,
    ];

    pub fn name(self) -> &'static str {
        match self {
            FixedWingActionMode::Surfaces => "surfaces",
            FixedWingActionMode::Rates => "rates",
            FixedWingActionMode::Attitude => "attitude",
            FixedWingActionMode::Guidance => "guidance",
        }
    }

    /// Action length.
    pub fn dim(self) -> usize {
        match self {
            FixedWingActionMode::Surfaces | FixedWingActionMode::Rates => 4,
            FixedWingActionMode::Attitude | FixedWingActionMode::Guidance => 3,
        }
    }
}

impl fmt::Display for FixedWingActionMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for FixedWingActionMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::ALL.into_iter().find(|m| m.name() == s).ok_or_else(|| format!("unknown fixed-wing action mode {s:?}"))
    }
}

/// Lateral part of a guidance setpoint.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Lateral {
    /// Course rate (rad/s, counter-clockwise).
    CourseRate(f64),
    /// Course over ground (rad, counter-clockwise from east).
    Course(f64),
    /// Follow a path with L1 guidance.
    Path(Path),
}

/// Vertical part of a guidance setpoint.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Vertical {
    /// Climb rate (m/s).
    ClimbRate(f64),
    /// Altitude (world z, m).
    Altitude(f64),
}

/// What a fixed-wing controller tracks. Angles and rates in pilot axes: + roll right, pitch
/// up, yaw right.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FixedWingSetpoint {
    /// Normalised surfaces, throttle, flaps and brakes, passed through.
    Surfaces(FixedWingInput),
    /// Body rates (rad/s) on top of the turn coordination, and throttle (0–1).
    Rates { rates: DVec3, throttle: f64 },
    /// Bank and pitch (rad) and airspeed (m/s).
    Attitude { roll: f64, pitch: f64, airspeed: f64 },
    /// Guidance with airspeed (m/s).
    Guidance { lateral: Lateral, vertical: Vertical, airspeed: f64 },
}

/// Bank, pitch and heading (rad; + right, nose up, counter-clockwise) of a body → world
/// rotation.
pub fn euler(rot: glam::DQuat) -> (f64, f64, f64) {
    let (yaw, neg_pitch, roll) = rot.to_euler(EulerRot::ZYX);
    (roll, -neg_pitch, yaw)
}

/// Setpoint → aircraft input.
#[derive(Clone, Debug)]
pub struct FixedWingController {
    model: FlightModel,
    config: FixedWingConfig,
    dt: f64,
    kp: DVec3,
    ki: DVec3,
    i_limit: DVec3,
    integral: DVec3,
    /// Reference pitch rate (rad/s) and the angle of attack it builds (rad), see
    /// [`rate_loop`](Self::rate_loop).
    pitch_ref: f64,
    alpha_pitch: f64,
    tecs: Tecs,
    /// Last rate setpoint (pilot axes) and bank / pitch setpoints, for diagnostics.
    rate_sp: DVec3,
    attitude_sp: (f64, f64),
}

impl FixedWingController {
    /// Controller for `def` running every `dt` seconds.
    pub fn new(def: &Arc<FixedWingDef>, dt: f64, config: &FixedWingConfig) -> Result<Self, ControlError> {
        if !(dt > 0.0 && dt.is_finite()) {
            return Err(ControlError::InvalidConfig("controller period must be positive".into()));
        }
        let model = FlightModel::new(def, config)?;
        let t = &config.tuning;
        let kp = model.rate_bandwidth(t);
        let ki = kp * kp * t.rate_integral;
        let i_limit = model.control_power() * t.rate_i_limit;
        Ok(Self {
            model,
            config: config.clone(),
            dt,
            kp,
            ki,
            i_limit,
            integral: DVec3::ZERO,
            pitch_ref: 0.0,
            alpha_pitch: 0.0,
            tecs: Tecs::default(),
            rate_sp: DVec3::ZERO,
            attitude_sp: (0.0, 0.0),
        })
    }

    pub fn model(&self) -> &FlightModel {
        &self.model
    }

    pub fn config(&self) -> &FixedWingConfig {
        &self.config
    }

    /// Rate-loop proportional gains (rad/s per rad/s error → rad/s²), pilot axes.
    pub fn rate_gains(&self) -> DVec3 {
        self.kp
    }

    /// Last rate setpoint (pilot axes, rad/s) and bank/pitch setpoints (rad).
    pub fn rate_setpoint(&self) -> DVec3 {
        self.rate_sp
    }

    pub fn attitude_setpoint(&self) -> (f64, f64) {
        self.attitude_sp
    }

    /// Clear the internal state.
    pub fn reset(&mut self) {
        self.integral = DVec3::ZERO;
        self.pitch_ref = 0.0;
        self.alpha_pitch = 0.0;
        self.tecs.reset();
        self.rate_sp = DVec3::ZERO;
        self.attitude_sp = (0.0, 0.0);
    }

    /// Input for this step.
    pub fn update(&mut self, setpoint: &FixedWingSetpoint, a: &FixedWing) -> FixedWingInput {
        let lim = &self.config.limits;
        let t = &self.config.tuning;
        let g = self.model.gravity;
        let v = a.flow().airspeed.max(0.8 * self.model.stall_speed);
        let rho = self.density(a);
        let (roll, pitch, _) = euler(a.orientation());
        let trim = self.model.trim_at(v);
        let lin = self.model.linear_at(v, rho);
        let tw_per_throttle = lin.thrust_per_throttle / (self.model.mass * g);
        let pitch_range = (lim.pitch_min, lim.pitch_max);
        let base = guidance::TecsInput {
            dt: self.dt,
            airspeed: a.flow().airspeed,
            climb: a.lin_vel_world().z,
            airspeed_sp: v,
            climb_sp: None,
            gravity: g,
            speed_gain: t.speed_gain,
            accel_limit: lim.accel,
            cutoff: t.speed_rate_cutoff,
            energy_p: t.energy_p,
            energy_i: t.energy_i,
            balance_p: t.balance_p,
            balance_i: t.balance_i,
            tw_per_throttle,
            trim_throttle: trim.controls[3],
            trim_alpha: trim.alpha,
            pitch_range,
            roll,
        };
        let (rates, throttle) = match *setpoint {
            FixedWingSetpoint::Surfaces(input) => return input.clamped(),
            FixedWingSetpoint::Rates { rates, throttle } => {
                (rates + self.coordination(a, roll, pitch, v, 0.0, 0.0), throttle)
            }
            FixedWingSetpoint::Attitude { roll: roll_sp, pitch: pitch_sp, airspeed } => {
                let (throttle, _) = self.tecs.update(&guidance::TecsInput { airspeed_sp: airspeed, ..base });
                (self.attitude(a, roll_sp, pitch_sp, roll, pitch, v), throttle)
            }
            FixedWingSetpoint::Guidance { lateral, vertical, airspeed } => {
                let vg = a.lin_vel_world().truncate();
                let ground_speed = vg.length().max(0.5 * v);
                let course = vg.y.atan2(vg.x);
                let turn = match lateral {
                    // Course rate → bank of the coordinated turn (a left turn banks left).
                    Lateral::CourseRate(rate) => rate * ground_speed,
                    Lateral::Course(c) => {
                        let rate = t.course_gain * wrap_angle(c - course);
                        rate * ground_speed
                    }
                    Lateral::Path(path) => guidance::l1_acceleration(
                        &path,
                        a.position().truncate(),
                        if vg.length() > 0.5 { vg } else { DVec2::from_angle(euler(a.orientation()).2) * v },
                        t.l1_period,
                        t.l1_damping,
                    ),
                };
                let roll_sp = (-turn / g).atan();
                let climb_sp = match vertical {
                    Vertical::ClimbRate(c) => c,
                    Vertical::Altitude(h) => t.altitude_gain * (h - a.position().z),
                }
                .clamp(-self.model.max_climb, self.model.max_climb);
                let (throttle, pitch_sp) =
                    self.tecs.update(&guidance::TecsInput { airspeed_sp: airspeed, climb_sp: Some(climb_sp), ..base });
                (self.attitude(a, roll_sp, pitch_sp.unwrap_or(0.0), roll, pitch, v), throttle)
            }
        };
        let (controls, throttle) = self.rate_loop(a, rates, v, &trim, throttle);
        let [aileron, elevator, rudder] = controls;
        FixedWingInput { aileron, elevator, rudder, throttle, ..FixedWingInput::default() }.clamped()
    }

    /// Air density; the design density before the aircraft has seen any air.
    fn density(&self, a: &FixedWing) -> f64 {
        let rho = a.flow().density;
        if rho > 0.0 { rho } else { self.model.design_density }
    }

    /// Body-rate setpoint (pilot axes) of the attitude loop.
    fn attitude(&mut self, a: &FixedWing, roll_sp: f64, pitch_sp: f64, roll: f64, pitch: f64, v: f64) -> DVec3 {
        let lim = &self.config.limits;
        let roll_sp = roll_sp.clamp(-lim.roll, lim.roll);
        let pitch_sp = pitch_sp.clamp(lim.pitch_min, lim.pitch_max);
        self.attitude_sp = (roll_sp, pitch_sp);
        let k = self.kp / self.config.tuning.attitude_ratio;
        let roll_rate = k.x * wrap_angle(roll_sp - roll);
        let pitch_rate = k.y * (pitch_sp - pitch);
        self.coordination(a, roll, pitch, v, roll_rate, pitch_rate)
    }

    /// Body rates (pilot axes) for Euler rates `roll_rate`, `pitch_rate` and the coordinated
    /// turn's heading rate `g·tan φ / V`, plus the sideslip correction on yaw.
    fn coordination(&self, a: &FixedWing, roll: f64, pitch: f64, v: f64, roll_rate: f64, pitch_rate: f64) -> DVec3 {
        let g = self.model.gravity;
        let roll_c = roll.clamp(-1.3, 1.3);
        let heading_rate = g * roll_c.tan() / v;
        let (sr, cr) = roll.sin_cos();
        let (sp, cp) = pitch.sin_cos();
        let beta = a.flow().beta;
        DVec3::new(
            roll_rate - sp * heading_rate,
            cr * pitch_rate + sr * cp * heading_rate,
            -sr * pitch_rate + cr * cp * heading_rate + self.config.tuning.sideslip_gain * beta,
        )
    }

    /// Surface commands for a pilot-axis rate setpoint.
    fn rate_loop(&mut self, a: &FixedWing, rate_sp: DVec3, v: f64, trim: &TrimPoint, throttle: f64) -> ([f64; 3], f64) {
        let rate_sp = rate_sp.clamp(-self.config.limits.rate, self.config.limits.rate);
        self.rate_sp = rate_sp;
        let m = &self.model;
        let rates = to_pilot(a.ang_vel_body());
        let err = rate_sp - rates;
        let accel = self.kp * err + self.integral;
        let lin = m.linear_at(v, self.density(a));
        // The angle of attack the commanded pitch rate builds, held against (a share of) the
        // pitch stiffness: q_ref follows the setpoint as the rate loop does (bandwidth K_p),
        // α̂̇ = q_ref − L_α/(m·V)·α̂. A feedforward, so gusts still meet the aircraft's own
        // stability.
        self.pitch_ref += (rate_sp.y - self.pitch_ref) * (1.0 - (-self.kp.y * self.dt).exp());
        let lift_rate = lin.lift_alpha / (m.mass * v);
        self.alpha_pitch += (self.pitch_ref - lift_rate * self.alpha_pitch) * self.dt;
        if !(self.alpha_pitch.is_finite() && self.pitch_ref.is_finite()) {
            (self.pitch_ref, self.alpha_pitch) = (0.0, 0.0);
        }
        let moment = m.inertia * accel
            - lin.damping * rates
            - lin.alpha_moment * self.alpha_pitch * self.config.tuning.pitch_stiffness;
        let du = lin.control.inverse() * moment;
        let u = [trim.controls[0] + du.x, trim.controls[1] + du.y, trim.controls[2] + du.z];
        // Integrate unless the surface is saturated in the same direction.
        let e = err.to_array();
        let mut next = self.integral.to_array();
        let (ki, lim) = (self.ki.to_array(), self.i_limit.to_array());
        for k in 0..3 {
            let d = ki[k] * e[k] * self.dt;
            if (u[k] >= 1.0 && d > 0.0) || (u[k] <= -1.0 && d < 0.0) {
                continue;
            }
            next[k] = (next[k] + d).clamp(-lim[k], lim[k]);
        }
        let next = DVec3::from_array(next);
        if next.is_finite() {
            self.integral = next;
        }
        (u, throttle)
    }
}

/// Setpoint ranges of the normalised action modes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FixedWingActionLimits {
    /// Rates at ±1 in `rates` (rad/s: roll, pitch, yaw).
    pub rate: DVec3,
    /// Bank and pitch at ±1 in `attitude` (rad).
    pub roll: f64,
    pub pitch: f64,
    /// Course rate at ±1 in `guidance` (rad/s).
    pub course_rate: f64,
    /// Climb rate at ±1 in `guidance` (m/s); default: 70 % of the steepest steady climb.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub climb_rate: Option<f64>,
    /// Airspeed at −1 and +1 (m/s); default: [`FlightModel::normal_speeds`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub airspeed: Option<[f64; 2]>,
}

impl Default for FixedWingActionLimits {
    fn default() -> Self {
        let deg = std::f64::consts::PI / 180.0;
        Self {
            rate: DVec3::new(90.0, 30.0, 20.0) * deg,
            roll: 45.0 * deg,
            pitch: 20.0 * deg,
            course_rate: 0.3,
            climb_rate: None,
            airspeed: None,
        }
    }
}

/// Maps normalised actions of one [`FixedWingActionMode`] to setpoints.
#[derive(Clone, Debug)]
pub struct FixedWingActionMap {
    mode: FixedWingActionMode,
    limits: FixedWingActionLimits,
    climb_rate: f64,
    airspeed: [f64; 2],
}

impl FixedWingActionMap {
    /// Map for `mode`, with the defaults of `limits` filled in from the controller's model.
    pub fn new(
        mode: FixedWingActionMode,
        limits: &FixedWingActionLimits,
        controller: &FixedWingController,
    ) -> Result<Self, ControlError> {
        let m = controller.model();
        let airspeed = limits.airspeed.unwrap_or_else(|| m.normal_speeds());
        let climb_rate = limits.climb_rate.unwrap_or(0.7 * m.max_climb);
        if !(airspeed[0] > 0.0 && airspeed[1] >= airspeed[0] && climb_rate >= 0.0) {
            return Err(ControlError::InvalidConfig(format!(
                "fixed-wing action limits: airspeed range {airspeed:?}, climb rate {climb_rate}"
            )));
        }
        Ok(Self { mode, limits: limits.clone(), climb_rate, airspeed })
    }

    pub fn mode(&self) -> FixedWingActionMode {
        self.mode
    }

    pub fn dim(&self) -> usize {
        self.mode.dim()
    }

    /// Airspeed range of the `attitude` and `guidance` modes (m/s).
    pub fn airspeed_range(&self) -> [f64; 2] {
        self.airspeed
    }

    /// Climb rate at ±1 in `guidance` (m/s).
    pub fn climb_rate(&self) -> f64 {
        self.climb_rate
    }

    /// Setpoint for `action` (length [`dim`](Self::dim), already in `[−1, 1]`).
    pub fn setpoint(&self, action: &[f64]) -> FixedWingSetpoint {
        let l = &self.limits;
        let speed = |x: f64| self.airspeed[0] + 0.5 * (x + 1.0) * (self.airspeed[1] - self.airspeed[0]);
        match self.mode {
            FixedWingActionMode::Surfaces => FixedWingSetpoint::Surfaces(FixedWingInput {
                aileron: action[0],
                elevator: action[1],
                rudder: action[2],
                throttle: 0.5 * (action[3] + 1.0),
                ..FixedWingInput::default()
            }),
            FixedWingActionMode::Rates => FixedWingSetpoint::Rates {
                rates: DVec3::new(action[0], action[1], action[2]) * l.rate,
                throttle: 0.5 * (action[3] + 1.0),
            },
            FixedWingActionMode::Attitude => FixedWingSetpoint::Attitude {
                roll: action[0] * l.roll,
                pitch: action[1] * l.pitch,
                airspeed: speed(action[2]),
            },
            FixedWingActionMode::Guidance => FixedWingSetpoint::Guidance {
                lateral: Lateral::CourseRate(action[0] * l.course_rate),
                vertical: Vertical::ClimbRate(action[1] * self.climb_rate),
                airspeed: speed(action[2]),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use autonomousim_vehicles::presets;

    fn controller(name: &str) -> FixedWingController {
        let def = Arc::new(presets::fixed_wing(name).unwrap());
        FixedWingController::new(&def, 0.002, &FixedWingConfig::default()).unwrap()
    }

    #[test]
    fn modes_and_ranges() {
        let c = controller("aerosonde_like");
        let m = FixedWingActionMap::new("surfaces".parse().unwrap(), &FixedWingActionLimits::default(), &c).unwrap();
        assert_eq!(m.dim(), 4);
        let FixedWingSetpoint::Surfaces(i) = m.setpoint(&[0.5, -1.0, 0.25, -1.0]) else { panic!() };
        assert_eq!((i.aileron, i.elevator, i.rudder, i.throttle, i.flap, i.brake), (0.5, -1.0, 0.25, 0.0, 0.0, 0.0));
        assert!("raw".parse::<FixedWingActionMode>().is_err());
        assert_eq!(FixedWingActionMode::default(), FixedWingActionMode::Attitude);
        let m = FixedWingActionMap::new(FixedWingActionMode::Attitude, &FixedWingActionLimits::default(), &c).unwrap();
        let [lo, hi] = m.airspeed_range();
        let model = c.model();
        assert!(lo > model.stall_speed && hi < model.max_speed && lo < hi, "{lo} {hi}");
        let FixedWingSetpoint::Attitude { roll, pitch, airspeed } = m.setpoint(&[1.0, -1.0, 0.0]) else { panic!() };
        assert!((roll - 45f64.to_radians()).abs() < 1e-12 && pitch < 0.0 && (airspeed - 0.5 * (lo + hi)).abs() < 1e-9);
        let m = FixedWingActionMap::new(FixedWingActionMode::Guidance, &FixedWingActionLimits::default(), &c).unwrap();
        assert!(m.climb_rate() > 0.5, "climb {}", m.climb_rate());
        assert_eq!(m.dim(), 3);
    }

    #[test]
    fn model_is_consistent() {
        for name in ["aerosonde_like", "c172_like"] {
            let c = controller(name);
            let m = c.model();
            // Aileron rolls right, elevator pitches up, rudder yaws right.
            for p in &m.schedule {
                let d = p.control;
                assert!(d.col(0).x > 0.0 && d.col(1).y > 0.0 && d.col(2).z > 0.0, "{name}: {d}");
                // Rate damping opposes the rates; pitch stiffness is stabilising.
                assert!(p.damping.col(0).x < 0.0 && p.damping.col(1).y < 0.0 && p.damping.col(2).z < 0.0, "{name}");
                assert!(p.alpha_moment.y < 0.0 && p.lift_alpha > 0.0 && p.thrust_per_throttle > 0.0, "{name}");
            }
            assert!(m.design.thrust_per_throttle > 0.0 && m.max_climb > 0.5, "{name}");
            let t = m.trim_at(m.design_airspeed());
            assert!((t.controls[1] - m.design.trim.controls.elevator).abs() < 0.02, "{name}");
            let kp = c.rate_gains();
            assert!(kp.x > 1.0 && kp.y > 1.0 && kp.z > 0.5, "{name}: {kp}");
        }
    }
}
