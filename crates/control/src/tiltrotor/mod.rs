//! Tiltrotors: normalised action modes and the controller between setpoints and the
//! tiltrotor's [`TiltrotorInput`].
//!
//! | Mode | Components | Meaning at ±1 |
//! |---|---|---|
//! | `raw` | throttle per rotor, tilt per rotor, aileron, elevator, rudder | throttle 0 at −1 and 1 at +1; tilt at either end of the mounts' range (+1 forward); full surface travel (+ roll right, nose up, nose right) |
//! | `attitude` | roll, pitch, yaw rate, climb rate, airspeed | ±`roll` (+ right wing down), ±`pitch` (+ nose up), ±`yaw_rate` (+ counter-clockwise), ±`climb`; airspeed from 0 at −1 to `speed[0]` at +1 |
//! | `velocity` (default) | vx, vy, vz, yaw rate | heading-frame velocity (+ forward, left, up): vx from −`speed[1]` to +`speed[0]`, vy ±`speed[1]` (while rotor-borne), vz ±`speed[2]`; ±`yaw_rate` |
//!
//! Components outside `[−1, 1]` are clipped and non-finite ones read as 0. The mode names are
//! shared with other families; a group resolves them by its vehicle.
//!
//! The controller converts by schedule: the mounts tilt with the (reference) airspeed along a
//! [`TiltSchedule`] inside the conversion corridor, from rotors up in hover to rotors forward
//! above the stall speed. A velocity loop asks for a specific force; the total rotor thrust and
//! the pitch that give it (forward and up) are solved on the aircraft's own force model, so
//! the rotors hold height and pitch drives speed in hover, while in cruise the wing holds
//! height through pitch and the rotors drive speed; the roll tilts the force sideways (turns
//! are coordinated on the wing). The rate loop inverts the moment model incrementally and
//! allocates the moment and the total thrust over the rotor thrusts, differential tilt and
//! the surfaces with least effort, so the surfaces take over as dynamic pressure builds. Each
//! rotor's thrust is turned into a rotor speed and the throttle that drives the motor to it.

mod allocation;
mod controller;
mod schedule;

pub use controller::TiltrotorController;
pub use schedule::{SchedulePoint, TiltSchedule};

use crate::ControlError;
use autonomousim_vehicles::tiltrotor::{MAX_ROTORS, TiltrotorDef, TiltrotorInput};
use glam::DVec3;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TiltrotorActionMode {
    /// Throttles, tilts and surfaces.
    Raw,
    /// Roll, pitch, yaw rate, climb rate and airspeed.
    Attitude,
    /// Heading-frame velocity and yaw rate.
    #[default]
    Velocity,
}

impl TiltrotorActionMode {
    pub const ALL: [TiltrotorActionMode; 3] =
        [TiltrotorActionMode::Raw, TiltrotorActionMode::Attitude, TiltrotorActionMode::Velocity];

    pub fn name(self) -> &'static str {
        match self {
            TiltrotorActionMode::Raw => "raw",
            TiltrotorActionMode::Attitude => "attitude",
            TiltrotorActionMode::Velocity => "velocity",
        }
    }
}

impl fmt::Display for TiltrotorActionMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for TiltrotorActionMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::ALL.into_iter().find(|m| m.name() == s).ok_or_else(|| format!("unknown tiltrotor action mode {s:?}"))
    }
}

/// What a tiltrotor's controller tracks. Angles, rates and velocities in FLU/ENU.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TiltrotorSetpoint {
    /// Pilot inputs, passed through.
    Raw(TiltrotorInput),
    /// Roll (+ right wing down) and pitch (+ nose up) angles (rad), yaw rate (rad/s,
    /// counter-clockwise), climb rate and forward airspeed (m/s).
    Attitude { roll: f64, pitch: f64, yaw_rate: f64, climb: f64, airspeed: f64 },
    /// Velocity over ground in the heading frame (m/s; forward, left, up) and yaw rate.
    Velocity { velocity: DVec3, yaw_rate: f64 },
    /// Hover at a world position (m) with a heading (rad).
    Position { position: DVec3, yaw: f64 },
}

/// Everything configurable about [`TiltrotorController`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TiltrotorConfig {
    /// Rate-loop bandwidth (rad/s); default: `0.5/(motor_lag + τ_servo)`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rate_bandwidth: Option<f64>,
    /// Each outer loop's bandwidth is the inner one's over this ratio (attitude, velocity,
    /// position).
    pub loop_ratio: f64,
    /// Time constant (s) with which the throttles drive the rotor speeds to their targets.
    pub motor_lag: f64,
    /// Largest pitch either way (rad): the schedule's trims and the velocity loop keep within.
    pub max_pitch: f64,
    /// Largest bank the velocity loop asks for (rad).
    pub max_bank: f64,
    /// How fast the velocity reference follows the command (m/s²; horizontal, vertical).
    pub accel: [f64; 2],
    /// Largest differential tilt (rad; left mounts forward, right back) for yaw.
    pub max_differential_tilt: f64,
    /// Largest speed the position loop asks for (m/s, horizontal and vertical).
    pub max_speed: [f64; 2],
    /// Air density and gravity of the schedule's trims.
    pub design_density: f64,
    pub gravity: f64,
}

impl Default for TiltrotorConfig {
    fn default() -> Self {
        Self {
            rate_bandwidth: None,
            loop_ratio: 4.0,
            motor_lag: 0.02,
            max_pitch: 15f64.to_radians(),
            max_bank: 30f64.to_radians(),
            accel: [2.0, 1.0],
            max_differential_tilt: 0.3,
            max_speed: [5.0, 2.0],
            design_density: 1.225,
            gravity: 9.80665,
        }
    }
}

/// Setpoint ranges of the normalised action modes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TiltrotorActionLimits {
    /// Roll and pitch at ±1 in `attitude` (rad).
    pub roll: f64,
    pub pitch: f64,
    /// Yaw rate at ±1 (rad/s).
    pub yaw_rate: f64,
    /// Climb rate at ±1 in `attitude` (m/s).
    pub climb: f64,
    /// Speeds (m/s): forward (and the `attitude` airspeed at +1), sideways and backward,
    /// vertical; default: 90 % of the schedule's fastest speed, 3 and 2.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speed: Option<[f64; 3]>,
}

impl Default for TiltrotorActionLimits {
    fn default() -> Self {
        Self { roll: 30f64.to_radians(), pitch: 15f64.to_radians(), yaw_rate: 0.5, climb: 3.0, speed: None }
    }
}

/// Normalised actions → setpoints for one action mode.
#[derive(Clone, Debug)]
pub struct TiltrotorActionMap {
    mode: TiltrotorActionMode,
    limits: TiltrotorActionLimits,
    rotors: usize,
    /// Tilt range of the mounts (rad).
    tilt: [f64; 2],
    speed: [f64; 3],
}

impl TiltrotorActionMap {
    pub fn new(
        mode: TiltrotorActionMode,
        limits: &TiltrotorActionLimits,
        def: &TiltrotorDef,
        controller: &TiltrotorController,
    ) -> Result<Self, ControlError> {
        let t = &def.controls.tilt;
        let speed = limits.speed.unwrap_or([0.9 * controller.schedule().max_speed(), 3.0, 2.0]);
        if !speed.iter().all(|s| s.is_finite() && *s >= 0.0) {
            return Err(ControlError::InvalidConfig(format!("tiltrotor action speeds {speed:?}")));
        }
        Ok(Self { mode, limits: limits.clone(), rotors: def.rotors.len(), tilt: [t.min, t.max], speed })
    }

    pub fn mode(&self) -> TiltrotorActionMode {
        self.mode
    }

    pub fn limits(&self) -> &TiltrotorActionLimits {
        &self.limits
    }

    /// Speeds (forward, sideways and backward, vertical; m/s).
    pub fn speeds(&self) -> [f64; 3] {
        self.speed
    }

    /// Action length.
    pub fn dim(&self) -> usize {
        match self.mode {
            TiltrotorActionMode::Raw => 2 * self.rotors + 3,
            TiltrotorActionMode::Attitude => 5,
            TiltrotorActionMode::Velocity => 4,
        }
    }

    /// Setpoint for `action` (length [`dim`](Self::dim), already in `[−1, 1]`).
    pub fn setpoint(&self, action: &[f64]) -> TiltrotorSetpoint {
        let l = &self.limits;
        match self.mode {
            TiltrotorActionMode::Raw => {
                let n = self.rotors;
                let [lo, hi] = self.tilt;
                let mut input = TiltrotorInput::default();
                for k in 0..n.min(MAX_ROTORS) {
                    input.throttle[k] = 0.5 * (action[k] + 1.0);
                    input.tilt[k] = lo + 0.5 * (action[n + k] + 1.0) * (hi - lo);
                }
                input.aileron = action[2 * n];
                input.elevator = action[2 * n + 1];
                input.rudder = action[2 * n + 2];
                TiltrotorSetpoint::Raw(input)
            }
            TiltrotorActionMode::Attitude => TiltrotorSetpoint::Attitude {
                roll: action[0] * l.roll,
                pitch: action[1] * l.pitch,
                yaw_rate: action[2] * l.yaw_rate,
                climb: action[3] * l.climb,
                airspeed: 0.5 * (action[4] + 1.0) * self.speed[0],
            },
            TiltrotorActionMode::Velocity => {
                let [fwd, side, vert] = self.speed;
                let vx = if action[0] >= 0.0 { action[0] * fwd } else { action[0] * side };
                TiltrotorSetpoint::Velocity {
                    velocity: DVec3::new(vx, action[1] * side, action[2] * vert),
                    yaw_rate: action[3] * l.yaw_rate,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use autonomousim_vehicles::presets;
    use std::sync::Arc;

    #[test]
    fn actions_span_the_setpoints() {
        let def = Arc::new(presets::tiltrotor("quadtilt_like").unwrap());
        let c = TiltrotorController::new(&def, 0.002, &TiltrotorConfig::default()).unwrap();
        let limits = TiltrotorActionLimits::default();
        let map = TiltrotorActionMap::new(TiltrotorActionMode::Raw, &limits, &def, &c).unwrap();
        assert_eq!(map.dim(), 11);
        let mut a = vec![-1.0; 11];
        a[4] = 1.0;
        a[8] = 0.5;
        let TiltrotorSetpoint::Raw(u) = map.setpoint(&a) else { panic!() };
        assert_eq!(u.throttle, [0.0; 4]);
        assert_eq!(u.tilt[0], def.controls.tilt.max);
        assert_eq!(u.tilt[1], def.controls.tilt.min);
        assert_eq!((u.aileron, u.elevator, u.rudder), (0.5, -1.0, -1.0));
        let map = TiltrotorActionMap::new(TiltrotorActionMode::Velocity, &limits, &def, &c).unwrap();
        let fwd = map.speeds()[0];
        assert!(fwd > 1.5 * c.schedule().stall_speed(), "{fwd}");
        let sp = map.setpoint(&[1.0, 0.5, -1.0, 1.0]);
        assert_eq!(sp, TiltrotorSetpoint::Velocity { velocity: DVec3::new(fwd, 1.5, -2.0), yaw_rate: 0.5 });
        let map = TiltrotorActionMap::new(TiltrotorActionMode::Attitude, &limits, &def, &c).unwrap();
        let TiltrotorSetpoint::Attitude { airspeed, climb, .. } = map.setpoint(&[0.0, 0.0, 0.0, 1.0, -1.0]) else {
            panic!()
        };
        assert_eq!((airspeed, climb), (0.0, 3.0));
        for m in TiltrotorActionMode::ALL {
            assert_eq!(m.name().parse::<TiltrotorActionMode>().unwrap(), m);
        }
    }
}
