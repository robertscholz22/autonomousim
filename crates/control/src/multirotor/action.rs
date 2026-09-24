//! Normalised action modes: policy outputs in `[−1, 1]` → cascade setpoints.
//!
//! | Mode | Components | Meaning at ±1 |
//! |---|---|---|
//! | `motors` | one per rotor | rotor speed from `ω_min` (−1) to `ω_max` (+1) |
//! | `ctbr` | roll, pitch, yaw rate, thrust | ±`rates`; collective from 0 (−1) to full (+1) |
//! | `attitude` | tilt x, tilt y, yaw rate, thrust | tilt vector in the heading frame up to `tilt`, ±`yaw_rate` |
//! | `velocity` | vx, vy, vz, yaw rate | ±`speed_xy` (horizontal length clipped to it), ±`speed_z`, ±`yaw_rate` |
//! | `position` | dx, dy, dz, yaw | offset from the current position in the heading frame, ±`offset`; heading change ±`yaw_offset` |
//!
//! Components outside `[−1, 1]` are clipped and non-finite ones read as 0.

use super::{Frame, Setpoint, StateEstimate, YawCommand};
use autonomousim_core::math::quat::{from_yaw, yaw};
use autonomousim_vehicles::multirotor::{MAX_ROTORS, MultirotorDef};
use glam::{DVec2, DVec3};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionMode {
    Motors,
    /// Collective thrust and body rates.
    #[default]
    Ctbr,
    Attitude,
    Velocity,
    Position,
}

impl ActionMode {
    pub const ALL: [ActionMode; 5] =
        [ActionMode::Motors, ActionMode::Ctbr, ActionMode::Attitude, ActionMode::Velocity, ActionMode::Position];

    pub fn name(self) -> &'static str {
        match self {
            ActionMode::Motors => "motors",
            ActionMode::Ctbr => "ctbr",
            ActionMode::Attitude => "attitude",
            ActionMode::Velocity => "velocity",
            ActionMode::Position => "position",
        }
    }

    /// Action length for a vehicle with `num_rotors` rotors.
    pub fn dim(self, num_rotors: usize) -> usize {
        match self {
            ActionMode::Motors => num_rotors,
            _ => 4,
        }
    }
}

impl fmt::Display for ActionMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for ActionMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::ALL.into_iter().find(|m| m.name() == s).ok_or_else(|| format!("unknown action mode {s:?}"))
    }
}

/// Full-scale values of the normalised actions.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ActionLimits {
    /// `ctbr` body rates (rad/s).
    pub rates: DVec3,
    /// `attitude` tilt (rad).
    pub tilt: f64,
    /// `attitude` and `velocity` heading rate (rad/s).
    pub yaw_rate: f64,
    /// `velocity` horizontal and vertical speed (m/s).
    pub speed_xy: f64,
    pub speed_z: f64,
    pub velocity_frame: Frame,
    /// `position` offset (m) and heading change (rad).
    pub offset: DVec3,
    pub yaw_offset: f64,
}

impl Default for ActionLimits {
    fn default() -> Self {
        use std::f64::consts::PI;
        Self {
            rates: DVec3::new(2.0 * PI, 2.0 * PI, PI),
            tilt: 35f64.to_radians(),
            yaw_rate: PI / 2.0,
            speed_xy: 5.0,
            speed_z: 2.0,
            velocity_frame: Frame::Heading,
            offset: DVec3::new(5.0, 5.0, 2.0),
            yaw_offset: PI,
        }
    }
}

/// Maps normalised actions of one [`ActionMode`] to setpoints for one vehicle.
#[derive(Clone, Debug)]
pub struct ActionMap {
    mode: ActionMode,
    limits: ActionLimits,
    n: usize,
    omega_min: f64,
    omega_max: f64,
    thrust_max: f64,
}

impl ActionMap {
    /// `max_thrust`: collective at full rotor speed (N), e.g.
    /// [`MultirotorController::max_thrust`](super::MultirotorController::max_thrust).
    pub fn new(mode: ActionMode, limits: ActionLimits, def: &MultirotorDef, max_thrust: f64) -> Self {
        Self {
            mode,
            limits,
            n: def.rotors.len(),
            omega_min: def.rotor.omega_min,
            omega_max: def.rotor.omega_max,
            thrust_max: max_thrust,
        }
    }

    pub fn mode(&self) -> ActionMode {
        self.mode
    }

    pub fn limits(&self) -> &ActionLimits {
        &self.limits
    }

    pub fn dim(&self) -> usize {
        self.mode.dim(self.n)
    }

    /// Normalised collective that produces `thrust` (N) in the `ctbr` and `attitude` modes.
    pub fn thrust_action(&self, thrust: f64) -> f64 {
        2.0 * thrust / self.thrust_max - 1.0
    }

    /// Setpoint for `action` (length [`dim`](Self::dim)); `est` anchors the `position` mode.
    pub fn setpoint(&self, action: &[f64], est: &StateEstimate) -> Setpoint {
        assert_eq!(action.len(), self.dim(), "{} action length", self.mode);
        let a = |i: usize| if action[i].is_finite() { action[i].clamp(-1.0, 1.0) } else { 0.0 };
        let l = &self.limits;
        let thrust = |x: f64| 0.5 * (x + 1.0) * self.thrust_max;
        let disc = |x: f64, y: f64| DVec2::new(x, y).clamp_length_max(1.0);
        match self.mode {
            ActionMode::Motors => {
                let mut w = [0.0; MAX_ROTORS];
                for (i, w) in w.iter_mut().enumerate().take(self.n) {
                    *w = self.omega_min + 0.5 * (a(i) + 1.0) * (self.omega_max - self.omega_min);
                }
                Setpoint::Motors(w)
            }
            ActionMode::Ctbr => Setpoint::Ctbr { thrust: thrust(a(3)), rates: DVec3::new(a(0), a(1), a(2)) * l.rates },
            ActionMode::Attitude => Setpoint::Attitude {
                tilt: disc(a(0), a(1)) * l.tilt,
                yaw: YawCommand::Rate(a(2) * l.yaw_rate),
                thrust: thrust(a(3)),
            },
            ActionMode::Velocity => {
                let xy = disc(a(0), a(1)) * l.speed_xy;
                Setpoint::Velocity {
                    velocity: xy.extend(a(2) * l.speed_z),
                    frame: l.velocity_frame,
                    yaw: YawCommand::Rate(a(3) * l.yaw_rate),
                }
            }
            ActionMode::Position => {
                let heading = yaw(est.attitude);
                let offset = DVec3::new(a(0), a(1), a(2)) * l.offset;
                Setpoint::Position {
                    position: est.position + from_yaw(heading) * offset,
                    yaw: YawCommand::Angle(heading + a(3) * l.yaw_offset),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use autonomousim_vehicles::presets;
    use glam::DQuat;

    fn est() -> StateEstimate {
        StateEstimate {
            position: DVec3::new(1.0, 2.0, 3.0),
            velocity: DVec3::ZERO,
            attitude: DQuat::from_rotation_z(std::f64::consts::FRAC_PI_2),
            rates: DVec3::ZERO,
            ground_contact: false,
        }
    }

    #[test]
    fn modes_scale_and_clip() {
        let def = presets::multirotor("cf2x").unwrap();
        let map = |m| ActionMap::new(m, ActionLimits::default(), &def, 1.0);
        for m in ActionMode::ALL {
            assert_eq!(m.name().parse::<ActionMode>().unwrap(), m);
            assert_eq!(map(m).dim(), 4);
            // Garbage in: finite, bounded setpoints out.
            let sp = map(m).setpoint(&[f64::NAN, 7.0, -f64::INFINITY, 0.5], &est());
            assert_eq!(sp.mode(), m);
        }
        let Setpoint::Motors(w) = map(ActionMode::Motors).setpoint(&[-1.0, 1.0, 0.0, 5.0], &est()) else { panic!() };
        assert_eq!(&w[..4], &[0.0, def.rotor.omega_max, 0.5 * def.rotor.omega_max, def.rotor.omega_max]);
        let Setpoint::Ctbr { thrust, rates } = map(ActionMode::Ctbr).setpoint(&[0.5, 0.0, f64::NAN, 0.0], &est())
        else {
            panic!()
        };
        assert!((thrust - 0.5).abs() < 1e-15 && (rates.x - std::f64::consts::PI).abs() < 1e-15 && rates.z == 0.0);
        assert!((map(ActionMode::Ctbr).thrust_action(0.25) + 0.5).abs() < 1e-15);
        let Setpoint::Velocity { velocity, .. } = map(ActionMode::Velocity).setpoint(&[1.0, 1.0, -1.0, 0.0], &est())
        else {
            panic!()
        };
        assert!((velocity.truncate().length() - 5.0).abs() < 1e-12 && velocity.z == -2.0);
        // Position offsets are in the heading frame (here: facing north).
        let Setpoint::Position { position, yaw: YawCommand::Angle(y) } =
            map(ActionMode::Position).setpoint(&[1.0, 0.0, 0.5, -0.5], &est())
        else {
            panic!()
        };
        assert!((position - DVec3::new(1.0, 7.0, 4.0)).length() < 1e-12 && y.abs() < 1e-12);
    }
}
