//! Tiltrotors: normalised action modes and the controller between setpoints and the
//! tiltrotor's [`TiltrotorInput`].
//!
//! | Mode | Components | Meaning at ±1 |
//! |---|---|---|
//! | `raw` | throttle per rotor, tilt per rotor, aileron, elevator, rudder | throttle 0 at −1 and 1 at +1; tilt at either end of the mounts' range (+1 forward); full surface travel (+ roll right, nose up, nose right) |
//!
//! Components outside `[−1, 1]` are clipped and non-finite ones read as 0. `raw` shares its
//! name with the ground vehicles' mode; a group resolves it by its vehicle.

use crate::ControlError;
use autonomousim_vehicles::tiltrotor::{MAX_ROTORS, Tiltrotor, TiltrotorDef, TiltrotorInput};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TiltrotorActionMode {
    /// Throttles, tilts and surfaces.
    #[default]
    Raw,
}

impl TiltrotorActionMode {
    pub const ALL: [TiltrotorActionMode; 1] = [TiltrotorActionMode::Raw];

    pub fn name(self) -> &'static str {
        match self {
            TiltrotorActionMode::Raw => "raw",
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

/// What a tiltrotor's controller tracks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TiltrotorSetpoint {
    /// Pilot inputs, passed through.
    Raw(TiltrotorInput),
}

/// Setpoint → tiltrotor input.
#[derive(Clone, Debug)]
pub struct TiltrotorController {
    dt: f64,
}

impl TiltrotorController {
    pub fn new(def: &TiltrotorDef, dt: f64) -> Result<Self, ControlError> {
        if dt.is_nan() || dt <= 0.0 {
            return Err(ControlError::InvalidConfig(format!("{}: controller step {dt}", def.name)));
        }
        Ok(Self { dt })
    }

    pub fn dt(&self) -> f64 {
        self.dt
    }

    /// Clear the internal state.
    pub fn reset(&mut self) {}

    /// Input for `setpoint` on `_t`.
    pub fn update(&mut self, setpoint: &TiltrotorSetpoint, _t: &Tiltrotor) -> TiltrotorInput {
        match setpoint {
            TiltrotorSetpoint::Raw(input) => input.clamped(),
        }
    }
}

/// Normalised actions → setpoints for one action mode.
#[derive(Clone, Debug)]
pub struct TiltrotorActionMap {
    mode: TiltrotorActionMode,
    rotors: usize,
    /// Tilt range of the mounts (rad).
    tilt: [f64; 2],
}

impl TiltrotorActionMap {
    pub fn new(mode: TiltrotorActionMode, def: &TiltrotorDef) -> Result<Self, ControlError> {
        let t = &def.controls.tilt;
        Ok(Self { mode, rotors: def.rotors.len(), tilt: [t.min, t.max] })
    }

    pub fn mode(&self) -> TiltrotorActionMode {
        self.mode
    }

    /// Action length.
    pub fn dim(&self) -> usize {
        match self.mode {
            TiltrotorActionMode::Raw => 2 * self.rotors + 3,
        }
    }

    /// Setpoint for `action` (length [`dim`](Self::dim), already in `[−1, 1]`).
    pub fn setpoint(&self, action: &[f64]) -> TiltrotorSetpoint {
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
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use autonomousim_vehicles::presets;

    #[test]
    fn raw_actions_span_the_inputs() {
        let def = presets::tiltrotor("quadtilt_like").unwrap();
        let map = TiltrotorActionMap::new(TiltrotorActionMode::Raw, &def).unwrap();
        assert_eq!(map.dim(), 11);
        let mut a = vec![-1.0; 11];
        a[4] = 1.0;
        a[8] = 0.5;
        let TiltrotorSetpoint::Raw(u) = map.setpoint(&a);
        assert_eq!(u.throttle, [0.0; 4]);
        assert_eq!(u.tilt[0], def.controls.tilt.max);
        assert_eq!(u.tilt[1], def.controls.tilt.min);
        assert_eq!((u.aileron, u.elevator, u.rudder), (0.5, -1.0, -1.0));
        assert_eq!("raw".parse::<TiltrotorActionMode>().unwrap(), TiltrotorActionMode::Raw);
    }
}
