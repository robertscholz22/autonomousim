//! Fixed-wing aircraft: normalised action modes and the controller between setpoints and the
//! aircraft's [`FixedWingInput`].
//!
//! | Mode | Components | Meaning at ±1 |
//! |---|---|---|
//! | `surfaces` | aileron, elevator, rudder, throttle | full deflection (+ roll right, pitch up, yaw right); throttle −1 → 0 (idle), +1 → full |
//!
//! Flaps stay up and the brakes released in `surfaces`; scripted agents set them through a
//! [`FixedWingSetpoint::Surfaces`] command. Components outside `[−1, 1]` are clipped and
//! non-finite ones read as 0.

use autonomousim_vehicles::fixedwing::{FixedWing, FixedWingInput};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FixedWingActionMode {
    /// Control surfaces and throttle directly.
    #[default]
    Surfaces,
}

impl FixedWingActionMode {
    pub const ALL: [FixedWingActionMode; 1] = [FixedWingActionMode::Surfaces];

    pub fn name(self) -> &'static str {
        match self {
            FixedWingActionMode::Surfaces => "surfaces",
        }
    }

    /// Action length.
    pub fn dim(self) -> usize {
        match self {
            FixedWingActionMode::Surfaces => 4,
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

/// What a fixed-wing controller tracks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FixedWingSetpoint {
    /// Normalised surfaces, throttle, flaps and brakes, passed through.
    Surfaces(FixedWingInput),
}

/// Setpoint → aircraft input.
#[derive(Clone, Debug, Default)]
pub struct FixedWingController {}

impl FixedWingController {
    pub fn new() -> Self {
        Self {}
    }

    /// Clear the internal state.
    pub fn reset(&mut self) {}

    /// Input for this step.
    pub fn update(&mut self, setpoint: &FixedWingSetpoint, _aircraft: &FixedWing) -> FixedWingInput {
        match setpoint {
            FixedWingSetpoint::Surfaces(input) => input.clamped(),
        }
    }
}

/// Maps normalised actions of one [`FixedWingActionMode`] to setpoints.
#[derive(Clone, Debug)]
pub struct FixedWingActionMap {
    mode: FixedWingActionMode,
}

impl FixedWingActionMap {
    pub fn new(mode: FixedWingActionMode) -> Self {
        Self { mode }
    }

    pub fn mode(&self) -> FixedWingActionMode {
        self.mode
    }

    pub fn dim(&self) -> usize {
        self.mode.dim()
    }

    /// Setpoint for `action` (length [`dim`](Self::dim), already in `[−1, 1]`).
    pub fn setpoint(&self, action: &[f64]) -> FixedWingSetpoint {
        match self.mode {
            FixedWingActionMode::Surfaces => FixedWingSetpoint::Surfaces(FixedWingInput {
                aileron: action[0],
                elevator: action[1],
                rudder: action[2],
                throttle: 0.5 * (action[3] + 1.0),
                ..FixedWingInput::default()
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn surfaces_mode() {
        let m = FixedWingActionMap::new("surfaces".parse().unwrap());
        assert_eq!(m.dim(), 4);
        let FixedWingSetpoint::Surfaces(i) = m.setpoint(&[0.5, -1.0, 0.25, -1.0]);
        assert_eq!((i.aileron, i.elevator, i.rudder, i.throttle, i.flap, i.brake), (0.5, -1.0, 0.25, 0.0, 0.0, 0.0));
        let FixedWingSetpoint::Surfaces(i) = m.setpoint(&[0.0, 0.0, 0.0, 1.0]);
        assert_eq!(i.throttle, 1.0);
        assert!("raw".parse::<FixedWingActionMode>().is_err());
    }
}
