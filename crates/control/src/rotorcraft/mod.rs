//! Helicopter action modes and controller. `sticks` passes the pilot inputs through.

use crate::ControlError;
use autonomousim_vehicles::rotorcraft::{Helicopter, HelicopterDef, HelicopterInput};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HelicopterActionMode {
    /// Collective, longitudinal and lateral cyclic, pedals.
    #[default]
    Sticks,
}

impl HelicopterActionMode {
    pub const ALL: [HelicopterActionMode; 1] = [HelicopterActionMode::Sticks];

    pub fn name(self) -> &'static str {
        match self {
            HelicopterActionMode::Sticks => "sticks",
        }
    }

    /// Action length.
    pub fn dim(self) -> usize {
        match self {
            HelicopterActionMode::Sticks => 4,
        }
    }
}

impl fmt::Display for HelicopterActionMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for HelicopterActionMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::ALL.into_iter().find(|m| m.name() == s).ok_or_else(|| format!("unknown helicopter action mode {s:?}"))
    }
}

/// What a helicopter's controller tracks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HelicopterSetpoint {
    /// Pilot inputs, passed through.
    Sticks(HelicopterInput),
}

/// A helicopter's reference controller.
#[derive(Clone, Debug)]
pub struct HelicopterController {
    dt: f64,
}

impl HelicopterController {
    pub fn new(def: &Arc<HelicopterDef>, dt: f64) -> Result<Self, ControlError> {
        def.validate().map_err(|e| ControlError::InvalidConfig(e.to_string()))?;
        if dt.is_nan() || dt <= 0.0 {
            return Err(ControlError::InvalidConfig(format!("helicopter controller step {dt}")));
        }
        Ok(Self { dt })
    }

    pub fn dt(&self) -> f64 {
        self.dt
    }

    /// Clear the internal state.
    pub fn reset(&mut self) {}

    /// Pilot input for `setpoint`.
    pub fn update(&mut self, setpoint: &HelicopterSetpoint, _h: &Helicopter) -> HelicopterInput {
        match setpoint {
            HelicopterSetpoint::Sticks(input) => input.clamped(),
        }
    }
}

/// Normalised actions → helicopter setpoints.
#[derive(Clone, Debug)]
pub struct HelicopterActionMap {
    mode: HelicopterActionMode,
}

impl HelicopterActionMap {
    pub fn new(mode: HelicopterActionMode) -> Self {
        Self { mode }
    }

    pub fn mode(&self) -> HelicopterActionMode {
        self.mode
    }

    pub fn dim(&self) -> usize {
        self.mode.dim()
    }

    /// Setpoint for `action` (length [`dim`](Self::dim), already in `[−1, 1]`).
    pub fn setpoint(&self, action: &[f64]) -> HelicopterSetpoint {
        match self.mode {
            HelicopterActionMode::Sticks => {
                HelicopterSetpoint::Sticks(HelicopterInput::from_array([action[0], action[1], action[2], action[3]]))
            }
        }
    }
}
