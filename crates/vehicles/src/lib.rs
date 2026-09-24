//! Vehicle definitions and models (multirotors and wheeled ground vehicles; fixed-wing and
//! rotorcraft later).
//!
//! A [`VehicleDef`] is immutable, loaded from TOML (`type = "multirotor"`, `"wheeled"`, …) and
//! shared between instances by `Arc`; instances hold the per-agent state.

pub mod ground;
pub mod multirotor;
pub mod presets;
mod vehicle;

pub use vehicle::{Family, SharedDef, Vehicle};

use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum VehicleError {
    #[error("cannot read vehicle file: {0}")]
    Io(#[from] std::io::Error),
    #[error("cannot parse vehicle definition: {0}")]
    Parse(#[from] toml::de::Error),
    #[error("invalid vehicle definition: {0}")]
    Invalid(String),
    #[error("unknown vehicle preset {0:?}")]
    UnknownPreset(String),
}

/// Any vehicle definition.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum VehicleDef {
    Multirotor(multirotor::MultirotorDef),
    Wheeled(ground::WheeledDef),
}

impl VehicleDef {
    pub fn from_toml(s: &str) -> Result<Self, VehicleError> {
        let mut def: Self = toml::from_str(s)?;
        match &mut def {
            VehicleDef::Multirotor(m) => m.finish()?,
            VehicleDef::Wheeled(w) => w.finish()?,
        }
        Ok(def)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, VehicleError> {
        Self::from_toml(&std::fs::read_to_string(path)?)
    }

    pub fn name(&self) -> &str {
        match self {
            VehicleDef::Multirotor(m) => &m.name,
            VehicleDef::Wheeled(w) => &w.name,
        }
    }

    pub fn family(&self) -> Family {
        match self {
            VehicleDef::Multirotor(_) => Family::Multirotor,
            VehicleDef::Wheeled(_) => Family::Wheeled,
        }
    }
}
