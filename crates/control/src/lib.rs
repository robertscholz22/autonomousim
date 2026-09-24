//! Reference controllers, control allocation and action modes.
//!
//! [`multirotor`] holds a PX4-style cascade (position → velocity → attitude → rate →
//! allocation) whose gains are derived from the vehicle definition, and the normalised action
//! modes that learning agents use to drive it. [`ground`] holds the speed, steering and
//! side-drive loops of wheeled vehicles and their action modes. The enums re-exported here hold
//! either family, for agents that do not know which one they drive.

mod family;
pub mod ground;
pub mod multirotor;

pub use family::{ActionMapping, AgentActionMode, Command, Controller};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ControlError {
    #[error("control allocation impossible: {0}")]
    Unallocatable(String),
    #[error("invalid controller configuration: {0}")]
    InvalidConfig(String),
}
