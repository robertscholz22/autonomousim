//! Reference controllers, control allocation and action modes.
//!
//! [`multirotor`] holds a PX4-style cascade (position → velocity → attitude → rate →
//! allocation) whose gains are derived from the vehicle definition, and the normalised action
//! modes that learning agents use to drive it.

pub mod ground;
pub mod multirotor;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ControlError {
    #[error("control allocation impossible: {0}")]
    Unallocatable(String),
    #[error("invalid controller configuration: {0}")]
    InvalidConfig(String),
}
