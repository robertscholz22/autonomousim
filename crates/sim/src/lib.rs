//! Simulation orchestration: scenarios, world instances with their agents, observations,
//! batched stepping across threads and recording.
//!
//! - [`Scenario`] (TOML/JSON) → [`CompiledScenario`] (validated, maps built, shared by `Arc`);
//! - [`WorldInstance`]: one map, environment and set of agents; steps physics ticks and policy
//!   steps, writes observations, state rows and events;
//! - [`BatchSim`]: many worlds stepped in parallel on a dedicated thread pool, with flat
//!   output arrays per agent group (the layout Python sees);
//! - [`record`]: MCAP recording through a [`TelemetrySink`](record::TelemetrySink);
//! - [`policy`]: trained policies exported from Python, run without Python.

pub mod agent;
pub mod batch;
pub mod drive;
pub mod events;
pub mod interaction;
pub mod obs;
pub mod policy;
pub mod record;
pub mod scenario;
pub mod world;

pub use agent::{Agent, EnvState};
pub use batch::BatchSim;
pub use events::Events;
pub use obs::{ObsTerm, TermKind};
pub use scenario::{CompiledGroup, CompiledScenario, Goal, GroupSpec, MapSource, Scenario, Testworld};
pub use world::{STATE_DIM, STATE_FIELDS, WorldInstance};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum SimError {
    #[error("invalid scenario: {0}")]
    Scenario(String),
    #[error("invalid policy: {0}")]
    Policy(String),
    #[error(transparent)]
    Vehicle(#[from] autonomousim_vehicles::VehicleError),
    #[error(transparent)]
    Control(#[from] autonomousim_control::ControlError),
    #[error(transparent)]
    Sensor(#[from] autonomousim_sensors::SensorError),
    #[error(transparent)]
    Rate(#[from] autonomousim_core::time::RateError),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("recording error: {0}")]
    Record(String),
}
