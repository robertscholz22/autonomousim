//! Tyre models: the Magic Formula (MF 5.2 / 6.1 from `.tir` files) and Fiala, with road-plane
//! contact and transient slip, and track patches (road wheels on a track).

mod fiala;
mod mf;
mod model;
mod road;
mod tir;
mod track;

pub use fiala::{FialaOutput, FialaParams};
pub use mf::{MfInput, MfOutput, MfParams, MfVersion, REQUIRED};
pub use model::{
    REFERENCE_FRICTION, REFERENCE_ROLLING_RESISTANCE, Surface, Tire, TireForces, TireModel, TireState, WheelMotion,
};
pub use road::{RoadContact, road_contact};
pub use tir::{TirError, TirFile, TirValue};
pub use track::{TRACK_CELLS, TrackPatch};
