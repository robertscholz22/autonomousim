//! Rotorcraft: the rotor model shared by main and tail rotors (and the tiltrotor's
//! proprotors), and single-main-rotor helicopters with a geared tail rotor, a governed engine,
//! fins and skids, plus a trim solver.

mod def;
mod model;
mod rotor;
mod trim;

pub use def::{EngineDef, FuselageDef, HelicopterControlsDef, HelicopterDef, PitchChannel, RotorMount};
pub use model::{Helicopter, HelicopterDisplay, HelicopterInit, HelicopterInput, HelicopterLoads};
pub use rotor::{Rotor, RotorDef, RotorInput, RotorLoads, RotorState, Spin};
pub use trim::{HelicopterLinear, HelicopterTrim, attitude};
