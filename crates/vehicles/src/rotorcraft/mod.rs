//! Rotorcraft: the helicopter rotor model shared by main and tail rotors (and the tiltrotor's
//! proprotors).

mod rotor;

pub use rotor::{Rotor, RotorDef, RotorInput, RotorLoads, RotorState, Spin};
