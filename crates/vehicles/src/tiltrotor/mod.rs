//! Tiltrotors: a rigid airframe with lifting surfaces and fixed-pitch electric rotors on
//! mounts that tilt from hover to forward flight, plus a level-flight trim and the conversion
//! corridor.

mod def;
mod model;
mod trim;

pub use def::{MAX_ROTORS, SurfaceMix, TiltMount, TiltrotorControlsDef, TiltrotorDef};
pub use model::{Tiltrotor, TiltrotorInit, TiltrotorInput, TiltrotorLoads};
pub use trim::{CorridorPoint, TiltrotorTrim, TrimLimits};
