//! Multirotor model: a single rigid body with rotors (first-order motors, thrust and drag
//! torque ∝ ω², spin-up reaction, rotor gyroscopics, rotor drag, Cheeseman–Bennett ground
//! effect, density scaling), quadratic body drag, optional battery sag, and sphere colliders
//! for penalty contacts.

mod aero;
mod battery;
mod def;
mod model;

pub use aero::{AirData, GroundPlane, ground_effect};
pub use battery::BatteryState;
pub use def::{
    BatteryDef, BodyDef, ColliderDef, ColliderPart, ContactDef, MultirotorDef, RotorMount, RotorParams,
    SEA_LEVEL_DENSITY, Spin,
};
pub use model::{InitialState, MotorInit, Multirotor, MultirotorScales, StepEnv};

/// Maximum number of rotors (octocopters and coaxial X8).
pub const MAX_ROTORS: usize = 8;
