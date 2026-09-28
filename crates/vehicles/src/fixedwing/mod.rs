//! Fixed-wing aircraft: a rigid airframe with a whole-aircraft aerodynamic model (stability
//! derivatives or JSBSim-style coefficient tables), one propeller on an electric motor or a
//! piston engine (rotor inertia, torque reaction and gyroscopic moment), control surfaces with
//! servo lag and rate limits, JSBSim-like landing gear, airframe colliders and an optional
//! battery; plus a trim solver for steady straight flight.

mod aero;
mod def;
mod gear;
mod model;
mod propulsion;
mod table;
mod trim;

pub use aero::{
    AeroCoefficients, AeroForces, AeroInput, AeroModel, Derivatives, Geometry, Pressure, TableModel, Term, Var,
};
pub use def::{AirframeDef, ControlsDef, FixedWingDef, SurfaceDef};
pub use gear::{GearDef, WheelContact};
pub use model::{FixedWing, FixedWingInit, FixedWingInput};
pub use propulsion::{
    ElectricMotorDef, EngineDef, PistonEngineDef, PropellerDef, Propulsion, PropulsionDef, PropulsionOutput,
};
pub use table::{Curve, Table};
pub use trim::Trim;
