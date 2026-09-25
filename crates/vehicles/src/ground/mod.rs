//! Ground vehicles: definitions, tyres, powertrains and the wheeled multibody model.

mod def;
mod powertrain;
mod statics;
pub mod tire;
mod tree;
pub mod units;
mod wheeled;

pub use def::{
    AxleDef, BrakeDef, ChassisDef, DamperDef, GroundColliderDef, GroundPart, STANDARD_GRAVITY, SpringDef, StaticState,
    SteerMode, SteeringDef, StopDef, SuspensionDef, TireSpec, WheelDef, WheeledDef, deflection_at, travel_direction,
};
pub use powertrain::{
    CombustionDef, Coupling, DifferentialDef, DriveInput, ElectricDef, EngineDef, GearboxDef, LinearTable, MAX_WHEELS,
    MotorDef, MotorSide, Powertrain, PowertrainDef, PowertrainStatus, WheelCommands,
};
pub use units::{
    CouplingDef, CouplingJoint, CouplingKind, DollyDef, HitchDef, TrailerDef, UnitDef, UnitJoint, yaw_pitch_roll,
};
pub use wheeled::{GroundStepEnv, WheelState, Wheeled, WheeledInit};
