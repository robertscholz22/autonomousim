//! Ground vehicles: definitions, tyres, powertrains and the wheeled multibody model.

mod def;
mod powertrain;
pub mod tire;
mod wheeled;

pub use def::{
    AxleDef, BrakeDef, ChassisDef, DamperDef, GroundColliderDef, GroundPart, STANDARD_GRAVITY, SpringDef, StaticState,
    SteeringDef, StopDef, SuspensionDef, TireSpec, WheelDef, WheeledDef, deflection_at, travel_direction,
};
pub use powertrain::{
    CombustionDef, Coupling, DifferentialDef, DriveInput, ElectricDef, EngineDef, GearboxDef, LinearTable, MotorDef,
    MotorSide, Powertrain, PowertrainDef, PowertrainStatus,
};
pub use wheeled::{GroundStepEnv, WheelState, Wheeled, WheeledInit};
