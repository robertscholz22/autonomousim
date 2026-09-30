//! Ground vehicles: definitions, tyres, powertrains, the wheeled multibody model and the
//! linearised single-track (bicycle) model.

mod def;
mod powertrain;
pub mod single_track;
mod statics;
pub mod tire;
mod tree;
pub mod units;
mod wheeled;

pub use def::{
    AxleDef, BrakeDef, ChassisDef, DamperDef, FeetDef, GroundColliderDef, GroundPart, RiderDef, RollerDef,
    STANDARD_GRAVITY, SpringDef, StaticState, Steer, SteerMode, SteerName, SteeringDef, SteeringHeadDef, StopDef,
    SuspensionDef, TireSpec, TrackDef, TrackSteering, TrailingArmDef, WheelDef, WheeledDef, deflection_at,
    inertia_tensor, travel_direction,
};
pub use powertrain::{
    CombustionDef, Coupling, DifferentialDef, DriveInput, ElectricDef, EngineDef, GearboxDef, LinearTable, MAX_WHEELS,
    MotorDef, MotorSide, Powertrain, PowertrainDef, PowertrainStatus, TorqueConverterDef, WheelCommands,
};
pub use units::{
    CouplingDef, CouplingJoint, CouplingKind, DollyDef, HitchDef, TrailerDef, UnitDef, UnitJoint, yaw_pitch_roll,
};
pub use wheeled::{
    CurrentPoses, GroundStepEnv, KinematicLimits, KinematicState, KinematicTarget, WheelState, Wheeled, WheeledInit,
    wheel_angle,
};
