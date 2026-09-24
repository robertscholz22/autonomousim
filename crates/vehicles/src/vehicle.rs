//! Vehicle instances of any family behind one enum, with the queries and step phases that the
//! simulation shares between families. Family-specific parts (rotor commands, driver input)
//! are reached through [`Vehicle::as_multirotor`] and friends.

use crate::VehicleDef;
use crate::ground::{GroundPart, Wheeled, WheeledDef, WheeledInit};
use crate::multirotor::{ColliderPart, InitialState, MotorInit, Multirotor, MultirotorDef};
use autonomousim_core::contact::{ContactModel, ContactPoint, SphereCollider, StaticScene};
use autonomousim_core::dynamics::{DynamicsError, MbState};
use autonomousim_core::math::Pose;
use glam::{DQuat, DVec3};
use serde::Serialize;
use std::sync::Arc;

/// Vehicle families; each has its own controller and action modes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Family {
    Multirotor,
    Wheeled,
}

impl Family {
    pub fn name(self) -> &'static str {
        match self {
            Family::Multirotor => "multirotor",
            Family::Wheeled => "wheeled",
        }
    }
}

impl std::fmt::Display for Family {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// A vehicle definition shared by `Arc` between instances.
#[derive(Clone, Debug, PartialEq)]
pub enum SharedDef {
    Multirotor(Arc<MultirotorDef>),
    Wheeled(Arc<WheeledDef>),
}

impl From<VehicleDef> for SharedDef {
    fn from(d: VehicleDef) -> Self {
        match d {
            VehicleDef::Multirotor(m) => SharedDef::Multirotor(Arc::new(m)),
            VehicleDef::Wheeled(w) => SharedDef::Wheeled(Arc::new(w)),
        }
    }
}

impl SharedDef {
    pub fn name(&self) -> &str {
        match self {
            SharedDef::Multirotor(m) => &m.name,
            SharedDef::Wheeled(w) => &w.name,
        }
    }

    pub fn family(&self) -> Family {
        match self {
            SharedDef::Multirotor(_) => Family::Multirotor,
            SharedDef::Wheeled(_) => Family::Wheeled,
        }
    }

    pub fn as_multirotor(&self) -> Option<&Arc<MultirotorDef>> {
        match self {
            SharedDef::Multirotor(m) => Some(m),
            _ => None,
        }
    }

    pub fn as_wheeled(&self) -> Option<&Arc<WheeledDef>> {
        match self {
            SharedDef::Wheeled(w) => Some(w),
            _ => None,
        }
    }

    /// Nominal total mass (kg).
    pub fn mass(&self) -> f64 {
        match self {
            SharedDef::Multirotor(m) => m.body.mass,
            SharedDef::Wheeled(w) => w.total_mass(),
        }
    }

    /// Penalty contact model of the colliders for a vehicle of `mass` stepped every `dt`.
    pub fn contact_model(&self, mass: f64, dt: f64) -> ContactModel {
        match self {
            SharedDef::Multirotor(m) => m.contact.model(mass, dt),
            SharedDef::Wheeled(w) => w.contact.model(mass, dt),
        }
    }

    /// Collision spheres in the body frame.
    pub fn sphere_colliders(&self) -> Vec<SphereCollider> {
        match self {
            SharedDef::Multirotor(m) => m.sphere_colliders(),
            SharedDef::Wheeled(w) => w.sphere_colliders(),
        }
    }
}

/// One simulated vehicle of any family.
///
/// The shared step phases are [`begin_step`](Self::begin_step) →
/// family-specific actuation (rotors, or drive and tyres) → [`apply_contacts`](Self::apply_contacts) /
/// [`apply_force`](Self::apply_force) → [`finish_step`](Self::finish_step).
#[derive(Clone, Debug)]
// Agents hold their vehicle inline; boxing would add an indirection to every query.
#[allow(clippy::large_enum_variant)]
pub enum Vehicle {
    Multirotor(Multirotor),
    Wheeled(Wheeled),
}

macro_rules! each {
    ($self:expr, $v:ident => $e:expr) => {
        match $self {
            Vehicle::Multirotor($v) => $e,
            Vehicle::Wheeled($v) => $e,
        }
    };
}

impl Vehicle {
    /// Instance with nominal parameters for physics step `dt`.
    pub fn new(def: &SharedDef, dt: f64) -> Self {
        match def {
            SharedDef::Multirotor(m) => Vehicle::Multirotor(Multirotor::new(m.clone(), dt)),
            SharedDef::Wheeled(w) => Vehicle::Wheeled(Wheeled::new(w.clone(), dt)),
        }
    }

    #[inline]
    pub fn family(&self) -> Family {
        match self {
            Vehicle::Multirotor(_) => Family::Multirotor,
            Vehicle::Wheeled(_) => Family::Wheeled,
        }
    }

    #[inline]
    pub fn name(&self) -> &str {
        each!(self, v => &v.def().name)
    }

    #[inline]
    pub fn as_multirotor(&self) -> Option<&Multirotor> {
        match self {
            Vehicle::Multirotor(m) => Some(m),
            _ => None,
        }
    }

    #[inline]
    pub fn as_multirotor_mut(&mut self) -> Option<&mut Multirotor> {
        match self {
            Vehicle::Multirotor(m) => Some(m),
            _ => None,
        }
    }

    #[inline]
    pub fn as_wheeled(&self) -> Option<&Wheeled> {
        match self {
            Vehicle::Wheeled(w) => Some(w),
            _ => None,
        }
    }

    #[inline]
    pub fn as_wheeled_mut(&mut self) -> Option<&mut Wheeled> {
        match self {
            Vehicle::Wheeled(w) => Some(w),
            _ => None,
        }
    }

    // ---------------------------------------------------------------- state access

    /// Multibody state (the free base first: position, quaternion; angular, linear velocity).
    #[inline]
    pub fn state(&self) -> &MbState {
        each!(self, v => &v.state)
    }

    #[inline]
    pub fn state_mut(&mut self) -> &mut MbState {
        each!(self, v => &mut v.state)
    }

    #[inline]
    pub fn position(&self) -> DVec3 {
        each!(self, v => v.position())
    }

    /// Body → world rotation.
    #[inline]
    pub fn orientation(&self) -> DQuat {
        each!(self, v => v.orientation())
    }

    #[inline]
    pub fn pose(&self) -> Pose {
        each!(self, v => v.pose())
    }

    #[inline]
    pub fn lin_vel_body(&self) -> DVec3 {
        each!(self, v => v.lin_vel_body())
    }

    #[inline]
    pub fn lin_vel_world(&self) -> DVec3 {
        each!(self, v => v.lin_vel_world())
    }

    #[inline]
    pub fn ang_vel_body(&self) -> DVec3 {
        each!(self, v => v.ang_vel_body())
    }

    /// Current total mass (kg), with randomisation.
    #[inline]
    pub fn mass(&self) -> f64 {
        each!(self, v => v.mass())
    }

    /// Specific force (accelerometer reading, body frame) of the last step.
    #[inline]
    pub fn specific_force_body(&self) -> DVec3 {
        each!(self, v => v.specific_force_body())
    }

    /// Angular acceleration (body frame) over the last step.
    #[inline]
    pub fn ang_acc_body(&self) -> DVec3 {
        each!(self, v => v.ang_acc_body())
    }

    #[inline]
    pub fn colliders(&self) -> &[SphereCollider] {
        each!(self, v => v.colliders())
    }

    /// Active contacts of the colliders with the static world in the last step.
    #[inline]
    pub fn contacts(&self) -> &[ContactPoint] {
        each!(self, v => v.contacts())
    }

    /// Whether contact on collider group `group` is expected (landing gear, skids) rather than
    /// a collision.
    #[inline]
    pub fn is_gear(&self, group: u8) -> bool {
        match self {
            Vehicle::Multirotor(_) => ColliderPart::from_group(group) == Some(ColliderPart::Gear),
            Vehicle::Wheeled(_) => group == GroundPart::Skid as u8,
        }
    }

    /// Put the vehicle at `pose` with the given velocities and clear its transient state
    /// (rotors idle; a ground vehicle's suspension at its static travel, wheels rolling).
    pub fn place(&mut self, pose: Pose, lin_vel_world: DVec3, ang_vel_body: DVec3) {
        match self {
            Vehicle::Multirotor(v) => {
                v.reset(&InitialState { pose, lin_vel_world, ang_vel_body, motors: MotorInit::Idle, soc: 1.0 })
            }
            Vehicle::Wheeled(v) => v.reset(&WheeledInit { pose, lin_vel_world, ang_vel_body }),
        }
    }

    // ---------------------------------------------------------------- step phases

    /// Forward kinematics and cleared force accumulators.
    #[inline]
    pub fn begin_step(&mut self) {
        each!(self, v => v.begin_step())
    }

    /// Penalty contacts of the colliders with the static world.
    #[inline]
    pub fn apply_contacts(&mut self, scene: &StaticScene) {
        each!(self, v => v.apply_contacts(scene))
    }

    /// Add an external force (world frame) at a world point.
    #[inline]
    pub fn apply_force(&mut self, force: DVec3, point: DVec3) {
        each!(self, v => v.apply_force(force, point))
    }

    /// Forward dynamics and integration.
    #[inline]
    pub fn finish_step(&mut self, gravity: DVec3) -> Result<(), DynamicsError> {
        each!(self, v => v.finish_step(gravity))
    }
}
