//! Controllers, commands and action modes of any vehicle family behind one enum each, so that
//! agents can hold them without knowing the family. Action modes are per family; their names
//! are disjoint, so a mode name alone selects the family.

use crate::ControlError;
use crate::ground::GroundSetpoint;
use crate::ground::{GroundActionLimits, GroundActionMap, GroundActionMode, GroundConfig, GroundController};
use crate::multirotor::{ActionLimits, ActionMap, ActionMode, ControllerConfig, MultirotorController};
use crate::multirotor::{Setpoint, StateEstimate, YawCommand};
use autonomousim_core::math::Pose;
use autonomousim_core::math::quat::yaw;
use autonomousim_vehicles::{Family, SharedDef, Vehicle};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;
use std::str::FromStr;

/// An action mode of some family, written by its name (`"ctbr"`, `"vk"`, …).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AgentActionMode {
    Multirotor(ActionMode),
    Ground(GroundActionMode),
}

impl AgentActionMode {
    /// The default mode of a family.
    pub fn default_for(family: Family) -> Self {
        match family {
            Family::Multirotor => ActionMode::default().into(),
            Family::Wheeled => GroundActionMode::default().into(),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            AgentActionMode::Multirotor(m) => m.name(),
            AgentActionMode::Ground(m) => m.name(),
        }
    }

    /// Whether this mode drives vehicles of `family`.
    pub fn fits(self, family: Family) -> bool {
        matches!(
            (self, family),
            (AgentActionMode::Multirotor(_), Family::Multirotor) | (AgentActionMode::Ground(_), Family::Wheeled)
        )
    }
}

impl From<ActionMode> for AgentActionMode {
    fn from(m: ActionMode) -> Self {
        AgentActionMode::Multirotor(m)
    }
}

impl From<GroundActionMode> for AgentActionMode {
    fn from(m: GroundActionMode) -> Self {
        AgentActionMode::Ground(m)
    }
}

impl fmt::Display for AgentActionMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for AgentActionMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if let Ok(m) = s.parse::<ActionMode>() {
            return Ok(m.into());
        }
        if let Ok(m) = s.parse::<GroundActionMode>() {
            return Ok(m.into());
        }
        let names: Vec<&str> =
            ActionMode::ALL.iter().map(|m| m.name()).chain(GroundActionMode::ALL.iter().map(|m| m.name())).collect();
        Err(format!("unknown action mode {s:?} (expected one of {})", names.join(", ")))
    }
}

impl Serialize for AgentActionMode {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.name())
    }
}

impl<'de> Deserialize<'de> for AgentActionMode {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

/// What an agent's controller tracks.
#[derive(Clone, Copy, Debug, PartialEq)]
// Per-wheel ground input makes the ground variant large; commands are replaced every policy
// step, and boxing would allocate each time.
#[allow(clippy::large_enum_variant)]
pub enum Command {
    Multirotor(Setpoint),
    Ground(GroundSetpoint),
}

impl From<Setpoint> for Command {
    fn from(s: Setpoint) -> Self {
        Command::Multirotor(s)
    }
}

impl From<GroundSetpoint> for Command {
    fn from(s: GroundSetpoint) -> Self {
        Command::Ground(s)
    }
}

impl Command {
    /// Hold a vehicle of `family` at `pose`: hover there with its heading, or stand still.
    pub fn hold(family: Family, pose: &Pose) -> Self {
        match family {
            Family::Multirotor => {
                Setpoint::Position { position: pose.pos, yaw: YawCommand::Angle(yaw(pose.rot)) }.into()
            }
            Family::Wheeled => GroundSetpoint::SpeedCurvature { speed: 0.0, curvature: 0.0 }.into(),
        }
    }

    pub fn as_multirotor(&self) -> Option<&Setpoint> {
        match self {
            Command::Multirotor(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_ground(&self) -> Option<&GroundSetpoint> {
        match self {
            Command::Ground(s) => Some(s),
            _ => None,
        }
    }
}

/// An agent's reference controller.
#[derive(Clone, Debug)]
// The multirotor cascade is the large variant and by far the most common one.
#[allow(clippy::large_enum_variant)]
pub enum Controller {
    Multirotor(MultirotorController),
    Ground(GroundController),
}

impl Controller {
    /// Controller for `def` running every `dt`; each family takes its own configuration.
    pub fn new(
        def: &SharedDef,
        dt: f64,
        multirotor: &ControllerConfig,
        ground: &GroundConfig,
    ) -> Result<Self, ControlError> {
        Ok(match def {
            SharedDef::Multirotor(d) => Controller::Multirotor(MultirotorController::new(d, dt, multirotor)?),
            SharedDef::Wheeled(d) => Controller::Ground(GroundController::new(d, dt, ground)?),
        })
    }

    /// Clear the internal state (integrators, filters) and align it with `vehicle`.
    pub fn reset(&mut self, vehicle: &Vehicle) {
        match (self, vehicle) {
            (Controller::Multirotor(c), Vehicle::Multirotor(v)) => {
                c.reset();
                c.sync_motors(v.motor_speeds());
            }
            (Controller::Ground(c), _) => c.reset(),
            (c, v) => panic!("{} controller for a {} vehicle", c.family(), v.family()),
        }
    }

    pub fn family(&self) -> Family {
        match self {
            Controller::Multirotor(_) => Family::Multirotor,
            Controller::Ground(_) => Family::Wheeled,
        }
    }

    pub fn as_multirotor(&self) -> Option<&MultirotorController> {
        match self {
            Controller::Multirotor(c) => Some(c),
            _ => None,
        }
    }

    pub fn as_ground(&self) -> Option<&GroundController> {
        match self {
            Controller::Ground(c) => Some(c),
            _ => None,
        }
    }
}

/// Normalised actions → commands, for the action mode of a group.
#[derive(Clone, Debug)]
pub enum ActionMapping {
    Multirotor(ActionMap),
    Ground(GroundActionMap),
}

impl ActionMapping {
    /// Map for `mode` on vehicles `def` driven by `controller`; the mode must belong to the
    /// vehicle's family.
    pub fn new(
        mode: AgentActionMode,
        multirotor: &ActionLimits,
        ground: &GroundActionLimits,
        def: &SharedDef,
        controller: &Controller,
    ) -> Result<Self, ControlError> {
        match (mode, def, controller) {
            (AgentActionMode::Multirotor(m), SharedDef::Multirotor(d), Controller::Multirotor(c)) => {
                Ok(ActionMapping::Multirotor(ActionMap::new(m, multirotor.clone(), d, c.max_thrust())))
            }
            (AgentActionMode::Ground(m), SharedDef::Wheeled(d), Controller::Ground(_)) => {
                Ok(ActionMapping::Ground(GroundActionMap::new(m, ground, d)?))
            }
            _ => Err(ControlError::InvalidConfig(format!(
                "action mode {mode} does not drive {} vehicles ({:?})",
                def.family(),
                def.name()
            ))),
        }
    }

    pub fn mode(&self) -> AgentActionMode {
        match self {
            ActionMapping::Multirotor(m) => m.mode().into(),
            ActionMapping::Ground(m) => m.mode().into(),
        }
    }

    /// Action length.
    pub fn dim(&self) -> usize {
        match self {
            ActionMapping::Multirotor(m) => m.dim(),
            ActionMapping::Ground(m) => m.dim(),
        }
    }

    pub fn as_multirotor(&self) -> Option<&ActionMap> {
        match self {
            ActionMapping::Multirotor(m) => Some(m),
            _ => None,
        }
    }

    pub fn as_ground(&self) -> Option<&GroundActionMap> {
        match self {
            ActionMapping::Ground(m) => Some(m),
            _ => None,
        }
    }

    /// Command for `action` (length [`dim`](Self::dim)); `vehicle` anchors relative modes.
    pub fn command(&self, action: &[f64], vehicle: &Vehicle) -> Command {
        match (self, vehicle) {
            (ActionMapping::Multirotor(m), Vehicle::Multirotor(v)) => m.setpoint(action, &StateEstimate::of(v)).into(),
            (ActionMapping::Ground(m), _) => m.setpoint(action).into(),
            (m, v) => panic!("{} action mode for a {} vehicle", m.mode(), v.family()),
        }
    }
}
