//! The multibody tree of a wheeled vehicle, shared by the simulation and the static solver.
//!
//! Links, in order: the towing unit's chassis (free), its wheels' carrier → knuckle → wheel
//! chains, then each further unit's body (on its joint to the unit ahead) followed by its own
//! wheel chains. Per wheel: carrier (`KcTravel`, if sprung) → steering knuckle (massless
//! revolute about the carrier's z axis, prescribed, if steered) → wheel (revolute about the
//! lateral axis).

use super::def::WheeledDef;
use super::units::UnitJoint;
use autonomousim_core::dynamics::{JointType, MultibodyModel};
use autonomousim_core::math::{Pose, RigidInertia};
use glam::{DMat3, DVec3};
use std::sync::Arc;

/// Links and coordinates of one wheel.
#[derive(Clone, Copy, Debug)]
pub(super) struct CornerLinks {
    pub wheel: usize,
    /// Link whose angular velocity the (non-spinning) carrier has.
    pub carrier: usize,
    /// `(q, v)` offsets of the travel, steering and spin coordinates.
    pub travel: Option<(usize, usize)>,
    pub steer: Option<(usize, usize)>,
    pub spin: (usize, usize),
}

/// Link and joint coordinates of a unit (the towing unit's are the free joint's).
#[derive(Clone, Copy, Debug)]
pub(super) struct UnitLinks {
    pub link: usize,
    pub q: usize,
    pub v: usize,
}

pub(super) struct Tree {
    pub model: MultibodyModel,
    pub corners: Vec<CornerLinks>,
    pub units: Vec<UnitLinks>,
}

/// The tree of `def` (validated), with wheel spin inertias `spin` (one per wheel).
pub(super) fn build(def: &WheeledDef, spin: &[f64]) -> Tree {
    let mut model = MultibodyModel::new();
    let mut corners = Vec::with_capacity(def.num_wheels());
    let mut units: Vec<UnitLinks> = Vec::with_capacity(def.units.len() + 1);
    for u in 0..=def.units.len() {
        let link = if u == 0 {
            let c = &def.chassis;
            let inertia = RigidInertia::new(c.mass, c.com, DMat3::from_diagonal(c.inertia));
            model.add_link("chassis", None, JointType::Free, Pose::IDENTITY, inertia)
        } else {
            let unit = &def.units[u - 1];
            let c = &unit.chassis;
            let joint = match unit.joint {
                UnitJoint::Coupling(_) => JointType::Spherical,
                UnitJoint::Hinge => JointType::Revolute { axis: DVec3::Y },
                UnitJoint::Turntable => JointType::Revolute { axis: DVec3::Z },
            };
            model.add_link(
                unit.name.clone(),
                Some(units[unit.parent].link),
                joint,
                Pose::from_translation(unit.position),
                RigidInertia::new(c.mass, c.com, DMat3::from_diagonal(c.inertia)),
            )
        };
        units.push(UnitLinks { link, q: model.q_offset(link), v: model.v_offset(link) });
        for (w, (axle, side)) in def.wheels().enumerate().filter(|(_, (a, _))| a.unit == u) {
            let (mut parent, mut frame) = (link, Pose::from_translation(def.wheel_position(w)));
            let mut travel = None;
            if let Some(s) = &axle.suspension {
                let table = s.table(side).expect("validated");
                let carrier = model.add_link(
                    format!("carrier_{w}"),
                    Some(parent),
                    JointType::KcTravel(Arc::new(table)),
                    frame,
                    RigidInertia::diag(s.carrier_mass, s.carrier_inertia),
                );
                travel = Some((model.q_offset(carrier), model.v_offset(carrier)));
                (parent, frame) = (carrier, Pose::IDENTITY);
            }
            let carrier = parent;
            let mut steer = None;
            if axle.is_steered() {
                let knuckle = model.add_link(
                    format!("knuckle_{w}"),
                    Some(parent),
                    JointType::Revolute { axis: DVec3::Z },
                    frame,
                    RigidInertia::ZERO,
                );
                model.set_prescribed(knuckle, true);
                steer = Some((model.q_offset(knuckle), model.v_offset(knuckle)));
                (parent, frame) = (knuckle, Pose::IDENTITY);
            }
            let mut moments = axle.wheel.inertia;
            moments.y = spin[w];
            let wheel = model.add_link(
                format!("wheel_{w}"),
                Some(parent),
                JointType::Revolute { axis: DVec3::Y },
                frame,
                RigidInertia::diag(axle.wheel.mass, moments),
            );
            let spin = (model.q_offset(wheel), model.v_offset(wheel));
            corners.push(CornerLinks { wheel, carrier, travel, steer, spin });
        }
    }
    Tree { model, corners, units }
}
