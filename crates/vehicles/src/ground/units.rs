//! Articulated vehicles: trailer definitions, the couplings between units, and composing a
//! towing vehicle with its trailers into one [`WheeledDef`].
//!
//! A composed vehicle is a chain of **units**, each one body of the multibody tree: the towing
//! unit (the definition's chassis) and the units in [`WheeledDef::units`]. A unit hangs from
//! the unit ahead on a joint whose centre is the unit frame's origin (composition moves the
//! trailer's positions there), with axes parallel to the unit ahead at design:
//!
//! - **coupling** (a fifth wheel at the kingpin, or a drawbar eye on a pintle hitch): a ball
//!   joint. Roll works against a torsional spring and damper (stiff for a fifth wheel, light
//!   for an eye); pitch and yaw are free up to end stops.
//! - **hinge**: a revolute joint about the lateral axis (a drawbar on its dolly).
//! - **turntable**: a revolute joint about the vertical axis (a trailer body on its dolly).
//!
//! A trailer file (`type = "trailer"`) gives its coupling point in its own frame. A semitrailer
//! or centre-axle trailer is one unit; a drawbar trailer with a `[dolly]` becomes three: the
//! drawbar (from the eye to its hinge on the dolly), the dolly and the body on its turntable.

use super::def::{AxleDef, ChassisDef, GroundColliderDef, WheeledDef};
use crate::VehicleError;
use glam::{DMat3, DQuat, DVec3, EulerRot};
use serde::{Deserialize, Serialize};

/// How a trailer couples to the unit ahead.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CouplingKind {
    /// Kingpin on a fifth wheel (semitrailers).
    FifthWheel,
    /// Drawbar eye on a pintle or clevis hitch (drawbar and centre-axle trailers).
    Drawbar,
}

/// A coupling point for a trailer at the rear of a unit.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HitchDef {
    pub kind: CouplingKind,
    /// Centre of the fifth wheel's kingpin lock or of the pintle, in the unit's frame (m).
    pub position: DVec3,
}

/// A trailer's coupling as given in its file: the kingpin or eye position in the trailer's
/// frame (the dolly's, for a drawbar trailer with a dolly), and settings that default by kind.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CouplingDef {
    pub kind: CouplingKind,
    pub position: DVec3,
    #[serde(default)]
    pub roll_stiffness: Option<f64>,
    #[serde(default)]
    pub roll_damping: Option<f64>,
    #[serde(default)]
    pub pitch_limit: Option<f64>,
    #[serde(default)]
    pub yaw_limit: Option<f64>,
    #[serde(default)]
    pub stop_stiffness: Option<f64>,
    #[serde(default)]
    pub stop_damping: Option<f64>,
}

/// The ball joint of a coupling with its roll spring and end stops (all resolved).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CouplingJoint {
    pub kind: CouplingKind,
    /// Torsional roll spring (N·m/rad) and damper (N·m·s/rad).
    pub roll_stiffness: f64,
    pub roll_damping: f64,
    /// Free pitch and yaw either way (rad), beyond which stops engage.
    pub pitch_limit: f64,
    pub yaw_limit: f64,
    /// Stop stiffness (N·m/rad) and damping (N·m·s/rad, while engaged).
    pub stop_stiffness: f64,
    pub stop_damping: f64,
}

/// The joint between a unit and the unit ahead.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum UnitJoint {
    Coupling(CouplingJoint),
    /// Revolute about the lateral (y) axis.
    Hinge,
    /// Revolute about the vertical (z) axis.
    Turntable,
}

/// A unit behind the towing unit.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UnitDef {
    pub name: String,
    /// Unit it hangs from (0: the towing unit; always an earlier unit).
    pub parent: usize,
    pub joint: UnitJoint,
    /// Joint centre in the parent unit's frame (m); the origin of this unit's frame.
    pub position: DVec3,
    pub chassis: ChassisDef,
    #[serde(default)]
    pub colliders: Vec<GroundColliderDef>,
    /// Coupling point for a further trailer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hitch: Option<HitchDef>,
}

/// A trailer definition (`type = "trailer"`), coupled behind a towing vehicle by
/// [`WheeledDef::with_trailers`]. Positions are in the trailer's own frame (FLU, any origin).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrailerDef {
    pub name: String,
    #[serde(default)]
    pub source: String,
    /// Kingpin or drawbar eye (in the dolly's frame when there is a dolly).
    pub coupling: CouplingDef,
    /// The trailer body.
    pub chassis: ChassisDef,
    /// The body's axles (unsteered; `unit` is set by the composition).
    pub axles: Vec<AxleDef>,
    #[serde(default)]
    pub colliders: Vec<GroundColliderDef>,
    /// Coupling point for a further trailer, on the body.
    #[serde(default)]
    pub hitch: Option<HitchDef>,
    /// Front dolly of a drawbar trailer: the body sits on its turntable.
    #[serde(default)]
    pub dolly: Option<DollyDef>,
}

/// The dolly of a drawbar trailer, in its own frame.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DollyDef {
    pub chassis: ChassisDef,
    pub axles: Vec<AxleDef>,
    #[serde(default)]
    pub colliders: Vec<GroundColliderDef>,
    /// Hinge of the drawbar on the dolly (m).
    pub hinge: DVec3,
    /// Drawbar mass (kg), a slender bar from the eye to the hinge.
    pub drawbar_mass: f64,
    /// Turntable centre in the dolly's frame, and the same point in the body's frame (m).
    pub turntable: DVec3,
    pub mount: DVec3,
}

impl CouplingDef {
    /// The joint, with defaults by kind: a fifth wheel is stiff in roll (2·10⁷ N·m/rad) and
    /// pitches ±15°; an eye is nearly free in roll (10⁴ N·m/rad) and pitches ±30°; both yaw
    /// ±90° against stops of 10⁷ N·m/rad.
    pub fn joint(&self) -> CouplingJoint {
        let (roll, damping, pitch) = match self.kind {
            CouplingKind::FifthWheel => (2e7, 2e5, 15f64.to_radians()),
            CouplingKind::Drawbar => (1e4, 1e3, 30f64.to_radians()),
        };
        CouplingJoint {
            kind: self.kind,
            roll_stiffness: self.roll_stiffness.unwrap_or(roll),
            roll_damping: self.roll_damping.unwrap_or(damping),
            pitch_limit: self.pitch_limit.unwrap_or(pitch),
            yaw_limit: self.yaw_limit.unwrap_or(90f64.to_radians()),
            stop_stiffness: self.stop_stiffness.unwrap_or(1e7),
            stop_damping: self.stop_damping.unwrap_or(1e5),
        }
    }
}

/// Yaw, pitch and roll of a joint rotation, `rot = R_z(yaw)·R_y(pitch)·R_x(roll)`.
pub fn yaw_pitch_roll(rot: DQuat) -> [f64; 3] {
    let (yaw, pitch, roll) = rot.to_euler(EulerRot::ZYX);
    [yaw, pitch, roll]
}

impl CouplingJoint {
    pub fn validate(&self) -> Result<(), String> {
        let pos = |x: f64| x > 0.0 && x.is_finite();
        let nonneg = |x: f64| x >= 0.0 && x.is_finite();
        let limit = |x: f64| pos(x) && x < std::f64::consts::FRAC_PI_2;
        if !(nonneg(self.roll_stiffness) && nonneg(self.roll_damping) && pos(self.stop_stiffness)) {
            return Err("coupling stiffness and damping must be non-negative (stops positive)".into());
        }
        if !(nonneg(self.stop_damping) && limit(self.pitch_limit) && pos(self.yaw_limit) && self.yaw_limit < 3.0) {
            return Err("coupling limits need 0 < pitch < 90° and 0 < yaw < 172°".into());
        }
        Ok(())
    }

    /// Torque on the unit behind (its coordinates, N·m) at joint rotation `rot` (the unit
    /// behind in the unit ahead) and relative angular velocity `omega` (unit behind's axes):
    /// the roll spring and damper and the pitch and yaw stops, each a generalised force on its
    /// angle `rot = R_z(ψ)·R_y(θ)·R_x(φ)`, mapped to a torque by the Euler-rate Jacobian.
    pub fn torque(&self, rot: DQuat, omega: DVec3) -> DVec3 {
        let [yaw, pitch, roll] = yaw_pitch_roll(rot);
        // Columns: the yaw, pitch and roll axes in the unit behind's coordinates.
        let a = DMat3::from_cols(rot.inverse() * DVec3::Z, DQuat::from_rotation_x(-roll) * DVec3::Y, DVec3::X);
        let rates = a.inverse() * omega;
        let stop = |angle: f64, limit: f64, rate: f64| {
            let over = angle - angle.clamp(-limit, limit);
            if over == 0.0 { 0.0 } else { -self.stop_stiffness * over - self.stop_damping * rate }
        };
        let generalised = DVec3::new(
            stop(yaw, self.yaw_limit, rates.x),
            stop(pitch, self.pitch_limit, rates.y),
            -self.roll_stiffness * roll - self.roll_damping * rates.z,
        );
        a.inverse().transpose() * generalised
    }

    /// Stiffness of the joint against small rotations about the unit behind's axes at `rot`
    /// (N·m/rad; roll spring and engaged stops, for the static solver).
    pub(super) fn stiffness(&self, rot: DQuat) -> DVec3 {
        let [yaw, pitch, _] = yaw_pitch_roll(rot);
        let engaged = |angle: f64, limit: f64| if angle.abs() > limit { self.stop_stiffness } else { 0.0 };
        DVec3::new(self.roll_stiffness, engaged(pitch, self.pitch_limit), engaged(yaw, self.yaw_limit))
    }
}

impl TrailerDef {
    /// Parse a trailer file (`type = "trailer"`, which may be left out).
    pub fn from_toml(s: &str) -> Result<Self, VehicleError> {
        let mut table: toml::Table = toml::from_str(s)?;
        match table.remove("type") {
            None => {}
            Some(toml::Value::String(t)) if t == "trailer" => {}
            Some(t) => return Err(VehicleError::Invalid(format!("not a trailer definition (type = {t})"))),
        }
        Ok(table.try_into()?)
    }

    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self, VehicleError> {
        Self::from_toml(&std::fs::read_to_string(path)?)
    }

    /// Checks before composing (the composed definition validates the rest).
    fn check(&self) -> Result<(), String> {
        if self.axles.is_empty() && self.dolly.as_ref().is_none_or(|d| d.axles.is_empty()) {
            return Err("a trailer needs axles".into());
        }
        let axles = self.axles.iter().chain(self.dolly.iter().flat_map(|d| &d.axles));
        if axles.clone().any(|a| a.is_steered()) {
            return Err("trailer axles cannot steer".into());
        }
        if axles.clone().any(|a| a.unit != 0) {
            return Err("trailer axles take no unit (the composition sets it)".into());
        }
        if let Some(d) = &self.dolly {
            let bar = d.hinge - self.coupling.position;
            if !(d.drawbar_mass > 0.0 && d.drawbar_mass.is_finite()) || bar.x >= 0.0 {
                return Err("the drawbar needs a positive mass and its hinge behind the eye".into());
            }
            if self.coupling.kind != CouplingKind::Drawbar {
                return Err("a trailer with a dolly couples by a drawbar".into());
            }
        }
        self.coupling.joint().validate()
    }
}

/// Move a unit's positions so that its frame's origin is `origin` (given in the old frame).
fn shift(origin: DVec3, chassis: &ChassisDef, axles: &[AxleDef], colliders: &[GroundColliderDef]) -> Shifted {
    let mut chassis = chassis.clone();
    chassis.com -= origin;
    let axles = axles
        .iter()
        .map(|a| {
            let mut a = a.clone();
            a.position -= origin;
            a
        })
        .collect();
    let colliders = colliders
        .iter()
        .map(|c| {
            let mut c = *c;
            c.center -= origin;
            c
        })
        .collect();
    Shifted { chassis, axles, colliders }
}

struct Shifted {
    chassis: ChassisDef,
    axles: Vec<AxleDef>,
    colliders: Vec<GroundColliderDef>,
}

impl WheeledDef {
    /// This vehicle towing `trailers`, each coupled to the hitch of the one ahead (the
    /// towing unit's for the first). The composition keeps this definition's name.
    pub fn with_trailers(&self, trailers: &[TrailerDef]) -> Result<WheeledDef, VehicleError> {
        let mut out = self.clone();
        for t in trailers {
            let fail = |msg: String| VehicleError::Invalid(format!("{} + {}: {msg}", self.name, t.name));
            t.check().map_err(fail)?;
            let ahead = out.units.len();
            let hitch = match ahead {
                0 => out.hitch,
                k => out.units[k - 1].hitch,
            }
            .ok_or_else(|| fail("the unit ahead has no hitch".into()))?;
            if hitch.kind != t.coupling.kind {
                return Err(fail(format!("a {:?} coupling on a {:?} hitch", t.coupling.kind, hitch.kind)));
            }
            let joint = UnitJoint::Coupling(t.coupling.joint());
            let push = |out: &mut WheeledDef, unit: UnitDef, axles: Vec<AxleDef>| {
                let index = out.units.len() + 1;
                out.axles.extend(axles.into_iter().map(|mut a| {
                    a.unit = index;
                    a
                }));
                out.units.push(unit);
            };
            let body_hitch = |origin: DVec3| t.hitch.map(|h| HitchDef { position: h.position - origin, ..h });
            match &t.dolly {
                None => {
                    let origin = t.coupling.position;
                    let s = shift(origin, &t.chassis, &t.axles, &t.colliders);
                    let unit = UnitDef {
                        name: t.name.clone(),
                        parent: ahead,
                        joint,
                        position: hitch.position,
                        chassis: s.chassis,
                        colliders: s.colliders,
                        hitch: body_hitch(origin),
                    };
                    push(&mut out, unit, s.axles);
                }
                Some(d) => {
                    // The drawbar: a slender bar from the eye (its origin) to the hinge.
                    let bar = d.hinge - t.coupling.position;
                    let (m, l) = (d.drawbar_mass, bar.length());
                    let drawbar = UnitDef {
                        name: format!("{}_drawbar", t.name),
                        parent: ahead,
                        joint,
                        position: hitch.position,
                        chassis: ChassisDef {
                            mass: m,
                            com: 0.5 * bar,
                            inertia: DVec3::new(1e-3 * m * l * l, m * l * l / 12.0, m * l * l / 12.0),
                            drag_area: DVec3::ZERO,
                        },
                        colliders: Vec::new(),
                        hitch: None,
                    };
                    push(&mut out, drawbar, Vec::new());
                    let s = shift(d.hinge, &d.chassis, &d.axles, &d.colliders);
                    let dolly = UnitDef {
                        name: format!("{}_dolly", t.name),
                        parent: ahead + 1,
                        joint: UnitJoint::Hinge,
                        position: bar,
                        chassis: s.chassis,
                        colliders: s.colliders,
                        hitch: None,
                    };
                    push(&mut out, dolly, s.axles);
                    let s = shift(d.mount, &t.chassis, &t.axles, &t.colliders);
                    let body = UnitDef {
                        name: t.name.clone(),
                        parent: ahead + 2,
                        joint: UnitJoint::Turntable,
                        position: d.turntable - d.hinge,
                        chassis: s.chassis,
                        colliders: s.colliders,
                        hitch: body_hitch(d.mount),
                    };
                    push(&mut out, body, s.axles);
                }
            }
        }
        out.finish()?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn euler_angles_compose_yaw_pitch_roll() {
        let (y, p, r) = (0.4, -0.2, 0.1);
        let rot = DQuat::from_rotation_z(y) * DQuat::from_rotation_y(p) * DQuat::from_rotation_x(r);
        let [a, b, c] = yaw_pitch_roll(rot);
        assert!((a - y).abs() < 1e-12 && (b - p).abs() < 1e-12 && (c - r).abs() < 1e-12);
    }

    /// The torques are the negative gradient of the stored energy: work over a small rotation.
    #[test]
    fn coupling_torque_is_conservative() {
        let def = CouplingDef {
            kind: CouplingKind::FifthWheel,
            position: DVec3::ZERO,
            roll_stiffness: None,
            roll_damping: None,
            pitch_limit: None,
            yaw_limit: None,
            stop_stiffness: None,
            stop_damping: None,
        };
        let j = def.joint();
        let energy = |rot: DQuat| {
            let [yaw, pitch, roll] = yaw_pitch_roll(rot);
            let over = |x: f64, l: f64| x - x.clamp(-l, l);
            0.5 * j.roll_stiffness * roll * roll
                + 0.5 * j.stop_stiffness * (over(pitch, j.pitch_limit).powi(2) + over(yaw, j.yaw_limit).powi(2))
        };
        let rot = DQuat::from_rotation_z(1.7) * DQuat::from_rotation_y(0.3) * DQuat::from_rotation_x(0.05);
        let tau = j.torque(rot, DVec3::ZERO);
        let h = 1e-6;
        for axis in [DVec3::X, DVec3::Y, DVec3::Z] {
            // A rotation of the unit behind about its own axis: rot · exp(h·axis).
            let du = (energy(rot * DQuat::from_scaled_axis(h * axis))
                - energy(rot * DQuat::from_scaled_axis(-h * axis)))
                / (2.0 * h);
            assert!((tau.dot(axis) + du).abs() < 1e-3 * du.abs().max(1.0), "{axis}: {} vs {}", tau.dot(axis), -du);
        }
    }
}
