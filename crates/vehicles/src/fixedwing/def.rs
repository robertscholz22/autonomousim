//! Fixed-wing aircraft definition as loaded from TOML (immutable, shared between instances).

use super::aero::{AeroModel, Geometry};
use super::gear::GearDef;
use super::propulsion::PropulsionDef;
use crate::VehicleError;
use crate::multirotor::{BatteryDef, ColliderDef, ColliderPart, ContactDef};
use autonomousim_core::contact::SphereCollider;
use autonomousim_core::math::RigidInertia;
use glam::{DMat3, DQuat, DVec3};
use serde::{Deserialize, Serialize};

/// A rigid airframe with a whole-aircraft aerodynamic model, one propeller, control surfaces,
/// landing gear and colliders. The body frame (FLU) has its origin at the centre of mass.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FixedWingDef {
    pub name: String,
    /// Where the parameters come from.
    #[serde(default)]
    pub source: String,
    pub body: AirframeDef,
    pub geometry: Geometry,
    pub aero: AeroModel,
    #[serde(default)]
    pub controls: ControlsDef,
    pub propulsion: PropulsionDef,
    #[serde(default)]
    pub gear: Vec<GearDef>,
    /// Airframe colliders (penalty contacts; the gear wheels are added as `gear` colliders
    /// that only other agents and water see).
    #[serde(default)]
    pub colliders: Vec<ColliderDef>,
    #[serde(default)]
    pub contact: ContactDef,
    /// Battery of an electric motor (the motor's fixed `voltage` otherwise).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub battery: Option<BatteryDef>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AirframeDef {
    /// Total mass (kg).
    pub mass: f64,
    /// Moments of inertia about the body axes through the centre of mass (kg·m²).
    pub inertia: DVec3,
    /// Product of inertia `J_xz = ∫xz dm` in FRD axes (Beard & McLain's sign; kg·m²).
    #[serde(default)]
    pub product_xz: f64,
}

impl AirframeDef {
    /// Inertia matrix in the FLU body frame (the FRD product changes sign twice).
    pub fn inertia_matrix(&self) -> DMat3 {
        let [jx, jy, jz] = self.inertia.to_array();
        let jxz = self.product_xz;
        DMat3::from_cols(DVec3::new(jx, 0.0, jxz), DVec3::new(0.0, jy, 0.0), DVec3::new(jxz, 0.0, jz))
    }

    pub fn rigid_inertia(&self) -> RigidInertia {
        RigidInertia::new(self.mass, DVec3::ZERO, self.inertia_matrix())
    }
}

/// One control surface channel: deflection range and servo.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SurfaceDef {
    /// Largest positive deflection (rad; trailing edge down for elevator and flap).
    pub max: f64,
    /// Largest negative deflection (rad, ≤ 0; `−max` if absent).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min: Option<f64>,
    /// Servo rate limit (rad/s; 0: none).
    #[serde(default)]
    pub rate: f64,
    /// Servo first-order time constant (s; 0: none).
    #[serde(default)]
    pub tau: f64,
}

impl SurfaceDef {
    fn symmetric(max: f64) -> Self {
        Self { max, min: None, rate: 0.0, tau: 0.0 }
    }

    /// Deflection for a signed fraction `w` of the range (−1 at `min`, +1 at `max`).
    pub fn deflection(&self, w: f64) -> f64 {
        let w = w.clamp(-1.0, 1.0);
        if w >= 0.0 { w * self.max } else { -w * self.min.unwrap_or(-self.max) }
    }

    fn validate(&self) -> Result<(), String> {
        if !(self.max >= 0.0 && self.min.is_none_or(|m| m <= 0.0) && self.rate >= 0.0 && self.tau >= 0.0) {
            return Err(format!("invalid control surface {self:?}"));
        }
        Ok(())
    }
}

/// Control surfaces. Normalised commands: aileron +1 rolls right, elevator +1 pitches the nose
/// up, rudder +1 yaws the nose right (and steers the nose wheel right); flap 0–1 deploys the
/// flaps from 0 to `flap.max`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ControlsDef {
    pub aileron: SurfaceDef,
    pub elevator: SurfaceDef,
    pub rudder: SurfaceDef,
    /// No flaps unless given.
    #[serde(default = "no_flap")]
    pub flap: SurfaceDef,
}

fn no_flap() -> SurfaceDef {
    SurfaceDef::symmetric(0.0)
}

impl Default for ControlsDef {
    /// ±25° surfaces with a 20 ms, 300°/s servo; no flaps.
    fn default() -> Self {
        let s = SurfaceDef { max: 25f64.to_radians(), min: None, rate: 300f64.to_radians(), tau: 0.02 };
        Self { aileron: s.clone(), elevator: s.clone(), rudder: s, flap: no_flap() }
    }
}

impl ControlsDef {
    pub fn surfaces(&self) -> [&SurfaceDef; 4] {
        [&self.aileron, &self.elevator, &self.rudder, &self.flap]
    }
}

impl FixedWingDef {
    /// Parse a definition without the `type` tag.
    pub fn from_toml(s: &str) -> Result<Self, VehicleError> {
        let mut def: Self = toml::from_str(s)?;
        def.finish()?;
        Ok(def)
    }

    /// Normalise the propeller axis and validate (called by the loaders).
    pub fn finish(&mut self) -> Result<(), VehicleError> {
        self.propulsion.axis = self.propulsion.axis.normalize_or(DVec3::X);
        self.validate()
    }

    pub fn validate(&self) -> Result<(), VehicleError> {
        let fail = |msg: String| Err(VehicleError::Invalid(format!("{}: {msg}", self.name)));
        let b = &self.body;
        let i = b.inertia;
        // Positive definite: positive moments and J_x·J_z > J_xz².
        if !(b.mass > 0.0 && i.min_element() > 0.0 && i.x * i.z > b.product_xz * b.product_xz) {
            return fail("mass and inertia must be positive (definite)".into());
        }
        let g = &self.geometry;
        if !(g.area > 0.0 && g.span > 0.0 && g.chord > 0.0 && g.aero_reference.is_finite()) {
            return fail("wing area, span and chord must be positive".into());
        }
        if let Err(e) = self.aero.validate() {
            return fail(e);
        }
        for s in self.controls.surfaces() {
            if let Err(e) = s.validate() {
                return fail(e);
            }
        }
        if let Err(e) = self.propulsion.validate() {
            return fail(e);
        }
        if let super::propulsion::EngineDef::Electric(e) = &self.propulsion.engine
            && e.voltage.is_none()
            && self.battery.is_none()
        {
            return fail("an electric motor needs a voltage or a battery".into());
        }
        for gear in &self.gear {
            if let Err(e) = gear.validate() {
                return fail(e);
            }
        }
        if self.colliders.iter().any(|c| !c.radius.is_finite() || c.radius <= 0.0 || !c.center.is_finite()) {
            return fail("collider radii must be positive".into());
        }
        Ok(())
    }

    /// Airframe colliders, then one `gear` sphere per wheel resting on its contact point.
    pub fn sphere_colliders(&self) -> Vec<SphereCollider> {
        let frame = self.colliders.iter().map(|c| SphereCollider::new(0, c.center, c.radius, c.part as u8));
        let wheels = self.gear.iter().map(|g| {
            SphereCollider::new(0, g.position + DVec3::Z * g.wheel_radius, g.wheel_radius, ColliderPart::Gear as u8)
        });
        frame.chain(wheels).collect()
    }

    /// Stall angles of attack `(positive, negative)` (rad): where lift peaks.
    pub fn stall_angles(&self) -> (f64, f64) {
        self.aero.stall_angles(&self.geometry)
    }

    /// Straight-flight stall speed (m/s) in air of density `rho` under gravity `g`, flaps up:
    /// the speed at which the lift at the stall angle carries the weight.
    pub fn stall_speed(&self, rho: f64, g: f64) -> f64 {
        let cl_max = self.aero.lift_at(&self.geometry, self.stall_angles().0).max(0.1);
        (2.0 * self.body.mass * g / (rho * self.geometry.area * cl_max)).sqrt()
    }

    /// Deflection signs of aileron, elevator and rudder for the normalised commands.
    pub fn control_signs(&self) -> [f64; 3] {
        self.aero.control_signs(&self.geometry)
    }

    /// Attitude (a pitch rotation) and height of the centre of mass above level ground when
    /// standing on the gear: the front-most and rear-most wheels level, the struts at the
    /// average static compression. `None` without gear.
    pub fn resting_pose(&self, gravity: f64) -> Option<(DQuat, f64)> {
        let by_x = |a: &&GearDef, b: &&GearDef| a.position.x.total_cmp(&b.position.x);
        let front = self.gear.iter().max_by(by_x)?.position;
        let rear = self.gear.iter().min_by(by_x)?.position;
        let d = front - rear;
        let rot = if d.x > 1e-6 { DQuat::from_rotation_y(d.z.atan2(d.x)) } else { DQuat::IDENTITY };
        let k: f64 = self.gear.iter().map(|g| g.spring).sum();
        let sag = self.body.mass * gravity / k;
        let lowest = self.gear.iter().map(|g| (rot * g.position).z).fold(f64::INFINITY, f64::min);
        Some((rot, -lowest - sag))
    }
}
