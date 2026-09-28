//! Tiltrotor definition as loaded from TOML (immutable, shared between instances).

use crate::VehicleError;
use crate::aero::AeroSurface;
use crate::fixedwing::{AirframeDef, ElectricMotorDef, EngineDef, PropellerDef, Propulsion, PropulsionDef};
use crate::multirotor::{ColliderDef, ColliderPart, ContactDef};
use crate::rotorcraft::{FuselageDef, PitchChannel};
use autonomousim_core::contact::SphereCollider;
use glam::DVec3;
use serde::{Deserialize, Serialize};

/// Most rotors a tiltrotor may have (the size of the per-rotor input arrays).
pub const MAX_ROTORS: usize = 4;

/// A tiltrotor: a rigid airframe with lifting surfaces (wing panels, tail) whose control
/// surfaces are mixed from aileron, elevator and rudder channels, fuselage drag, and up to
/// [`MAX_ROTORS`] fixed-pitch propellers on electric motors, each on a mount that tilts about
/// the body y axis from pointing up (tilt 0) to pointing forward (tilt π/2). The body frame
/// (FLU) has its origin at the centre of mass.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TiltrotorDef {
    pub name: String,
    /// Where the parameters come from.
    #[serde(default)]
    pub source: String,
    pub body: AirframeDef,
    #[serde(default)]
    pub fuselage: FuselageDef,
    /// Wing panels, tailplane, fin.
    #[serde(default)]
    pub surfaces: Vec<AeroSurface>,
    pub controls: TiltrotorControlsDef,
    pub rotors: Vec<TiltMount>,
    /// Propeller and motor shared by all rotors (the motor needs its supply `voltage`).
    pub propeller: PropellerDef,
    pub motor: ElectricMotorDef,
    /// Rotor (induced) drag coefficient `K_d` (N·s/m per rad/s): in-plane force `−ω·K_d·v_⊥`
    /// at the hub, from the flow across the disc in edgewise flight.
    #[serde(default)]
    pub rotor_drag: f64,
    /// Gear (`gear`), airframe (`frame`) and rotor (`rotor`) colliders.
    #[serde(default)]
    pub colliders: Vec<ColliderDef>,
    #[serde(default)]
    pub contact: ContactDef,
}

/// A rotor on a mount that tilts about the body y axis through `pivot`. At tilt `τ` the thrust
/// axis is `(sin τ, 0, cos τ)` and the hub sits `offset` along it from the pivot.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TiltMount {
    #[serde(default)]
    pub name: String,
    /// Tilt pivot (body frame, m).
    pub pivot: DVec3,
    /// Hub distance from the pivot along the thrust axis (m).
    #[serde(default)]
    pub offset: f64,
    /// `+1` when the propeller's angular velocity points along the thrust axis
    /// (counter-clockwise seen from above in hover), `−1` otherwise.
    pub sense: f64,
    /// Whether the mount follows the tilt commands (within the `tilt` channel's range); a
    /// fixed mount stays at `tilt`.
    #[serde(default = "yes")]
    pub tilting: bool,
    /// Tilt of a fixed mount (rad).
    #[serde(default)]
    pub tilt: f64,
}

fn yes() -> bool {
    true
}

/// Control channels. Aileron +1 rolls right, elevator +1 pitches the nose up, rudder +1 yaws
/// the nose right: each maps [−1, 1] onto its deflection range (rad) through a servo, and each
/// surface with a flap is deflected by the `mixing` gains times the channel deflections. The
/// `tilt` channel gives the range of the tilting mounts (rad; commands are clamped into it)
/// and their servo.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TiltrotorControlsDef {
    pub aileron: PitchChannel,
    pub elevator: PitchChannel,
    pub rudder: PitchChannel,
    pub tilt: PitchChannel,
    #[serde(default)]
    pub mixing: Vec<SurfaceMix>,
}

impl TiltrotorControlsDef {
    pub fn surface_channels(&self) -> [&PitchChannel; 3] {
        [&self.aileron, &self.elevator, &self.rudder]
    }
}

/// Deflection of surface `surface` (by name): `aileron·δ_a + elevator·δ_e + rudder·δ_r`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SurfaceMix {
    pub surface: String,
    #[serde(default)]
    pub aileron: f64,
    #[serde(default)]
    pub elevator: f64,
    #[serde(default)]
    pub rudder: f64,
}

impl TiltMount {
    /// Thrust axis at tilt `tilt` (body frame).
    pub fn axis(tilt: f64) -> DVec3 {
        let (s, c) = tilt.sin_cos();
        DVec3::new(s, 0.0, c)
    }

    /// Hub at tilt `tilt` (body frame).
    pub fn hub(&self, tilt: f64) -> DVec3 {
        self.pivot + Self::axis(tilt) * self.offset
    }
}

impl TiltrotorDef {
    /// Parse a definition without the `type` tag.
    pub fn from_toml(s: &str) -> Result<Self, VehicleError> {
        let mut def: Self = toml::from_str(s)?;
        def.finish()?;
        Ok(def)
    }

    /// Validate (called by the loaders).
    pub fn finish(&mut self) -> Result<(), VehicleError> {
        self.validate()
    }

    pub fn validate(&self) -> Result<(), VehicleError> {
        let fail = |msg: String| Err(VehicleError::Invalid(format!("{}: {msg}", self.name)));
        let b = &self.body;
        let i = b.inertia;
        if !(b.mass > 0.0 && i.min_element() > 0.0 && i.x * i.z > b.product_xz * b.product_xz) {
            return fail("mass and inertia must be positive (definite)".into());
        }
        if !(self.fuselage.drag_area.is_finite() && self.fuselage.drag_area.min_element() >= 0.0) {
            return fail("fuselage drag areas must be non-negative".into());
        }
        for s in &self.surfaces {
            if let Err(e) = s.validate() {
                return fail(e.to_string());
            }
        }
        let c = &self.controls;
        for (name, ch) in [("aileron", &c.aileron), ("elevator", &c.elevator), ("rudder", &c.rudder), ("tilt", &c.tilt)]
        {
            if let Err(e) = ch.validate(name) {
                return fail(e);
            }
        }
        for m in &c.mixing {
            if !self.surfaces.iter().any(|s| s.name == m.surface && s.flap.is_some()) {
                return fail(format!("mixing names {:?}, which is no surface with a flap", m.surface));
            }
            if ![m.aileron, m.elevator, m.rudder].iter().all(|g| g.is_finite()) {
                return fail(format!("mixing gains of {:?} must be finite", m.surface));
            }
        }
        if self.rotors.is_empty() || self.rotors.len() > MAX_ROTORS {
            return fail(format!("a tiltrotor needs 1 to {MAX_ROTORS} rotors"));
        }
        for r in &self.rotors {
            if !(r.pivot.is_finite()
                && r.offset.is_finite()
                && r.tilt.is_finite()
                && (r.sense.abs() - 1.0).abs() < 1e-12)
            {
                return fail(format!("rotor {:?} needs a finite pivot, offset and tilt and a sense of ±1", r.name));
            }
        }
        if self.motor.voltage.is_none() {
            return fail("the motor needs a supply voltage".into());
        }
        if let Err(e) = self.propulsion_def(DVec3::ZERO, 1.0).validate() {
            return fail(e);
        }
        if !(self.rotor_drag.is_finite() && self.rotor_drag >= 0.0) {
            return fail("`rotor_drag` must be non-negative".into());
        }
        if self.colliders.iter().any(|c| !c.radius.is_finite() || c.radius <= 0.0 || !c.center.is_finite()) {
            return fail("collider radii must be positive".into());
        }
        Ok(())
    }

    fn propulsion_def(&self, position: DVec3, sense: f64) -> PropulsionDef {
        PropulsionDef {
            position,
            axis: DVec3::Z,
            sense,
            propeller: self.propeller.clone(),
            engine: EngineDef::Electric(self.motor.clone()),
        }
    }

    /// Propeller and motor model (its axis and position unused: the mounts give them).
    pub fn propulsion(&self) -> Propulsion {
        Propulsion::new(&self.propulsion_def(DVec3::ZERO, 1.0))
    }

    /// Mixing gains `[aileron, elevator, rudder]` of each surface (zero when unmixed).
    pub fn mixing_gains(&self) -> Vec<[f64; 3]> {
        self.surfaces
            .iter()
            .map(|s| {
                self.controls
                    .mixing
                    .iter()
                    .filter(|m| m.surface == s.name)
                    .fold([0.0; 3], |g, m| [g[0] + m.aileron, g[1] + m.elevator, g[2] + m.rudder])
            })
            .collect()
    }

    /// Tilt of rotor `k` for a commanded tilt (clamped to the channel's range; a fixed mount
    /// keeps its own).
    pub fn mount_tilt(&self, k: usize, command: f64) -> f64 {
        let r = &self.rotors[k];
        let t = &self.controls.tilt;
        if r.tilting {
            if command.is_finite() { command.clamp(t.min, t.max) } else { r.tilt.clamp(t.min, t.max) }
        } else {
            r.tilt
        }
    }

    /// Span of the horizontal surfaces (m): twice the farthest tip from the centre line.
    pub fn span(&self) -> f64 {
        let tip = |s: &AeroSurface| s.position.y.abs() + 0.5 * s.span;
        2.0 * self.surfaces.iter().filter(|s| s.roll.cos().abs() > 0.5).map(tip).fold(0.0, f64::max)
    }

    /// Wing-borne stall speed (m/s) in air of density `rho` under gravity `g`, surfaces
    /// centred: the speed at which the horizontal surfaces' greatest lift (over body angles of
    /// attack 0–40°, in 0.1° steps) carries the weight.
    pub fn stall_speed(&self, rho: f64, g: f64) -> f64 {
        let lift = |alpha: f64| -> f64 {
            self.surfaces
                .iter()
                .filter(|s| s.roll.cos().abs() > 0.5)
                .map(|s| s.area * s.roll.cos().signum() * s.coefficients(alpha + s.incidence, 0.0).cl)
                .sum()
        };
        let best = (0..=400).map(|i| lift((i as f64 * 0.1).to_radians())).fold(0.0, f64::max);
        (2.0 * self.body.mass * g / (rho * best.max(1e-6))).sqrt()
    }

    /// Propeller disc radius (m).
    pub fn rotor_radius(&self) -> f64 {
        0.5 * self.propeller.diameter
    }

    pub fn sphere_colliders(&self) -> Vec<SphereCollider> {
        self.colliders.iter().map(|c| SphereCollider::new(0, c.center, c.radius, c.part as u8)).collect()
    }

    /// Height of the centre of mass above level ground when resting on the gear (m); `None`
    /// without gear colliders.
    pub fn gear_height(&self) -> Option<f64> {
        self.colliders.iter().filter(|c| c.part == ColliderPart::Gear).map(|c| c.radius - c.center.z).reduce(f64::max)
    }
}
