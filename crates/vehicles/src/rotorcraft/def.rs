//! Helicopter definition as loaded from TOML (immutable, shared between instances).

use super::rotor::{Rotor, RotorDef};
use crate::VehicleError;
use crate::aero::AeroSurface;
use crate::fixedwing::AirframeDef;
use crate::multirotor::{ColliderDef, ColliderPart, ContactDef};
use autonomousim_core::contact::SphereCollider;
use glam::{DMat3, DVec3};
use serde::{Deserialize, Serialize};

/// A single-main-rotor helicopter: rigid airframe, main rotor with swashplate, tail rotor
/// geared to it, fuselage drag, fins, an engine with a rotor-speed governor and skids. The body
/// frame (FLU) has its origin at the centre of mass.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HelicopterDef {
    pub name: String,
    /// Where the parameters come from.
    #[serde(default)]
    pub source: String,
    pub body: AirframeDef,
    pub main_rotor: RotorMount,
    pub tail_rotor: RotorMount,
    /// Tail rotor speed over main rotor speed.
    pub tail_gear_ratio: f64,
    #[serde(default)]
    pub fuselage: FuselageDef,
    /// Horizontal stabiliser, fin and other fixed surfaces.
    #[serde(default)]
    pub surfaces: Vec<AeroSurface>,
    pub controls: HelicopterControlsDef,
    pub engine: EngineDef,
    /// Skids (`gear`), airframe (`frame`) and rotor-disc (`rotor`) colliders.
    #[serde(default)]
    pub colliders: Vec<ColliderDef>,
    #[serde(default)]
    pub contact: ContactDef,
}

/// A rotor and where it sits.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RotorMount {
    /// Hub (body frame, m).
    pub hub: DVec3,
    /// Shaft axis in the direction of positive thrust (body frame; normalised on load).
    pub axis: DVec3,
    pub rotor: RotorDef,
}

impl RotorMount {
    /// Shaft frame → body frame: z along the axis, x the body x axis (or z when the shaft lies
    /// along x) projected into the disc.
    pub fn frame(&self) -> DMat3 {
        let z = self.axis.normalize_or(DVec3::Z);
        let reference = if z.cross(DVec3::X).length_squared() > 1e-6 { DVec3::X } else { DVec3::Z };
        let x = (reference - z * reference.dot(z)).normalize();
        DMat3::from_cols(x, z.cross(x), z)
    }
}

/// Fuselage drag: `F_i = −½ρ·f_i·v_i·|v|` along each body axis, at `position`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FuselageDef {
    /// Equivalent flat-plate drag areas along body x, y and z (m²).
    pub drag_area: DVec3,
    /// Where the drag acts (body frame, m).
    #[serde(default)]
    pub position: DVec3,
}

/// A blade-pitch channel: a normalised input in [−1, 1] maps linearly onto `[min, max]` (rad),
/// through a servo with first-order lag `tau` (s) and rate limit `rate` (rad/s; 0: none).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PitchChannel {
    pub min: f64,
    pub max: f64,
    #[serde(default)]
    pub rate: f64,
    #[serde(default)]
    pub tau: f64,
}

impl PitchChannel {
    /// Pitch for input `w` (clamped to [−1, 1]).
    pub fn pitch(&self, w: f64) -> f64 {
        let w = if w.is_finite() { w.clamp(-1.0, 1.0) } else { 0.0 };
        0.5 * (self.min + self.max) + 0.5 * w * (self.max - self.min)
    }

    /// Input that gives `pitch` (unclamped).
    pub fn input(&self, pitch: f64) -> f64 {
        (2.0 * pitch - self.min - self.max) / (self.max - self.min)
    }

    /// One servo step from `x` toward `target`.
    pub fn servo(&self, x: f64, target: f64, dt: f64) -> f64 {
        let next = if self.tau > 0.0 { target + (x - target) * (-dt / self.tau).exp() } else { target };
        if self.rate > 0.0 { x + (next - x).clamp(-self.rate * dt, self.rate * dt) } else { next }
    }

    fn validate(&self, name: &str) -> Result<(), String> {
        if !(self.min.is_finite() && self.max > self.min && self.rate >= 0.0 && self.tau >= 0.0) {
            return Err(format!("invalid `{name}` channel {self:?}"));
        }
        Ok(())
    }
}

/// Swashplate and tail pitch. Inputs: collective +1 up; longitudinal cyclic +1 stick forward
/// (disc and nose down); lateral cyclic +1 stick right (roll right); pedal +1 nose right. The
/// cyclic channels give the amplitude of the 1/rev pitch; the model phases it for the main
/// rotor's direction of rotation (90° ahead of the wanted tilt).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HelicopterControlsDef {
    /// Main rotor collective at ¾ radius.
    pub collective: PitchChannel,
    pub longitudinal: PitchChannel,
    pub lateral: PitchChannel,
    /// Tail rotor collective.
    pub pedal: PitchChannel,
}

fn default_engine_lag() -> f64 {
    0.2
}

fn default_governor_bandwidth() -> f64 {
    2.0
}

/// Engine and governor, reduced to the main rotor shaft. The governor is a PI loop on rotor
/// speed tuned from the drive train's inertia for `governor_bandwidth`; the engine follows its
/// torque demand with first-order `lag`, limited to `max_torque` and `max_power`. A freewheel
/// lets the rotor overrun the engine (no negative torque), so it autorotates when the engine
/// stops.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EngineDef {
    /// Largest power delivered to the rotors (W).
    pub max_power: f64,
    /// Governed main rotor speed (rad/s).
    pub rated_speed: f64,
    /// Largest torque at the main rotor shaft (N·m); `1.2·max_power/rated_speed` when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_torque: Option<f64>,
    #[serde(default = "default_engine_lag")]
    pub lag: f64,
    #[serde(default = "default_governor_bandwidth")]
    pub governor_bandwidth: f64,
}

impl EngineDef {
    pub fn max_torque(&self) -> f64 {
        self.max_torque.unwrap_or(1.2 * self.max_power / self.rated_speed)
    }
}

impl HelicopterDef {
    /// Parse a definition without the `type` tag.
    pub fn from_toml(s: &str) -> Result<Self, VehicleError> {
        let mut def: Self = toml::from_str(s)?;
        def.finish()?;
        Ok(def)
    }

    /// Normalise the shaft axes and validate (called by the loaders).
    pub fn finish(&mut self) -> Result<(), VehicleError> {
        for m in [&mut self.main_rotor, &mut self.tail_rotor] {
            m.axis = m.axis.normalize_or(DVec3::Z);
        }
        self.validate()
    }

    pub fn validate(&self) -> Result<(), VehicleError> {
        let fail = |msg: String| Err(VehicleError::Invalid(format!("{}: {msg}", self.name)));
        let b = &self.body;
        let i = b.inertia;
        if !(b.mass > 0.0 && i.min_element() > 0.0 && i.x * i.z > b.product_xz * b.product_xz) {
            return fail("mass and inertia must be positive (definite)".into());
        }
        for (name, m) in [("main_rotor", &self.main_rotor), ("tail_rotor", &self.tail_rotor)] {
            if !(m.hub.is_finite() && m.axis.is_finite() && m.axis.length_squared() > 0.0) {
                return fail(format!("`{name}` needs a finite hub and axis"));
            }
            if let Err(e) = m.rotor.validate() {
                return fail(format!("{name}: {e}"));
            }
        }
        if !(self.tail_gear_ratio.is_finite() && self.tail_gear_ratio > 0.0) {
            return fail("`tail_gear_ratio` must be positive".into());
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
        for (name, ch) in [
            ("collective", &c.collective),
            ("longitudinal", &c.longitudinal),
            ("lateral", &c.lateral),
            ("pedal", &c.pedal),
        ] {
            if let Err(e) = ch.validate(name) {
                return fail(e);
            }
        }
        let e = &self.engine;
        if !(e.max_power > 0.0
            && e.rated_speed > 0.0
            && e.max_torque() > 0.0
            && e.lag >= 0.0
            && e.governor_bandwidth > 0.0)
        {
            return fail("engine power, rated speed, torque and governor bandwidth must be positive".into());
        }
        if self.colliders.iter().any(|c| !c.radius.is_finite() || c.radius <= 0.0 || !c.center.is_finite()) {
            return fail("collider radii must be positive".into());
        }
        Ok(())
    }

    pub fn sphere_colliders(&self) -> Vec<SphereCollider> {
        self.colliders.iter().map(|c| SphereCollider::new(0, c.center, c.radius, c.part as u8)).collect()
    }

    /// Main and tail rotor models.
    pub fn rotors(&self) -> (Rotor, Rotor) {
        let make = |m: &RotorMount| Rotor::new(m.rotor.clone()).expect("validated rotor");
        (make(&self.main_rotor), make(&self.tail_rotor))
    }

    /// Rotating inertia of the drive train at the main rotor shaft (kg·m²).
    pub fn drive_inertia(&self) -> f64 {
        let n = self.tail_gear_ratio;
        self.main_rotor.rotor.polar_inertia() + n * n * self.tail_rotor.rotor.polar_inertia()
    }

    /// Height of the centre of mass above level ground when resting on the skids (m): the
    /// lowest point of the `gear` colliders; `None` without skids.
    pub fn skid_height(&self) -> Option<f64> {
        self.colliders.iter().filter(|c| c.part == ColliderPart::Gear).map(|c| c.radius - c.center.z).reduce(f64::max)
    }
}
