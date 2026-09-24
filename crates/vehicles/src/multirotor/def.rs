//! Multirotor definition as loaded from TOML (immutable, shared between instances).

use crate::VehicleError;
use autonomousim_core::contact::{ContactModel, PenaltyParams, SphereCollider};
use glam::DVec3;
use serde::{Deserialize, Serialize};

/// Sea-level ISA air density (kg/m³), the default reference density of rotor coefficients.
pub const SEA_LEVEL_DENSITY: f64 = 1.225;

/// Multirotor of any rotor count and layout; the body frame (FLU) has its origin at the centre
/// of mass.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MultirotorDef {
    pub name: String,
    /// Where the parameters come from.
    #[serde(default)]
    pub source: String,
    pub body: BodyDef,
    /// Parameters shared by all rotors.
    pub rotor: RotorParams,
    pub rotors: Vec<RotorMount>,
    #[serde(default)]
    pub colliders: Vec<ColliderDef>,
    #[serde(default)]
    pub contact: ContactDef,
    #[serde(default)]
    pub battery: Option<BatteryDef>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BodyDef {
    /// Total mass (kg).
    pub mass: f64,
    /// Principal moments of inertia about the centre of mass, body axes (kg·m²).
    pub inertia: DVec3,
    /// Quadratic drag `C_d·A` per body axis (m²): `F_k = −½ρ·C_dA_k·|v_k|·v_k`.
    #[serde(default)]
    pub drag_area: DVec3,
    /// Linear angular damping per body axis (N·m·s/rad).
    #[serde(default)]
    pub angular_damping: DVec3,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RotorParams {
    /// Propeller radius (m), used by the ground-effect model.
    pub radius: f64,
    /// Thrust coefficient `k_T` (N/(rad/s)²) at `reference_density`.
    pub k_thrust: f64,
    /// Drag-torque coefficient `k_Q` (N·m/(rad/s)²) at `reference_density`.
    pub k_torque: f64,
    /// Inertia of propeller and motor rotor about the spin axis (kg·m²).
    #[serde(default)]
    pub inertia: f64,
    /// Rotor (induced) drag coefficient `K_d` (N·s/m per rad/s): in-plane force
    /// `−ω·K_d·v_⊥` at the hub.
    #[serde(default)]
    pub drag: f64,
    /// Rolling-moment coefficient (N·m·s/m per rad/s), RotorS convention.
    #[serde(default)]
    pub rolling_moment: f64,
    /// Idle speed (rad/s) the motor never goes below while armed.
    #[serde(default)]
    pub omega_min: f64,
    /// Maximum speed at full battery (rad/s).
    pub omega_max: f64,
    /// First-order motor time constants (s) when spinning up and down.
    pub tau_up: f64,
    pub tau_down: f64,
    #[serde(default = "default_reference_density")]
    pub reference_density: f64,
}

fn default_reference_density() -> f64 {
    SEA_LEVEL_DENSITY
}

/// Spin direction seen from above (looking along `−axis`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Spin {
    /// Counter-clockwise: angular velocity along `+axis`; reaction torque on the body along `−axis`.
    Ccw,
    Cw,
}

impl Spin {
    /// `+1` for CCW, `−1` for CW.
    pub fn sign(self) -> f64 {
        match self {
            Spin::Ccw => 1.0,
            Spin::Cw => -1.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RotorMount {
    /// Hub position in body coordinates (m).
    pub position: DVec3,
    /// Thrust direction in body coordinates (normalised on load).
    #[serde(default = "default_axis")]
    pub axis: DVec3,
    pub spin: Spin,
}

fn default_axis() -> DVec3 {
    DVec3::Z
}

/// What a collider belongs to (stored as [`SphereCollider::group`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum ColliderPart {
    /// Landing gear: ground contact is expected.
    #[default]
    Gear = 0,
    /// Airframe: contact means a collision.
    Frame = 1,
    /// Propeller disc: contact means a prop strike.
    Rotor = 2,
}

impl ColliderPart {
    pub fn from_group(group: u8) -> Option<Self> {
        match group {
            0 => Some(Self::Gear),
            1 => Some(Self::Frame),
            2 => Some(Self::Rotor),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ColliderDef {
    /// Centre in body coordinates (m).
    pub center: DVec3,
    pub radius: f64,
    #[serde(default)]
    pub part: ColliderPart,
}

/// Contact constants (see `autonomousim_core::contact`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ContactDef {
    /// Mass carried by one collider (kg); default: a quarter of the vehicle mass.
    pub effective_mass: Option<f64>,
    /// Solid contact frequency times the physics step, `ω_c·dt`.
    pub omega_dt: f64,
    pub zeta: f64,
    /// Foliage contact frequency (rad/s) and damping ratio.
    pub foliage_omega: f64,
    pub foliage_zeta: f64,
}

impl Default for ContactDef {
    fn default() -> Self {
        Self { effective_mass: None, omega_dt: 0.2, zeta: 0.8, foliage_omega: 20.0, foliage_zeta: 2.0 }
    }
}

impl ContactDef {
    pub fn model(&self, mass: f64, dt: f64) -> ContactModel {
        let m = self.effective_mass.unwrap_or(0.25 * mass);
        ContactModel {
            solid: PenaltyParams::from_frequency(m, self.omega_dt / dt, self.zeta),
            foliage: PenaltyParams::from_frequency(m, self.foliage_omega, self.foliage_zeta),
        }
    }
}

/// LiPo pack with a linear open-circuit voltage curve and internal resistance. Motor top speed
/// scales with the loaded voltage.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BatteryDef {
    /// Cells in series.
    pub cells: u32,
    /// Capacity (Ah).
    pub capacity: f64,
    /// Pack internal resistance (Ω).
    pub internal_resistance: f64,
    /// Mechanical shaft power / electrical power.
    #[serde(default = "default_efficiency")]
    pub efficiency: f64,
    /// Constant electronics load (W).
    #[serde(default)]
    pub idle_power: f64,
    /// Cell open-circuit voltage when full and empty (V).
    #[serde(default = "default_cell_full")]
    pub cell_full: f64,
    #[serde(default = "default_cell_empty")]
    pub cell_empty: f64,
}

fn default_efficiency() -> f64 {
    0.7
}
fn default_cell_full() -> f64 {
    4.2
}
fn default_cell_empty() -> f64 {
    3.3
}

impl MultirotorDef {
    /// Parse a definition without the `type` tag.
    pub fn from_toml(s: &str) -> Result<Self, VehicleError> {
        let mut def: Self = toml::from_str(s)?;
        def.finish()?;
        Ok(def)
    }

    /// Normalise rotor axes and validate (called by the loaders).
    pub fn finish(&mut self) -> Result<(), VehicleError> {
        for r in &mut self.rotors {
            r.axis = r.axis.normalize_or(DVec3::Z);
        }
        self.validate()
    }

    pub fn validate(&self) -> Result<(), VehicleError> {
        // Written so that NaN fails every check.
        let pos = |x: f64| x > 0.0;
        let fail = |msg: String| Err(VehicleError::Invalid(format!("{}: {msg}", self.name)));
        let b = &self.body;
        if !pos(b.mass) || !pos(b.inertia.min_element()) {
            return fail("mass and inertia must be positive".into());
        }
        if !(b.drag_area.min_element() >= 0.0 && b.angular_damping.min_element() >= 0.0) {
            return fail("drag must be non-negative".into());
        }
        let r = &self.rotor;
        if !(r.radius > 0.0 && r.k_thrust > 0.0 && r.k_torque >= 0.0 && r.inertia >= 0.0 && r.drag >= 0.0) {
            return fail("rotor radius and k_thrust must be positive, other coefficients non-negative".into());
        }
        if !(r.omega_min >= 0.0 && r.omega_max > r.omega_min) {
            return fail("need 0 <= omega_min < omega_max".into());
        }
        if !(r.tau_up > 0.0 && r.tau_down > 0.0 && r.reference_density > 0.0) {
            return fail("motor time constants and reference density must be positive".into());
        }
        if self.rotors.is_empty() || self.rotors.len() > super::MAX_ROTORS {
            return fail(format!("need 1..={} rotors", super::MAX_ROTORS));
        }
        if self.rotors.iter().any(|m| !m.position.is_finite() || m.axis.length_squared() < 0.5) {
            return fail("rotor positions must be finite and axes non-zero".into());
        }
        if self.colliders.iter().any(|c| !pos(c.radius) || !c.center.is_finite()) {
            return fail("collider radii must be positive".into());
        }
        if let Some(bat) = &self.battery
            && !(bat.cells > 0
                && bat.capacity > 0.0
                && bat.internal_resistance >= 0.0
                && bat.efficiency > 0.0
                && bat.efficiency <= 1.0
                && bat.cell_full > bat.cell_empty
                && bat.cell_empty > 0.0)
        {
            return fail("invalid battery parameters".into());
        }
        Ok(())
    }

    /// Sphere colliders attached to the body link.
    pub fn sphere_colliders(&self) -> Vec<SphereCollider> {
        self.colliders
            .iter()
            .map(|c| SphereCollider { link: 0, center: c.center, radius: c.radius, group: c.part as u8 })
            .collect()
    }

    /// Uniform rotor speed that balances `gravity` (m/s²) at air density `density`, ignoring
    /// ground effect and battery sag.
    pub fn hover_omega(&self, gravity: f64, density: f64) -> f64 {
        let lift: f64 = self.rotors.iter().map(|m| m.axis.z).sum::<f64>().max(1e-9);
        let k = self.rotor.k_thrust * density / self.rotor.reference_density;
        (self.body.mass * gravity / (k * lift)).sqrt()
    }

    /// Maximum total thrust over weight at sea level.
    pub fn thrust_to_weight(&self, gravity: f64) -> f64 {
        let lift: f64 = self.rotors.iter().map(|m| m.axis.z).sum();
        self.rotor.k_thrust * self.rotor.omega_max.powi(2) * lift / (self.body.mass * gravity)
    }
}
