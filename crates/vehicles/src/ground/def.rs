//! Wheeled vehicle definition as loaded from TOML (`type = "wheeled"`), shared between
//! instances.
//!
//! The chassis frame is FLU with an arbitrary origin; wheel positions, the centre of mass and
//! colliders are given in it. Axles list the **left** wheel; the right wheel mirrors it
//! (`y → −y`, and so do the suspension's lateral offset, toe and camber). Wheels are numbered
//! axle by axle, left before right (`2·axle + side`).
//!
//! Each wheel hangs from the chassis on a kinematics table over its travel (`KcTravel` joint,
//! travel positive in bump) or rigidly, may steer about the carrier's vertical axis (a
//! prescribed joint) and spins about the carrier's lateral axis. Wheel positions are the
//! **design** positions: at zero travel, which is the static ride height when the spring
//! preload is left to the definition (`preload` omitted), since the preload is then computed
//! to carry the static load.
//!
//! Trailers and other units behind the towing unit are in [`WheeledDef::units`] (see
//! [`super::units`]); their axles follow the towing unit's, unit by unit, with positions in
//! their unit's frame.

use super::powertrain::{MAX_WHEELS, PowertrainDef};
use super::tire::{FialaParams, MfParams, TirFile, Tire};
use super::units::{HitchDef, UnitDef, UnitJoint};
use crate::VehicleError;
use crate::multirotor::ContactDef;
use autonomousim_core::contact::SphereCollider;
use autonomousim_core::dynamics::{KcTable, KcTableSpec};
use autonomousim_core::math::spline::CubicSpline;
use glam::{DQuat, DVec3};
use serde::{Deserialize, Serialize};

/// Standard gravity, used for the automatic spring preloads.
pub const STANDARD_GRAVITY: f64 = 9.80665;

/// Built-in `.tir` files (`assets/tires`), by file stem.
const TIR_FILES: &[(&str, &str)] = &[
    ("Sedan_Pac02Tire", include_str!("../../../../assets/tires/Sedan_Pac02Tire.tir")),
    ("HMMWV_Pac02Tire", include_str!("../../../../assets/tires/HMMWV_Pac02Tire.tir")),
    ("Truck_Pac02Tire", include_str!("../../../../assets/tires/Truck_Pac02Tire.tir")),
];

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WheeledDef {
    pub name: String,
    /// Where the parameters come from.
    #[serde(default)]
    pub source: String,
    pub chassis: ChassisDef,
    pub axles: Vec<AxleDef>,
    #[serde(default)]
    pub steering: Option<SteeringDef>,
    pub powertrain: PowertrainDef,
    #[serde(default)]
    pub colliders: Vec<GroundColliderDef>,
    #[serde(default)]
    pub contact: ContactDef,
    /// Coupling point for a trailer at the rear of the towing unit (fifth wheel or pintle).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hitch: Option<HitchDef>,
    /// Units behind the towing unit (trailers, dollies and drawbars; unit `k + 1` is
    /// `units[k]`), usually added by [`with_trailers`](Self::with_trailers).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub units: Vec<UnitDef>,
    /// Resolved per axle by [`finish`](Self::finish).
    #[serde(skip)]
    tires: Vec<Tire>,
    /// Spring preload per wheel (N), given or computed.
    #[serde(skip)]
    preload: Vec<f64>,
    /// Static equilibrium under standard gravity, when there is one.
    #[serde(skip)]
    rest: Option<Box<StaticState>>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChassisDef {
    /// Sprung mass (kg): everything except the wheel carriers and wheels.
    pub mass: f64,
    /// Centre of mass in the chassis frame (m).
    pub com: DVec3,
    /// Principal moments of inertia about the centre of mass, chassis axes (kg·m²).
    pub inertia: DVec3,
    /// Quadratic drag `C_d·A` per chassis axis (m²): `F_k = −½ρ·C_dA_k·|v_k|·v_k` at the
    /// centre of mass.
    #[serde(default)]
    pub drag_area: DVec3,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AxleDef {
    #[serde(default)]
    pub name: String,
    /// Unit carrying the axle (0: the towing unit); axles are listed unit by unit.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub unit: usize,
    /// Left wheel centre at design ride height, chassis frame (m).
    pub position: DVec3,
    /// Independent suspension; `None` mounts the wheels rigidly (small robots).
    #[serde(default)]
    pub suspension: Option<SuspensionDef>,
    pub wheel: WheelDef,
    pub tire: TireSpec,
    /// Dual wheels: two tyres side by side at this centre distance (m), `position` being the
    /// middle of the pair.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dual: Option<f64>,
    /// Share of the steering angle (1 for a steered front axle, 0 unsteered, negative for
    /// counter-steering rear axles), or `"geometric"`: about the turn centre on the virtual
    /// rear axle (see [`Steer`]).
    #[serde(default)]
    pub steer: Steer,
    /// How the axle's wheels steer: from the steering command through the Ackermann geometry
    /// (with share `steer`), each wheel independently from its per-wheel command (falling
    /// back to the Ackermann angle without one), or from the unit's articulation angle.
    #[serde(default)]
    pub steer_mode: SteerMode,
    pub brake: BrakeDef,
    /// `steer` resolved to a share (by [`WheeledDef::finish`]).
    #[serde(skip)]
    share: f64,
}

/// Steering share of an axle.
///
/// A share `s` turns the axle's bicycle angle to `s·δ`. `Geometric` turns it about the turn
/// centre of the lead axle (the first steered axle with a share) on the virtual rear axle `x_r`
/// (the mean of the unit's unsteered axles): `tan δ_i = (x_i − x_r) tan δ_l / (x_l − x_r)`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Steer {
    Share(f64),
    Named(SteerName),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SteerName {
    Geometric,
}

impl Default for Steer {
    fn default() -> Self {
        Steer::Share(0.0)
    }
}

impl Steer {
    pub const GEOMETRIC: Self = Steer::Named(SteerName::Geometric);
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WheelDef {
    /// Rotating mass: rim and tyre (kg).
    pub mass: f64,
    /// Principal moments of inertia (kg·m²); `y` is the spin axis and includes the axle shaft.
    pub inertia: DVec3,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BrakeDef {
    /// Service brake torque at full pedal (N·m).
    pub max_torque: f64,
    /// Parking brake torque (N·m).
    #[serde(default)]
    pub parking_torque: f64,
    /// Air brakes: dead time (s) and first-order time constant (s) from the pedal to the
    /// service brake torque (default instant). The parking brake acts at once.
    #[serde(default, skip_serializing_if = "is_zero_f64")]
    pub delay: f64,
    #[serde(default, skip_serializing_if = "is_zero_f64")]
    pub time_constant: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SuspensionDef {
    /// Carrier pose over travel for the left wheel, relative to the design position (empty:
    /// a straight vertical guide).
    #[serde(default)]
    pub kinematics: KcTableSpec,
    /// Non-rotating unsprung mass at the wheel centre (upright, hub, share of the links; kg)
    /// and its principal moments of inertia (kg·m²).
    pub carrier_mass: f64,
    #[serde(default)]
    pub carrier_inertia: DVec3,
    pub spring: SpringDef,
    pub damper: DamperDef,
    #[serde(default)]
    pub bump_stop: Option<StopDef>,
    #[serde(default)]
    pub rebound_stop: Option<StopDef>,
    /// Anti-roll bar rate at the wheel (N/m of travel difference). Negative values soften
    /// roll, down to `−rate/2` (no roll stiffness, a pendulum axle): a solid axle with springs
    /// at spread `s` and track `t` is `rate·((s/t)² − 1)/2`.
    #[serde(default)]
    pub anti_roll: f64,
}

/// Suspension spring at the wheel: `force(s)` pushes the wheel away from the body. Either a
/// rate with a preload (omitted: carries the static load at zero travel), or a table.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpringDef {
    /// Wheel rate (N/m).
    #[serde(default)]
    pub rate: f64,
    /// Force at zero travel (N).
    #[serde(default)]
    pub preload: Option<f64>,
    /// Tabulated force (N) over travel (m), interpolated by a natural cubic spline and
    /// continued linearly.
    #[serde(default)]
    pub travel: Vec<f64>,
    #[serde(default)]
    pub force: Vec<f64>,
    #[serde(skip)]
    table: Option<CubicSpline>,
}

/// Damper at the wheel (N·s/m), in bump (compression) and rebound, optionally degressive:
/// `F = c·v / (1 + d·|v|)` with degressivity `d` (s/m).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DamperDef {
    pub bump: f64,
    pub rebound: f64,
    #[serde(default, skip_serializing_if = "is_zero_f64")]
    pub degressivity_bump: f64,
    #[serde(default, skip_serializing_if = "is_zero_f64")]
    pub degressivity_rebound: f64,
}

/// Linear end stop engaging at `travel` (bump: positive, rebound: negative).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StopDef {
    pub travel: f64,
    pub stiffness: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SteerMode {
    #[default]
    Ackermann,
    /// Per-wheel angle commands up to `max_angle` (rad), rate-limited to `rate` (rad/s).
    Independent { max_angle: f64, rate: f64 },
    /// Forced steering of a trailer axle: `k` × the unit's articulation angle (see
    /// `Wheeled::articulation`), so the axle steers against the turn and the trailer's rear
    /// follows the towing unit's path more closely.
    Articulation(f64),
}

impl AxleDef {
    /// Whether the axle's wheels have a steering joint.
    pub fn is_steered(&self) -> bool {
        self.steer != Steer::Share(0.0) || !matches!(self.steer_mode, SteerMode::Ackermann)
    }

    /// Steering share (after [`WheeledDef::finish`]; see [`Steer`]).
    pub fn share(&self) -> f64 {
        self.share
    }

    /// Whether the axle steers about the lead axle's turn centre.
    pub fn is_geometric(&self) -> bool {
        self.steer == Steer::GEOMETRIC
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SteeringDef {
    /// Steering angle of the equivalent bicycle at full lock (rad).
    pub max_angle: f64,
    /// Rate limit of that angle (rad/s).
    pub rate: f64,
    /// Ackermann fraction: 0 parallel steer, 1 full Ackermann about the unsteered axles.
    #[serde(default)]
    pub ackermann: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroundColliderDef {
    /// Centre in the chassis frame (m).
    pub center: DVec3,
    pub radius: f64,
    /// Friction scale (0 for a frictionless skid or caster ball).
    #[serde(default = "one")]
    pub friction: f64,
    #[serde(default)]
    pub part: GroundPart,
}

fn one() -> f64 {
    1.0
}

fn is_zero(x: &usize) -> bool {
    *x == 0
}

fn is_zero_f64(x: &f64) -> bool {
    *x == 0.0
}

/// Role of a chassis collider for event classification.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GroundPart {
    /// Body: contact means a collision.
    #[default]
    Body = 0,
    /// Skid or caster: ground contact is expected.
    Skid = 1,
}

/// Tyre of an axle.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum TireSpec {
    /// Magic Formula `.tir` file: a built-in name (`assets/tires`, without extension) or a path.
    Tir {
        file: String,
        /// Inflation pressure (Pa; MF 6.x files), default the file's.
        #[serde(default)]
        pressure: Option<f64>,
    },
    Fiala(FialaParams),
}

impl TireSpec {
    pub fn load(&self) -> Result<Tire, String> {
        match self {
            TireSpec::Tir { file, pressure } => {
                let tir = match TIR_FILES.iter().find(|(n, _)| n == file) {
                    Some((_, text)) => TirFile::parse(text),
                    None => TirFile::read(file),
                }
                .map_err(|e| format!("tyre {file}: {e}"))?;
                let params = MfParams::from_tir(&tir).map_err(|e| format!("tyre {file}: {e}"))?;
                Tire::magic_formula(params, *pressure).map_err(|e| format!("tyre {file}: {e}"))
            }
            TireSpec::Fiala(p) => Tire::fiala(p.clone()),
        }
    }
}

/// Static equilibrium on flat, level ground.
#[derive(Clone, Debug, PartialEq)]
pub struct StaticState {
    /// Height of the chassis frame origin above the ground (m), and the chassis pitch and roll
    /// (rad; the rest attitude is `R_y(pitch)·R_x(roll)`).
    pub height: f64,
    pub pitch: f64,
    pub roll: f64,
    /// Per wheel: vertical tyre load (N), suspension travel (m) and tyre deflection (m).
    pub loads: Vec<f64>,
    pub travel: Vec<f64>,
    pub deflection: Vec<f64>,
    /// Per unit behind the towing unit: its joint's pitch and roll relative to the unit ahead
    /// (rad; a hinge has only pitch, a turntable neither).
    pub joints: Vec<[f64; 2]>,
}

impl StaticState {
    pub fn rotation(&self) -> DQuat {
        DQuat::from_rotation_y(self.pitch) * DQuat::from_rotation_x(self.roll)
    }
}

impl SuspensionDef {
    /// Kinematics of the left (`side` 0) or right (1) wheel.
    pub fn table(&self, side: usize) -> Result<KcTable, String> {
        let mut spec = self.kinematics.clone();
        if spec.travel.is_empty() {
            spec.travel = vec![-0.1, 0.1];
        }
        if side == 1 {
            for v in [&mut spec.y, &mut spec.toe, &mut spec.camber] {
                v.iter_mut().for_each(|x| *x = -*x);
            }
        }
        KcTable::new(spec).map_err(|e| e.to_string())
    }

    /// Force pushing the wheel away from the body (N) at travel `s` from spring and stops,
    /// with spring preload `preload` (rate springs).
    pub fn spring_force(&self, s: f64, preload: f64) -> f64 {
        let spring = match &self.spring.table {
            Some(t) => t.eval(s).0,
            None => preload + self.spring.rate * s,
        };
        let bump = self.bump_stop.map_or(0.0, |b| b.stiffness * (s - b.travel).max(0.0));
        let rebound = self.rebound_stop.map_or(0.0, |r| r.stiffness * (s - r.travel).min(0.0));
        spring + bump + rebound
    }

    /// Damper force (N) at travel rate `ds` (positive in bump), opposing it.
    pub fn damper_force(&self, ds: f64) -> f64 {
        let d = &self.damper;
        let (c, k) = if ds > 0.0 { (d.bump, d.degressivity_bump) } else { (d.rebound, d.degressivity_rebound) };
        ds * c / (1.0 + k * ds.abs())
    }
}

impl WheeledDef {
    /// Parse a definition without the `type` tag.
    pub fn from_toml(s: &str) -> Result<Self, VehicleError> {
        let mut def: Self = toml::from_str(s)?;
        def.finish()?;
        Ok(def)
    }

    /// Validate, load the tyres and resolve the spring tables and automatic preloads (called
    /// by the loaders, and again after editing a definition).
    pub fn finish(&mut self) -> Result<(), VehicleError> {
        let name = self.name.clone();
        let fail = |msg: String| VehicleError::Invalid(format!("{name}: {msg}"));
        self.resolve_shares().map_err(fail)?;
        self.validate().map_err(fail)?;
        let tires: Result<Vec<Tire>, String> = self
            .axles
            .iter()
            .map(|a| {
                a.tire.load().map(|t| match a.dual {
                    Some(d) => t.with_dual(d),
                    None => t,
                })
            })
            .collect();
        self.tires = tires.map_err(fail)?;
        for a in &mut self.axles {
            if let Some(s) = &mut a.suspension {
                s.spring.table = if s.spring.travel.is_empty() {
                    None
                } else {
                    Some(
                        CubicSpline::new(s.spring.travel.clone(), s.spring.force.clone())
                            .map_err(|e| fail(e.to_string()))?,
                    )
                };
            }
        }
        self.preload =
            self.wheels().map(|(a, _)| a.suspension.as_ref().and_then(|s| s.spring.preload).unwrap_or(0.0)).collect();
        let auto = self
            .axles
            .iter()
            .any(|a| a.suspension.as_ref().is_some_and(|s| s.spring.travel.is_empty() && s.spring.preload.is_none()));
        if auto {
            // Each vehicle's automatic springs carry its own static load: the towing unit's
            // alone, then each trailer's (its coupling unit and the hinge and turntable units
            // behind it) behind the vehicles ahead, whose preloads stay.
            let mut starts: Vec<usize> = vec![0];
            starts.extend(
                (0..self.units.len()).filter(|&k| matches!(self.units[k].joint, UnitJoint::Coupling(_))).map(|k| k + 1),
            );
            for (i, &start) in starts.iter().enumerate() {
                let end = starts.get(i + 1).copied().unwrap_or(self.num_units());
                let sub = self.prefix(end);
                let preload = sub.solve_static(STANDARD_GRAVITY, Some(start)).map_err(fail)?.1;
                self.preload[..preload.len()].copy_from_slice(&preload);
            }
        }
        self.rest = self.solve_static(STANDARD_GRAVITY, None).ok().map(|s| Box::new(s.0));
        Ok(())
    }

    /// The first `units` units with their axles, tyres and current preloads (for statics).
    fn prefix(&self, units: usize) -> WheeledDef {
        let axles = self.axles.iter().take_while(|a| a.unit < units).count();
        WheeledDef {
            units: self.units[..units - 1].to_vec(),
            axles: self.axles[..axles].to_vec(),
            tires: self.tires[..axles].to_vec(),
            preload: self.preload[..2 * axles].to_vec(),
            rest: None,
            ..self.clone()
        }
    }

    /// Steering shares: given ones as they are, geometric ones from the lead axle (small-angle
    /// value, for the share's other uses; the wheels steer by the exact tangent law).
    fn resolve_shares(&mut self) -> Result<(), String> {
        for a in &mut self.axles {
            a.share = match a.steer {
                Steer::Share(s) => s,
                Steer::Named(_) => 0.0,
            };
        }
        if !self.axles.iter().any(|a| a.is_geometric()) {
            return Ok(());
        }
        let own = self.axles.iter().filter(|a| a.unit == 0);
        let lead = own.clone().find(|a| a.share != 0.0).ok_or("geometric steering needs a lead axle with a share")?;
        let (xl, sl) = (lead.position.x, lead.share);
        let rear: Vec<f64> = own.filter(|a| !a.is_steered()).map(|a| a.position.x).collect();
        if rear.is_empty() {
            return Err("geometric steering needs an unsteered axle".into());
        }
        let xr = rear.iter().sum::<f64>() / rear.len() as f64;
        for a in self.axles.iter_mut().filter(|a| a.is_geometric()) {
            a.share = sl * (a.position.x - xr) / (xl - xr);
        }
        Ok(())
    }

    /// Reference x of the Ackermann geometry: the mean of the towing unit's axles without a
    /// steering share (or of all of them).
    pub fn steer_reference(&self) -> f64 {
        let own = || self.axles.iter().filter(|a| a.unit == 0);
        let unsteered: Vec<f64> = own().filter(|a| a.share == 0.0 && !a.is_geometric()).map(|a| a.position.x).collect();
        if unsteered.is_empty() {
            own().map(|a| a.position.x).sum::<f64>() / own().count() as f64
        } else {
            unsteered.iter().sum::<f64>() / unsteered.len() as f64
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        let pos = |x: f64| x > 0.0 && x.is_finite();
        let nonneg = |x: f64| x >= 0.0 && x.is_finite();
        let c = &self.chassis;
        if !pos(c.mass) || !pos(c.inertia.min_element()) || !c.com.is_finite() {
            return Err("chassis mass and inertia must be positive".into());
        }
        if !nonneg(c.drag_area.min_element()) {
            return Err("drag area must be non-negative".into());
        }
        if self.axles.is_empty() || self.axles.len() > MAX_WHEELS / 2 {
            return Err(format!("need 1 to {} axles", MAX_WHEELS / 2));
        }
        if self.axles.windows(2).any(|w| w[1].unit < w[0].unit) || self.axles.iter().any(|a| a.unit > self.units.len())
        {
            return Err("axles must be listed unit by unit, on existing units".into());
        }
        for (k, u) in self.units.iter().enumerate() {
            let fail = |m: String| Err(format!("unit {} ({}): {m}", k + 1, u.name));
            let c = &u.chassis;
            if u.parent > k {
                return fail("hangs from a later unit".into());
            }
            if !pos(c.mass) || !pos(c.inertia.min_element()) || !c.com.is_finite() || !nonneg(c.drag_area.min_element())
            {
                return fail("mass and inertia must be positive, drag non-negative".into());
            }
            if !u.position.is_finite() || u.hitch.is_some_and(|h| !h.position.is_finite()) {
                return fail("positions must be finite".into());
            }
            if let UnitJoint::Coupling(j) = u.joint {
                j.validate().or_else(fail)?;
            }
        }
        for (i, a) in self.axles.iter().enumerate() {
            let fail = |m: &str| Err(format!("axle {i}: {m}"));
            let forced = matches!(a.steer_mode, SteerMode::Articulation(_));
            if a.unit > 0 && a.is_steered() && !forced {
                return fail("trailer axles only steer by articulation");
            }
            if let SteerMode::Articulation(k) = a.steer_mode {
                let joint = a.unit.checked_sub(1).map(|u| self.units[u].joint);
                if !k.is_finite()
                    || a.steer != Steer::Share(0.0)
                    || !matches!(joint, Some(UnitJoint::Coupling(_) | UnitJoint::Turntable))
                {
                    return fail(
                        "articulation steering needs a finite gain, no share, and a unit on a coupling or turntable",
                    );
                }
            }
            if a.dual.is_some_and(|d| !pos(d)) {
                return fail("the dual tyres' spacing must be positive");
            }
            if a.is_geometric() && a.unit > 0 {
                return fail("geometric steering is for the towing unit");
            }
            if !a.position.is_finite() || !pos(a.position.y) {
                return fail("the (left) wheel position needs y > 0");
            }
            if !pos(a.wheel.mass) || !pos(a.wheel.inertia.min_element()) {
                return fail("wheel mass and inertia must be positive");
            }
            if !a.share.is_finite() || (a.share != 0.0 && self.steering.is_none()) {
                return fail("a steered axle needs [steering]");
            }
            if let SteerMode::Independent { max_angle, rate } = a.steer_mode
                && !(pos(max_angle) && max_angle <= std::f64::consts::FRAC_PI_2 && pos(rate))
            {
                return fail("independent steering needs 0 < max_angle ≤ π/2 and a positive rate");
            }
            let b = &a.brake;
            if ![b.max_torque, b.parking_torque, b.delay, b.time_constant].into_iter().all(nonneg) {
                return fail("brake torques and times must be non-negative");
            }
            if let Some(s) = &a.suspension {
                s.table(0).map_err(|e| format!("axle {i}: kinematics: {e}"))?;
                if !pos(s.carrier_mass) || !nonneg(s.carrier_inertia.min_element()) {
                    return fail("carrier mass must be positive and its inertia non-negative");
                }
                let sp = &s.spring;
                let table = !sp.travel.is_empty();
                if table && (sp.travel.len() != sp.force.len() || sp.rate != 0.0 || sp.preload.is_some()) {
                    return fail("a spring table takes travel and force of equal length, and no rate or preload");
                }
                if !table && !pos(sp.rate) {
                    return fail("spring rate must be positive");
                }
                let d = &s.damper;
                if ![d.bump, d.rebound, d.degressivity_bump, d.degressivity_rebound].into_iter().all(nonneg) {
                    return fail("damping rates and degressivities must be non-negative");
                }
                if !s.anti_roll.is_finite() || (s.anti_roll < 0.0 && (table || sp.rate + 2.0 * s.anti_roll < 0.0)) {
                    return fail("a negative anti-roll rate needs a rate spring and rate + 2·anti_roll ≥ 0");
                }
                if s.bump_stop.is_some_and(|b| !(b.travel > 0.0 && pos(b.stiffness)))
                    || s.rebound_stop.is_some_and(|r| !(r.travel < 0.0 && pos(r.stiffness)))
                {
                    return fail("stops need positive stiffness, bump travel > 0 and rebound travel < 0");
                }
            }
        }
        if let Some(s) = &self.steering
            && !(pos(s.max_angle) && s.max_angle < 1.5 && pos(s.rate) && (0.0..=1.0).contains(&s.ackermann))
        {
            return Err("steering needs 0 < max_angle < 1.5 rad, a positive rate and ackermann in [0, 1]".into());
        }
        self.powertrain.validate(self.axles.len())?;
        if self
            .colliders
            .iter()
            .chain(self.units.iter().flat_map(|u| &u.colliders))
            .any(|c| !pos(c.radius) || !c.center.is_finite() || !nonneg(c.friction))
        {
            return Err("collider radii must be positive and friction non-negative".into());
        }
        Ok(())
    }

    /// Axle and side (0 left, 1 right) of every wheel, in wheel order.
    pub fn wheels(&self) -> impl Iterator<Item = (&AxleDef, usize)> {
        self.axles.iter().flat_map(|a| [(a, 0), (a, 1)])
    }

    pub fn num_wheels(&self) -> usize {
        2 * self.axles.len()
    }

    /// Number of units (the towing unit and those behind it).
    pub fn num_units(&self) -> usize {
        self.units.len() + 1
    }

    /// Unit carrying wheel `w`.
    pub fn wheel_unit(&self, w: usize) -> usize {
        self.axles[w / 2].unit
    }

    /// Chassis of unit `u` (0: the towing unit).
    pub fn unit_chassis(&self, u: usize) -> &ChassisDef {
        if u == 0 { &self.chassis } else { &self.units[u - 1].chassis }
    }

    /// Coupling point for a trailer behind the last unit.
    pub fn rear_hitch(&self) -> Option<HitchDef> {
        self.units.last().map_or(self.hitch, |u| u.hitch)
    }

    /// Link of unit `u` in the multibody tree (see [`super::tree`]).
    pub fn unit_link(&self, u: usize) -> usize {
        let links = |a: &AxleDef| 2 * (1 + usize::from(a.suspension.is_some()) + usize::from(a.is_steered()));
        (0..u).map(|v| 1 + self.axles.iter().filter(|a| a.unit == v).map(links).sum::<usize>()).sum()
    }

    /// Design position of wheel `w` in its unit's frame.
    pub fn wheel_position(&self, w: usize) -> DVec3 {
        let p = self.axles[w / 2].position;
        if w.is_multiple_of(2) { p } else { DVec3::new(p.x, -p.y, p.z) }
    }

    /// Tyre of axle `axle` (after [`finish`](Self::finish)).
    pub fn tire(&self, axle: usize) -> &Tire {
        &self.tires[axle]
    }

    /// Spring preload of wheel `w` (N).
    pub fn preload(&self, w: usize) -> f64 {
        self.preload[w]
    }

    /// Unsprung mass of wheel `w` (carrier and wheel; kg).
    pub fn unsprung_mass(&self, w: usize) -> f64 {
        let a = &self.axles[w / 2];
        a.wheel.mass + a.suspension.as_ref().map_or(0.0, |s| s.carrier_mass)
    }

    /// Mass of all units (kg).
    pub fn total_mass(&self) -> f64 {
        self.chassis.mass
            + self.units.iter().map(|u| u.chassis.mass).sum::<f64>()
            + (0..self.num_wheels()).map(|w| self.unsprung_mass(w)).sum::<f64>()
    }

    /// Mass of unit `u` with its wheels (kg).
    pub fn unit_mass(&self, u: usize) -> f64 {
        self.unit_chassis(u).mass
            + (0..self.num_wheels()).filter(|&w| self.wheel_unit(w) == u).map(|w| self.unsprung_mass(w)).sum::<f64>()
    }

    /// Centre of mass of the towing unit with its wheels at design (chassis frame).
    pub fn total_com(&self) -> DVec3 {
        let mut m = self.chassis.com * self.chassis.mass;
        for w in (0..self.num_wheels()).filter(|&w| self.wheel_unit(w) == 0) {
            m += self.wheel_position(w) * self.unsprung_mass(w);
        }
        m / self.unit_mass(0)
    }

    /// Wheel spin inertia including the driveline share (kg·m²), per wheel.
    pub fn spin_inertia(&self) -> Vec<f64> {
        let extra = self.powertrain.wheel_inertia(self.num_wheels());
        self.wheels().zip(extra).map(|((a, _), e)| a.wheel.inertia.y + e).collect()
    }

    /// Sphere colliders on their units' links (the towing unit's first).
    pub fn sphere_colliders(&self) -> Vec<SphereCollider> {
        let units = std::iter::once(&self.colliders).chain(self.units.iter().map(|u| &u.colliders));
        units
            .enumerate()
            .flat_map(|(u, cs)| {
                let link = self.unit_link(u);
                cs.iter().map(move |c| SphereCollider {
                    friction: c.friction,
                    ..SphereCollider::new(link, c.center, c.radius, c.part as u8)
                })
            })
            .collect()
    }

    /// Static equilibrium on flat ground under gravity `g` (m/s²). Needs two axles, or a
    /// vehicle whose axles carry it (at least two per chain of units).
    pub fn static_state(&self, g: f64) -> Result<StaticState, String> {
        Ok(self.solve_static(g, None)?.0)
    }

    /// The static equilibrium under standard gravity, when there is one (computed by
    /// [`finish`](Self::finish)).
    pub fn rest_state(&self) -> Option<&StaticState> {
        self.rest.as_deref()
    }

    /// Two-axle single units keep the lever-rule solver; everything else minimises energy.
    fn solve_static(&self, g: f64, auto: Option<usize>) -> Result<(StaticState, Vec<f64>), String> {
        if self.tires.len() != self.axles.len() {
            return Err("definition not finished".into());
        }
        if self.units.is_empty() && self.axles.len() == 2 {
            self.solve_two_axle(g, auto.is_some())
        } else if self.axles.len() >= 2 {
            super::statics::solve(self, g, auto)
        } else {
            Err("static equilibrium (and automatic preload) needs two axles or more".into())
        }
    }

    /// Static equilibrium by the energy solver, also for two-axle vehicles (for checks).
    #[doc(hidden)]
    pub fn static_state_general(&self, g: f64) -> Result<StaticState, String> {
        Ok(super::statics::solve(self, g, None)?.0)
    }

    /// Loads by rigid-body statics over the wheel contacts (two axles; the attitude from the
    /// tyre deflections and suspension travels is iterated), the travel where each spring
    /// carries its load, and the chassis pose that puts every wheel centre at its loaded
    /// radius. With `auto`, springs without a given preload are preloaded to sit at zero
    /// travel; the preloads are returned.
    fn solve_two_axle(&self, g: f64, auto: bool) -> Result<(StaticState, Vec<f64>), String> {
        let n = self.num_wheels();
        let total = self.total_mass();
        let tables: Vec<Option<KcTable>> =
            self.wheels().map(|(a, side)| a.suspension.as_ref().map(|s| s.table(side).expect("validated"))).collect();
        let mut travel = vec![0.0; n];
        let mut preload = self.preload.clone();
        let (mut pitch, mut roll, mut height) = (0.0, 0.0, 0.0);
        let mut loads = vec![0.0; n];
        let mut deflection = vec![0.0; n];
        for _ in 0..50 {
            let rot = DQuat::from_rotation_y(pitch) * DQuat::from_rotation_x(roll);
            // Wheel centres and the whole-vehicle COM with the current travels (chassis frame).
            let centers: Vec<DVec3> = (0..n)
                .map(|w| {
                    self.wheel_position(w) + tables[w].as_ref().map_or(DVec3::ZERO, |t| t.eval(travel[w]).position)
                })
                .collect();
            let mut com = self.chassis.com * self.chassis.mass;
            for w in 0..n {
                com += centers[w] * self.unsprung_mass(w);
            }
            let com = rot * (com / total);
            let world: Vec<DVec3> = centers.iter().map(|&c| rot * c).collect();
            // Axle loads by levers along x, sides by levers along y.
            let axle_x = |a: usize| 0.5 * (world[2 * a].x + world[2 * a + 1].x);
            let (xf, xr) = (axle_x(0), axle_x(1));
            let front = total * g * (com.x - xr) / (xf - xr);
            for (a, axle_load) in [(0, front), (1, total * g - front)] {
                let (yl, yr) = (world[2 * a].y, world[2 * a + 1].y);
                let left = axle_load * (com.y - yr) / (yl - yr);
                loads[2 * a] = left;
                loads[2 * a + 1] = axle_load - left;
            }
            if loads.iter().any(|&l| l.is_nan() || l <= 0.0) {
                return Err(format!("the centre of mass lies outside the wheelbase (loads {loads:?})"));
            }
            for w in 0..n {
                let axle = &self.axles[w / 2];
                let tire = &self.tires[w / 2];
                deflection[w] = deflection_at(tire, loads[w]);
                let (Some(s), Some(t)) = (&axle.suspension, &tables[w]) else { continue };
                // Generalised force the spring must supply: the tyre load less the unsprung
                // weight, through the vertical component of the travel direction.
                let q = |s_: f64| {
                    let dz = (rot * travel_direction(t, s_)).z;
                    (loads[w] - self.unsprung_mass(w) * g) * dz
                };
                let automatic = auto && s.spring.travel.is_empty() && s.spring.preload.is_none();
                if automatic {
                    travel[w] = 0.0;
                    preload[w] = q(0.0) - s.spring_force(0.0, 0.0);
                } else {
                    travel[w] = solve_travel(|x| s.spring_force(x, preload[w]) - q(x))?;
                }
            }
            // Pose: every wheel centre at its loaded radius above the ground (Gauss–Newton on
            // height, pitch and roll).
            let targets: Vec<f64> = (0..n).map(|w| self.tires[w / 2].radius() - deflection[w]).collect();
            let (h0, p0, r0) = (height, pitch, roll);
            for _ in 0..20 {
                let (mut jtj, mut jtr) = ([[0.0; 3]; 3], [0.0; 3]);
                for w in 0..n {
                    let c = centers[w];
                    let (sp, cp) = pitch.sin_cos();
                    let (sr, cr) = roll.sin_cos();
                    let z = height - sp * c.x + cp * (sr * c.y + cr * c.z);
                    let j = [1.0, -cp * c.x - sp * (sr * c.y + cr * c.z), cp * (cr * c.y - sr * c.z)];
                    let r = targets[w] - z;
                    for a in 0..3 {
                        jtr[a] += j[a] * r;
                        for b in 0..3 {
                            jtj[a][b] += j[a] * j[b];
                        }
                    }
                }
                let d = solve3(jtj, jtr).ok_or("singular wheel layout")?;
                height += d[0];
                pitch += d[1];
                roll += d[2];
            }
            if (height - h0).abs() + (pitch - p0).abs() + (roll - r0).abs() < 1e-13 {
                break;
            }
        }
        Ok((StaticState { height, pitch, roll, loads, travel, deflection, joints: Vec::new() }, preload))
    }
}

/// Tyre deflection (m) carrying `load` (N), by bisection on the vertical force.
pub fn deflection_at(tire: &Tire, load: f64) -> f64 {
    let (mut lo, mut hi) = (0.0, 0.5 * tire.radius());
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if tire.vertical_force(mid) < load { lo = mid } else { hi = mid }
    }
    0.5 * (lo + hi)
}

/// Root of an increasing function on ±0.5 m by bisection.
fn solve_travel(f: impl Fn(f64) -> f64) -> Result<f64, String> {
    let (mut lo, mut hi) = (-0.5, 0.5);
    if !(f(lo) < 0.0 && f(hi) > 0.0) {
        return Err("the spring cannot carry the static load within ±0.5 m of travel".into());
    }
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if f(mid) < 0.0 { lo = mid } else { hi = mid }
    }
    Ok(0.5 * (lo + hi))
}

fn solve3(a: [[f64; 3]; 3], b: [f64; 3]) -> Option<[f64; 3]> {
    let det = |m: [[f64; 3]; 3]| {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    let d = det(a);
    if d.abs() < 1e-300 {
        return None;
    }
    let mut x = [0.0; 3];
    for (k, xk) in x.iter_mut().enumerate() {
        let mut m = a;
        for r in 0..3 {
            m[r][k] = b[r];
        }
        *xk = det(m) / d;
    }
    Some(x)
}

/// `dp/ds`, the wheel centre's motion per unit travel in the joint (chassis) frame (the
/// subspace gives it in carrier coordinates).
pub fn travel_direction(table: &KcTable, s: f64) -> DVec3 {
    let p = table.eval(s);
    p.rotation * p.s.lin
}
