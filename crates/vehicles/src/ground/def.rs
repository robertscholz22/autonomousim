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
//! to carry the static load (two-axle vehicles).

use super::powertrain::PowertrainDef;
use super::tire::{FialaParams, MfParams, TirFile, Tire};
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
    /// Resolved per axle by [`finish`](Self::finish).
    #[serde(skip)]
    tires: Vec<Tire>,
    /// Spring preload per wheel (N), given or computed.
    #[serde(skip)]
    preload: Vec<f64>,
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
    /// Left wheel centre at design ride height, chassis frame (m).
    pub position: DVec3,
    /// Independent suspension; `None` mounts the wheels rigidly (small robots).
    #[serde(default)]
    pub suspension: Option<SuspensionDef>,
    pub wheel: WheelDef,
    pub tire: TireSpec,
    /// Share of the steering angle (1 for a steered front axle, 0 unsteered, negative for
    /// counter-steering rear axles).
    #[serde(default)]
    pub steer: f64,
    pub brake: BrakeDef,
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
    /// Anti-roll bar rate at the wheel (N/m of travel difference).
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

/// Damper at the wheel (N·s/m), in bump (compression) and rebound.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DamperDef {
    pub bump: f64,
    pub rebound: f64,
}

/// Linear end stop engaging at `travel` (bump: positive, rebound: negative).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StopDef {
    pub travel: f64,
    pub stiffness: f64,
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
        ds * if ds > 0.0 { self.damper.bump } else { self.damper.rebound }
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
        let fail = |msg: String| VehicleError::Invalid(format!("{}: {msg}", self.name));
        self.validate().map_err(fail)?;
        let tires: Result<Vec<Tire>, String> = self.axles.iter().map(|a| a.tire.load()).collect();
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
            let preload = self.solve_static(STANDARD_GRAVITY, true).map_err(fail)?.1;
            self.preload = preload;
        }
        Ok(())
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
        if self.axles.is_empty() || self.axles.len() > 4 {
            return Err("need 1 to 4 axles".into());
        }
        for (i, a) in self.axles.iter().enumerate() {
            let fail = |m: &str| Err(format!("axle {i}: {m}"));
            if !a.position.is_finite() || !pos(a.position.y) {
                return fail("the (left) wheel position needs y > 0");
            }
            if !pos(a.wheel.mass) || !pos(a.wheel.inertia.min_element()) {
                return fail("wheel mass and inertia must be positive");
            }
            if !a.steer.is_finite() || (a.steer != 0.0 && self.steering.is_none()) {
                return fail("a steered axle needs [steering]");
            }
            if !nonneg(a.brake.max_torque) || !nonneg(a.brake.parking_torque) {
                return fail("brake torques must be non-negative");
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
                if !nonneg(s.damper.bump) || !nonneg(s.damper.rebound) || !nonneg(s.anti_roll) {
                    return fail("damping and anti-roll rates must be non-negative");
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
        if self.colliders.iter().any(|c| !pos(c.radius) || !c.center.is_finite() || !nonneg(c.friction)) {
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

    /// Design position of wheel `w` in the chassis frame.
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

    pub fn total_mass(&self) -> f64 {
        self.chassis.mass + (0..self.num_wheels()).map(|w| self.unsprung_mass(w)).sum::<f64>()
    }

    /// Centre of mass of the whole vehicle at design (chassis frame).
    pub fn total_com(&self) -> DVec3 {
        let mut m = self.chassis.com * self.chassis.mass;
        for w in 0..self.num_wheels() {
            m += self.wheel_position(w) * self.unsprung_mass(w);
        }
        m / self.total_mass()
    }

    /// Wheel spin inertia including the driveline share (kg·m²), per wheel.
    pub fn spin_inertia(&self) -> Vec<f64> {
        let extra = self.powertrain.wheel_inertia(self.num_wheels());
        self.wheels().zip(extra).map(|((a, _), e)| a.wheel.inertia.y + e).collect()
    }

    /// Sphere colliders on the chassis link.
    pub fn sphere_colliders(&self) -> Vec<SphereCollider> {
        self.colliders
            .iter()
            .map(|c| SphereCollider {
                friction: c.friction,
                ..SphereCollider::new(0, c.center, c.radius, c.part as u8)
            })
            .collect()
    }

    /// Static equilibrium on flat ground under gravity `g` (m/s²), for two-axle vehicles.
    pub fn static_state(&self, g: f64) -> Result<StaticState, String> {
        Ok(self.solve_static(g, false)?.0)
    }

    /// Loads by rigid-body statics over the wheel contacts (two axles; the attitude from the
    /// tyre deflections and suspension travels is iterated), the travel where each spring
    /// carries its load, and the chassis pose that puts every wheel centre at its loaded
    /// radius. With `auto`, springs without a given preload are preloaded to sit at zero
    /// travel; the preloads are returned.
    fn solve_static(&self, g: f64, auto: bool) -> Result<(StaticState, Vec<f64>), String> {
        if self.axles.len() != 2 {
            return Err("static equilibrium (and automatic preload) needs exactly two axles".into());
        }
        if self.tires.len() != self.axles.len() {
            return Err("definition not finished".into());
        }
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
        Ok((StaticState { height, pitch, roll, loads, travel, deflection }, preload))
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
