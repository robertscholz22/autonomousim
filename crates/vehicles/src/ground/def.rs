//! Wheeled vehicle definition as loaded from TOML (`type = "wheeled"`), shared between
//! instances.
//!
//! The chassis frame is FLU with an arbitrary origin; wheel positions, the centre of mass and
//! colliders are given in it. Axles list the **left** wheel; the right wheel mirrors it
//! (`y → −y`, and so do the suspension's lateral offset, toe and camber). An axle on the
//! centreline (`y = 0`) has a single wheel instead: single-track vehicles (bicycles,
//! motorcycles). Wheels are numbered axle by axle, left before right.
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
//!
//! Single-track vehicles steer by a [`SteeringHeadDef`]: the front wheel and its fork turn
//! freely about the tilted steering axis, driven by the rider's steering torque, and may carry
//! a leaning rider ([`RiderDef`]) and feet that hold them up at a standstill ([`FeetDef`]).

use super::powertrain::{MAX_WHEELS, PowertrainDef};
use super::tire::{FialaParams, McParams, MfParams, TirFile, Tire, TireModel, TrackPatch};
use super::units::{HitchDef, UnitDef, UnitJoint};
use crate::VehicleError;
use crate::multirotor::ContactDef;
use autonomousim_core::contact::SphereCollider;
use autonomousim_core::dynamics::{KcTable, KcTableSpec};
use autonomousim_core::material::Material;
use autonomousim_core::math::spline::CubicSpline;
use glam::{DMat3, DQuat, DVec3};
use serde::{Deserialize, Serialize};

/// Standard gravity, used for the automatic spring preloads.
pub const STANDARD_GRAVITY: f64 = 9.80665;

/// Built-in `.tir` files (`assets/tires`), by file stem.
const TIR_FILES: &[(&str, &str)] = &[
    ("Sedan_Pac02Tire", include_str!("../../../../assets/tires/Sedan_Pac02Tire.tir")),
    ("HMMWV_Pac02Tire", include_str!("../../../../assets/tires/HMMWV_Pac02Tire.tir")),
    ("Truck_Pac02Tire", include_str!("../../../../assets/tires/Truck_Pac02Tire.tir")),
];

/// Built-in motorcycle Magic Formula tyres (`assets/tires/*.toml`), by file stem.
const MC_FILES: &[(&str, &str)] = &[
    ("Evangelou_120_70_ZR17", include_str!("../../../../assets/tires/Evangelou_120_70_ZR17.toml")),
    ("Evangelou_180_55_ZR17", include_str!("../../../../assets/tires/Evangelou_180_55_ZR17.toml")),
    ("Bicycle_37_622", include_str!("../../../../assets/tires/Bicycle_37_622.toml")),
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
    /// Tracks instead of tyres: the towing unit's axles without a `tire` are road wheels on a
    /// track on each side (see [`TrackDef`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub track: Option<TrackDef>,
    /// Coupling point for a trailer at the rear of the towing unit (fifth wheel or pintle).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hitch: Option<HitchDef>,
    /// Units behind the towing unit (trailers, dollies and drawbars; unit `k + 1` is
    /// `units[k]`), usually added by [`with_trailers`](Self::with_trailers).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub units: Vec<UnitDef>,
    /// A rider leaning on the towing unit (single-track vehicles).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rider: Option<RiderDef>,
    /// Feet put down at a standstill (single-track vehicles).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub feet: Option<FeetDef>,
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
    /// Moments of inertia about the centre of mass, chassis axes (kg·m²).
    pub inertia: DVec3,
    /// Off-diagonal entries of the inertia tensor `[I_xy, I_xz, I_yz]` (kg·m²; `I_xz = −∫xz dm`),
    /// zero for principal axes along the chassis axes.
    #[serde(default, skip_serializing_if = "is_zero_vec")]
    pub products: DVec3,
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
    /// Left wheel centre at design ride height, chassis frame (m); on the centreline (`y = 0`)
    /// the axle has a single wheel.
    pub position: DVec3,
    /// Independent suspension; `None` mounts the wheels rigidly (small robots).
    #[serde(default)]
    pub suspension: Option<SuspensionDef>,
    pub wheel: WheelDef,
    /// The tyre; omitted for the road wheels of a [`TrackDef`] track.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tire: Option<TireSpec>,
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
    /// A steering head: the (single) wheel steers freely about a tilted axis.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steering_head: Option<SteeringHeadDef>,
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
    /// A trailing arm instead of `kinematics`: the wheel swings about a lateral pivot.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trailing_arm: Option<TrailingArmDef>,
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

/// A road wheel's arm: the pivot lies `length` ahead of the wheel (negative: behind it, a
/// leading arm), the arm dropping at `angle` below the horizontal at zero travel (rad). The
/// wheel moves on the arc: `z(s) = s` and `x(s) = L·(cos θ₀ − cos θ)` with
/// `sin θ = sin θ₀ − s/|L|`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrailingArmDef {
    pub length: f64,
    pub angle: f64,
}

impl TrailingArmDef {
    /// The arc as a kinematics table (17 knots over the travel where the arm stays within
    /// ±72° of the horizontal).
    pub fn kinematics(&self) -> KcTableSpec {
        let (l, s0) = (self.length.abs(), self.angle.sin());
        let limit = 0.95;
        let (lo, hi) = (l * (s0 - limit), l * (s0 + limit));
        let travel: Vec<f64> = (0..17).map(|i| lo + (hi - lo) * i as f64 / 16.0).collect();
        let x = travel.iter().map(|s| self.length * (self.angle.cos() - (1.0 - (s0 - s / l).powi(2)).sqrt())).collect();
        KcTableSpec { z: travel.clone(), x, travel, ..Default::default() }
    }
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

impl ChassisDef {
    /// Inertia tensor about the centre of mass (chassis axes).
    pub fn inertia_tensor(&self) -> DMat3 {
        inertia_tensor(self.inertia, self.products)
    }
}

impl AxleDef {
    /// One wheel on the centreline rather than a left/right pair.
    pub fn is_single(&self) -> bool {
        self.position.y == 0.0
    }

    /// Number of wheels (1 or 2).
    pub fn num_wheels(&self) -> usize {
        if self.is_single() { 1 } else { 2 }
    }

    /// Whether the axle's wheels have a (prescribed) steering joint.
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

fn is_zero_vec(x: &DVec3) -> bool {
    *x == DVec3::ZERO
}

/// Symmetric tensor from its diagonal and off-diagonal entries `[xy, xz, yz]`.
pub fn inertia_tensor(moments: DVec3, products: DVec3) -> DMat3 {
    let (m, p) = (moments, products);
    DMat3::from_cols(DVec3::new(m.x, p.x, p.y), DVec3::new(p.x, m.y, p.z), DVec3::new(p.y, p.z, m.z))
}

/// Whether a symmetric tensor is positive definite (leading minors).
fn positive_definite(t: DMat3) -> bool {
    let (a, b, c) = (t.x_axis, t.y_axis, t.z_axis);
    a.x > 0.0 && a.x * b.y - a.y * b.x > 0.0 && t.determinant() > 0.0 && (a + b + c).is_finite()
}

/// Steering head of a single-track axle (bicycles, motorcycles): the wheel and the steered
/// body (fork, handlebar) turn freely about the steering axis, tilted back from the vertical by
/// the head angle and passing `offset` behind the wheel centre. The rider's steering torque
/// (`DriveInput::steering`, positive turning left, times `max_torque`), a damper and the lock
/// stops act on it. With a suspension, the fork slides along the axis (unless it has
/// kinematics of its own, in the steered frame).
///
/// The trail, the distance the ground contact trails behind the axis' ground point, is
/// `(R·sin λ − offset)/cos λ` for wheel radius `R`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SteeringHeadDef {
    /// Head angle `λ`: the axis' tilt back from the vertical (rad).
    pub angle: f64,
    /// Fork offset: the wheel centre's distance ahead of the axis, square to it (m).
    pub offset: f64,
    /// The steered body without the wheel: mass (kg), centre of mass (chassis frame, design
    /// pose, straight ahead) and inertia about it (chassis axes; kg·m²).
    pub mass: f64,
    pub com: DVec3,
    pub inertia: DVec3,
    #[serde(default, skip_serializing_if = "is_zero_vec")]
    pub products: DVec3,
    /// Steering lock each way (rad), where a stop of `lock_stiffness` (N·m/rad) engages.
    pub lock: f64,
    #[serde(default = "default_lock_stiffness")]
    pub lock_stiffness: f64,
    /// Steering damper (N·m·s/rad).
    #[serde(default)]
    pub damping: f64,
    /// The rider's largest steering torque (N·m).
    pub max_torque: f64,
}

fn default_lock_stiffness() -> f64 {
    500.0
}

impl SteeringHeadDef {
    /// The steering axis, pointing up (chassis frame).
    pub fn axis(&self) -> DVec3 {
        DVec3::new(-self.angle.sin(), 0.0, self.angle.cos())
    }

    /// The point of the axis square to the wheel centre `wheel` (chassis frame).
    pub fn pivot(&self, wheel: DVec3) -> DVec3 {
        wheel - self.offset * DVec3::new(self.angle.cos(), 0.0, self.angle.sin())
    }

    /// Stop and damper torque (N·m) at steering angle `delta` and rate `rate`.
    pub fn passive_torque(&self, delta: f64, rate: f64) -> f64 {
        let excess = delta - delta.clamp(-self.lock, self.lock);
        // The stop is damped over 10 ms.
        let stop = if excess != 0.0 { -self.lock_stiffness * (excess + 0.01 * rate) } else { 0.0 };
        stop - self.damping * rate
    }
}

/// A rider's upper body, leaning about the chassis x axis through the hip on a servo: torque
/// `stiffness·(φ_ref − φ) − damping·φ̇`, at most `max_torque`, towards the lean
/// `φ_ref = DriveInput::lean · max_lean` (positive: to the right, as roll). The legs and
/// the lower body belong to the chassis.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RiderDef {
    /// Mass (kg), centre of mass (chassis frame, upright) and inertia about it (chassis axes).
    pub mass: f64,
    pub com: DVec3,
    pub inertia: DVec3,
    #[serde(default, skip_serializing_if = "is_zero_vec")]
    pub products: DVec3,
    /// A point on the lean axis (chassis frame).
    pub hip: DVec3,
    /// Largest lean each way (rad), servo stiffness (N·m/rad), damping (N·m·s/rad) and torque
    /// (N·m).
    pub max_lean: f64,
    pub stiffness: f64,
    pub damping: f64,
    pub max_torque: f64,
}

impl RiderDef {
    /// Servo torque (N·m) at lean `phi` and rate `rate` for the normalised command `lean`.
    pub fn servo_torque(&self, lean: f64, phi: f64, rate: f64) -> f64 {
        (self.stiffness * (lean * self.max_lean - phi) - self.damping * rate).clamp(-self.max_torque, self.max_torque)
    }
}

/// The rider's feet: spheres on the chassis, put down below `speed` (forward, m/s) and lifted
/// back onto the pegs or pedals above `1.25·speed`. Down, they hover a little above flat
/// ground, so they catch the vehicle once it leans.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeetDef {
    /// Left foot's centre down and up (chassis frame; the right foot mirrors it) and its
    /// radius (m).
    pub down: DVec3,
    pub up: DVec3,
    pub radius: f64,
    #[serde(default = "default_feet_speed")]
    pub speed: f64,
}

fn default_feet_speed() -> f64 {
    1.5
}

/// Tracks on both sides of the towing unit. Every axle without a `tire` is a pair of road
/// wheels, each carrying a track patch ([`TrackPatch`]) in place of a tyre. Tie a side's road
/// wheels to its drive by a locked coupling (the band): their common spin times the patch
/// radius is the band speed, and their spin inertias carry the band's, sprocket's and idler's.
/// The sprocket and idler are colliders on the hull (mirrored to the right side), so the
/// track's ends meet steps and banks.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrackDef {
    pub patch: TrackPatch,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sprocket: Option<RollerDef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub idler: Option<RollerDef>,
    /// How `DriveInput::steering` steers (vehicles without per-wheel motor commands).
    #[serde(default, skip_serializing_if = "TrackSteering::is_brake")]
    pub steering: TrackSteering,
}

/// Steering of a tracked vehicle by `DriveInput::steering` (`> 0` turns left: the left track
/// is the inner one going forward).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum TrackSteering {
    /// Service brake on the inner track, `|steering|` of the pedal (Chrono's braked
    /// differential steering): the braking torque is lost.
    #[default]
    Brake,
    /// A controlled differential (regenerative, geared steering): a steering brake with
    /// capacity `|steering|·torque` (N·m, at the side's road wheels) holds the inner track at
    /// `ratio` times the outer track's speed, and passes `ratio` times its torque on to the
    /// outer track. Fully applied, the tracks turn at the fixed radius
    /// `(B/2)·(1 + ratio)/(1 − ratio)` (`B`: the track width), at any speed; slipping, wider.
    Regenerative { ratio: f64, torque: f64 },
}

impl TrackSteering {
    fn is_brake(&self) -> bool {
        *self == Self::Brake
    }
}

/// A sprocket or idler: its left centre in the chassis frame and its radius over the track (m).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RollerDef {
    pub position: DVec3,
    pub radius: f64,
}

impl TrackDef {
    /// Sprocket and idler of both sides as running-gear colliders (left, then right).
    fn colliders(&self) -> impl Iterator<Item = GroundColliderDef> + '_ {
        self.sprocket.iter().chain(&self.idler).flat_map(|r| {
            [1.0, -1.0].map(|s| GroundColliderDef {
                center: DVec3::new(r.position.x, s * r.position.y, r.position.z),
                radius: r.radius,
                friction: 1.0,
                part: GroundPart::Skid,
            })
        })
    }
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
        /// Crown radius of a toroidal tread (m; two-wheelers), 0 for a thin disc.
        #[serde(default, skip_serializing_if = "is_zero_f64")]
        crown_radius: f64,
        /// Model turn slip (MF 6.x files).
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        turn_slip: bool,
    },
    /// Motorcycle Magic Formula (MF-MC) parameters in TOML: a built-in name (`assets/tires`,
    /// without extension) or a path. Toroidal with the file's crown radius.
    Mc {
        file: String,
    },
    Fiala(FialaParams),
    /// A track patch (normally given once in [`TrackDef`]).
    Track(TrackPatch),
}

impl TireSpec {
    pub fn load(&self) -> Result<Tire, String> {
        match self {
            TireSpec::Tir { file, pressure, crown_radius, turn_slip } => {
                let tir = match TIR_FILES.iter().find(|(n, _)| n == file) {
                    Some((_, text)) => TirFile::parse(text),
                    None => TirFile::read(file),
                }
                .map_err(|e| format!("tyre {file}: {e}"))?;
                let params = MfParams::from_tir(&tir).map_err(|e| format!("tyre {file}: {e}"))?;
                let mut tire = Tire::magic_formula(params, *pressure).and_then(|t| t.with_crown(*crown_radius));
                if *turn_slip {
                    tire = tire.and_then(Tire::with_turn_slip);
                }
                tire.map_err(|e| format!("tyre {file}: {e}"))
            }
            TireSpec::Mc { file } => {
                let text = match MC_FILES.iter().find(|(n, _)| n == file) {
                    Some((_, text)) => text.to_string(),
                    None => std::fs::read_to_string(file).map_err(|e| format!("tyre {file}: {e}"))?,
                };
                let params: McParams = toml::from_str(&text).map_err(|e| format!("tyre {file}: {e}"))?;
                Tire::motorcycle(params).map_err(|e| format!("tyre {file}: {e}"))
            }
            TireSpec::Fiala(p) => Tire::fiala(p.clone()),
            TireSpec::Track(p) => Tire::track(p.clone()),
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
        let mut spec = match self.trailing_arm {
            Some(arm) if self.kinematics.travel.is_empty() => arm.kinematics(),
            Some(_) => return Err("give either kinematics or a trailing arm".into()),
            None => self.kinematics.clone(),
        };
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

    /// Kinematics of a wheel on a steering head with axis `axis` (steered frame): without
    /// kinematics of its own, a fork sliding along the axis.
    pub fn fork_table(&self, axis: DVec3) -> Result<KcTable, String> {
        if !self.kinematics.travel.is_empty() || self.trailing_arm.is_some() {
            return self.table(0);
        }
        let travel = vec![-0.1, 0.1];
        let (x, z) = (travel.iter().map(|s| s * axis.x).collect(), travel.iter().map(|s| s * axis.z).collect());
        KcTable::new(KcTableSpec { travel, x, z, ..Default::default() }).map_err(|e| e.to_string())
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
        let track = self.track.as_ref().map(|t| TireSpec::Track(t.patch.clone()));
        let tires: Result<Vec<Tire>, String> = self
            .axles
            .iter()
            .enumerate()
            .map(|(i, a)| {
                let spec = a.tire.as_ref().or(track.as_ref()).ok_or(format!("axle {i} needs a tire or [track]"))?;
                spec.load().map(|t| match a.dual {
                    Some(d) => t.with_dual(d),
                    None => t,
                })
            })
            .collect();
        self.tires = tires.map_err(fail)?;
        // Track patches without a design load share the vehicle's weight.
        let patches = self.tires.iter().filter(|t| t.is_track()).count();
        let share = self.total_mass() * STANDARD_GRAVITY / (2 * patches.max(1)) as f64;
        for t in &mut self.tires {
            if let TireModel::Track(p) = &mut t.model
                && p.nominal_load == 0.0
            {
                p.nominal_load = share;
            }
        }
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
            preload: self.preload[..self.axle_wheels(axles).start].to_vec(),
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
        if !pos(c.mass) || !positive_definite(c.inertia_tensor()) || !c.com.is_finite() {
            return Err("chassis mass and inertia must be positive".into());
        }
        if !nonneg(c.drag_area.min_element()) {
            return Err("drag area must be non-negative".into());
        }
        if self.axles.is_empty() || self.num_wheels() > MAX_WHEELS {
            return Err(format!("need 1 to {MAX_WHEELS} wheels"));
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
            if !pos(c.mass)
                || !positive_definite(c.inertia_tensor())
                || !c.com.is_finite()
                || !nonneg(c.drag_area.min_element())
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
            let track = a.tire.is_none() || matches!(a.tire, Some(TireSpec::Track(_)));
            if track && (a.unit > 0 || a.is_steered() || a.dual.is_some()) {
                return fail("road wheels are on the towing unit, unsteered and single");
            }
            if a.is_geometric() && a.unit > 0 {
                return fail("geometric steering is for the towing unit");
            }
            if !a.position.is_finite() || !nonneg(a.position.y) {
                return fail("the (left) wheel position needs y > 0 (y = 0: a single wheel)");
            }
            if a.is_single() && (track || a.dual.is_some() || a.suspension.as_ref().is_some_and(|s| s.anti_roll != 0.0))
            {
                return fail("a single wheel has a tyre, no dual and no anti-roll bar");
            }
            if let Some(h) = &a.steering_head {
                let tensor = inertia_tensor(h.inertia, h.products);
                if !a.is_single() || a.unit > 0 || a.is_steered() {
                    return fail("a steering head is for a single, otherwise unsteered wheel of the towing unit");
                }
                if !(0.0..1.2).contains(&h.angle)
                    || !h.offset.is_finite()
                    || !pos(h.mass)
                    || !h.com.is_finite()
                    || !positive_definite(tensor)
                {
                    return fail(
                        "a steering head needs 0 ≤ angle < 1.2 rad, a finite offset, and positive mass and inertia",
                    );
                }
                if !(pos(h.lock) && h.lock < 1.5 && pos(h.lock_stiffness) && nonneg(h.damping) && pos(h.max_torque)) {
                    return fail("a steering head needs 0 < lock < 1.5 rad and positive stiffness and torque");
                }
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
                if s.trailing_arm.is_some_and(|a| !(a.length.is_finite() && a.length != 0.0 && a.angle.abs() < 1.2)) {
                    return fail("a trailing arm needs a non-zero length and |angle| < 1.2 rad");
                }
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
        if let Some(t) = &self.track
            && t.sprocket.iter().chain(&t.idler).any(|r| !pos(r.radius) || !r.position.is_finite())
        {
            return Err("sprocket and idler need finite positions and positive radii".into());
        }
        if let Some(TrackDef { steering: TrackSteering::Regenerative { ratio, torque }, .. }) = &self.track
            && !((0.0..1.0).contains(ratio) && pos(*torque))
        {
            return Err("regenerative steering needs 0 <= ratio < 1 and a positive torque".into());
        }
        if let Some(s) = &self.steering
            && !(pos(s.max_angle) && s.max_angle < 1.5 && pos(s.rate) && (0.0..=1.0).contains(&s.ackermann))
        {
            return Err("steering needs 0 < max_angle < 1.5 rad, a positive rate and ackermann in [0, 1]".into());
        }
        if self.axles.iter().filter(|a| a.steering_head.is_some()).count() > 1 {
            return Err("one steering head at most".into());
        }
        if let Some(r) = &self.rider {
            let tensor = inertia_tensor(r.inertia, r.products);
            if !(pos(r.mass) && r.com.is_finite() && r.hip.is_finite() && positive_definite(tensor)) {
                return Err("the rider needs positive mass and inertia and finite positions".into());
            }
            if !(pos(r.max_lean) && r.max_lean < 1.5 && pos(r.stiffness) && nonneg(r.damping) && pos(r.max_torque)) {
                return Err("the rider needs 0 < max_lean < 1.5 rad and positive stiffness and torque".into());
            }
        }
        if let Some(f) = &self.feet
            && !(f.down.is_finite()
                && f.up.is_finite()
                && pos(f.down.y)
                && pos(f.up.y)
                && pos(f.radius)
                && nonneg(f.speed))
        {
            return Err(
                "the feet need finite positions left of the centreline (y > 0), a positive radius and speed".into()
            );
        }
        self.powertrain.validate(&self.axle_ranges())?;
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

    /// Axle and side (0 left or single, 1 right) of every wheel, in wheel order.
    pub fn wheels(&self) -> impl Iterator<Item = (&AxleDef, usize)> {
        self.axles.iter().flat_map(|a| (0..a.num_wheels()).map(move |side| (a, side)))
    }

    pub fn num_wheels(&self) -> usize {
        self.axles.iter().map(AxleDef::num_wheels).sum()
    }

    /// Axle of wheel `w`.
    pub fn wheel_axle(&self, w: usize) -> usize {
        let mut first = 0;
        for (a, axle) in self.axles.iter().enumerate() {
            first += axle.num_wheels();
            if w < first {
                return a;
            }
        }
        panic!("wheel {w} out of range")
    }

    /// Side of wheel `w`: 0 left (or a single wheel), 1 right.
    pub fn wheel_side(&self, w: usize) -> usize {
        w - self.axle_wheels(self.wheel_axle(w)).start
    }

    /// Wheels of axle `a` (`a = axles.len()`: the empty range after the last).
    pub fn axle_wheels(&self, a: usize) -> std::ops::Range<usize> {
        let start = self.axles[..a].iter().map(AxleDef::num_wheels).sum();
        start..start + self.axles.get(a).map_or(0, AxleDef::num_wheels)
    }

    /// [`axle_wheels`](Self::axle_wheels) of every axle.
    pub fn axle_ranges(&self) -> Vec<std::ops::Range<usize>> {
        (0..self.axles.len()).map(|a| self.axle_wheels(a)).collect()
    }

    /// Whether the towing unit stands on single wheels only (it needs a rider or feet to
    /// stay up).
    pub fn is_single_track(&self) -> bool {
        self.axles.iter().filter(|a| a.unit == 0).all(AxleDef::is_single)
    }

    /// The axle with a steering head and its wheel, if any.
    pub fn steering_head(&self) -> Option<(&SteeringHeadDef, usize)> {
        let a = self.axles.iter().position(|a| a.steering_head.is_some())?;
        Some((self.axles[a].steering_head.as_ref().expect("found"), self.axle_wheels(a).start))
    }

    /// Number of units (the towing unit and those behind it).
    pub fn num_units(&self) -> usize {
        self.units.len() + 1
    }

    /// Unit carrying wheel `w`.
    pub fn wheel_unit(&self, w: usize) -> usize {
        self.axles[self.wheel_axle(w)].unit
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
        let links = |a: &AxleDef| {
            a.num_wheels() * (1 + usize::from(a.suspension.is_some()) + usize::from(a.is_steered()))
                + usize::from(a.steering_head.is_some())
        };
        let rider = usize::from(self.rider.is_some());
        (0..u)
            .map(|v| {
                1 + self.axles.iter().filter(|a| a.unit == v).map(links).sum::<usize>() + rider * usize::from(v == 0)
            })
            .sum()
    }

    /// Design position of wheel `w` in its unit's frame.
    pub fn wheel_position(&self, w: usize) -> DVec3 {
        let p = self.axles[self.wheel_axle(w)].position;
        if self.wheel_side(w) == 0 { p } else { DVec3::new(p.x, -p.y, p.z) }
    }

    /// The last unit's rear on its centreline, in its frame: the rearmost extent of its wheels
    /// and colliders along x (for a single unit, the vehicle's rear).
    pub fn tail(&self) -> DVec3 {
        let u = self.num_units() - 1;
        let link = self.unit_link(u);
        let wheels = (0..self.num_wheels())
            .filter(|&w| self.wheel_unit(w) == u)
            .map(|w| self.wheel_position(w).x - self.wheel_tire(w).radius());
        let colliders = self.sphere_colliders().into_iter().filter(|c| c.link == link).map(|c| c.center.x - c.radius);
        DVec3::new(wheels.chain(colliders).fold(f64::INFINITY, f64::min).min(0.0), 0.0, 0.0)
    }

    /// Origin of unit `u`'s frame in the towing unit's, with all units in line (design pose).
    pub fn unit_origin(&self, u: usize) -> DVec3 {
        let mut p = DVec3::ZERO;
        let mut u = u;
        while u > 0 {
            p += self.units[u - 1].position;
            u = self.units[u - 1].parent;
        }
        p
    }

    /// Design position of wheel `w` in the towing unit's frame, all units in line.
    pub fn wheel_position_in_line(&self, w: usize) -> DVec3 {
        self.unit_origin(self.wheel_unit(w)) + self.wheel_position(w)
    }

    /// [`sphere_colliders`](Self::sphere_colliders) with their centres in the towing unit's
    /// frame, all units in line.
    pub fn colliders_in_line(&self) -> Vec<SphereCollider> {
        let origins: Vec<DVec3> = (0..self.num_units()).map(|u| self.unit_origin(u)).collect();
        let links: Vec<usize> = (0..self.num_units()).map(|u| self.unit_link(u)).collect();
        let mut cs = self.sphere_colliders();
        for c in &mut cs {
            let u = links.iter().position(|&l| l == c.link).expect("colliders sit on unit links");
            c.center += origins[u];
        }
        cs
    }

    /// Tyre of axle `axle` (after [`finish`](Self::finish)).
    pub fn tire(&self, axle: usize) -> &Tire {
        &self.tires[axle]
    }

    /// Tyre of wheel `w`'s axle.
    pub fn wheel_tire(&self, w: usize) -> &Tire {
        &self.tires[self.wheel_axle(w)]
    }

    /// Spring preload of wheel `w` (N).
    pub fn preload(&self, w: usize) -> f64 {
        self.preload[w]
    }

    /// Unsprung mass of wheel `w` (carrier and wheel; kg).
    pub fn unsprung_mass(&self, w: usize) -> f64 {
        let a = &self.axles[self.wheel_axle(w)];
        a.wheel.mass + a.suspension.as_ref().map_or(0.0, |s| s.carrier_mass)
    }

    /// Motion resistance on level `material` at low speed, as a share of the weight (for route
    /// costs): the tyres' rolling resistance on it; for tracks, their running gear's plus, on
    /// soft soil, compaction and bulldozing (Bekker) under evenly loaded road wheels.
    pub fn motion_resistance(&self, material: &Material, g: f64) -> f64 {
        let patches: Vec<&TrackPatch> = (0..self.axles.len())
            .filter_map(|a| match &self.tire(a).model {
                TireModel::Track(p) => Some(p),
                _ => None,
            })
            .collect();
        let Some(p) = patches.first() else { return material.rolling_resistance };
        let Some(s) = &material.soil else { return p.rolling_resistance };
        let weight = self.total_mass() * g;
        let z = s.sinkage(weight / (2 * patches.len()) as f64 / (p.width * p.length), p.width);
        p.rolling_resistance + 2.0 * (s.compaction(p.width, 0.0, z) + s.bulldozing(p.width, z, g)) / weight
    }

    /// Mass of all units (kg).
    pub fn total_mass(&self) -> f64 {
        self.chassis.mass
            + self.units.iter().map(|u| u.chassis.mass).sum::<f64>()
            + (0..self.num_wheels()).map(|w| self.unsprung_mass(w)).sum::<f64>()
            + self.extra_masses().map(|m| m.0).sum::<f64>()
    }

    /// Mass of unit `u` with its wheels (and the towing unit's with its steered body and
    /// rider; kg).
    pub fn unit_mass(&self, u: usize) -> f64 {
        let extra = if u == 0 { self.extra_masses().map(|m| m.0).sum::<f64>() } else { 0.0 };
        self.unit_chassis(u).mass
            + (0..self.num_wheels()).filter(|&w| self.wheel_unit(w) == u).map(|w| self.unsprung_mass(w)).sum::<f64>()
            + extra
    }

    /// The towing unit's steered body and rider: masses and centres (chassis frame, design).
    fn extra_masses(&self) -> impl Iterator<Item = (f64, DVec3)> + '_ {
        let heads = self.axles.iter().filter_map(|a| a.steering_head.as_ref()).map(|h| (h.mass, h.com));
        heads.chain(self.rider.iter().map(|r| (r.mass, r.com)))
    }

    /// Centre of mass of the towing unit with its wheels at design (chassis frame).
    pub fn total_com(&self) -> DVec3 {
        let mut m = self.chassis.com * self.chassis.mass;
        for w in (0..self.num_wheels()).filter(|&w| self.wheel_unit(w) == 0) {
            m += self.wheel_position(w) * self.unsprung_mass(w);
        }
        for (mass, com) in self.extra_masses() {
            m += com * mass;
        }
        m / self.unit_mass(0)
    }

    /// Wheel spin inertia including the driveline share (kg·m²), per wheel.
    pub fn spin_inertia(&self) -> Vec<f64> {
        let extra = self.powertrain.wheel_inertia(&self.axle_ranges(), self.num_wheels());
        self.wheels().zip(extra).map(|((a, _), e)| a.wheel.inertia.y + e).collect()
    }

    /// Sphere colliders on their units' links (the towing unit's first, then its sprockets and
    /// idlers, then its feet, down).
    pub fn sphere_colliders(&self) -> Vec<SphereCollider> {
        let rollers: Vec<GroundColliderDef> = self.track.iter().flat_map(|t| t.colliders()).collect();
        let feet: Vec<GroundColliderDef> = self
            .feet
            .iter()
            .flat_map(|f| {
                [1.0, -1.0].map(|s| GroundColliderDef {
                    center: DVec3::new(f.down.x, s * f.down.y, f.down.z),
                    radius: f.radius,
                    friction: 1.0,
                    part: GroundPart::Skid,
                })
            })
            .collect();
        let own = self.colliders.iter().chain(&rollers).chain(&feet).collect::<Vec<_>>();
        let units = std::iter::once(own).chain(self.units.iter().map(|u| u.colliders.iter().collect()));
        units
            .enumerate()
            .flat_map(|(u, cs)| {
                let link = self.unit_link(u);
                cs.into_iter().map(move |c| SphereCollider {
                    friction: c.friction,
                    ..SphereCollider::new(link, c.center, c.radius, c.part as u8)
                })
            })
            .collect()
    }

    /// The sprockets and idlers among [`sphere_colliders`](Self::sphere_colliders): collider
    /// index, side (0 left, 1 right) and radius.
    pub fn rollers(&self) -> Vec<(usize, usize, f64)> {
        let first = self.colliders.len();
        self.track
            .iter()
            .flat_map(|t| t.sprocket.iter().chain(&t.idler))
            .flat_map(|r| [0, 1].map(|side| (side, r.radius)))
            .enumerate()
            .map(|(k, (side, radius))| (first + k, side, radius))
            .collect()
    }

    /// Per wheel on a track, its neighbours along the track (front, rear): the next road
    /// wheels on the same side by design position.
    pub fn track_neighbours(&self) -> Vec<[Option<usize>; 2]> {
        let n = self.num_wheels();
        let on_track = |w: usize| self.wheel_tire(w).is_track() && self.wheel_unit(w) == 0;
        (0..n)
            .map(|w| {
                if !on_track(w) {
                    return [None; 2];
                }
                let x = self.wheel_position(w).x;
                let side = (0..n).filter(|&o| o != w && self.wheel_side(o) == self.wheel_side(w) && on_track(o));
                let front = side.clone().filter(|&o| self.wheel_position(o).x > x);
                let rear = side.filter(|&o| self.wheel_position(o).x < x);
                let by_x = |a: &usize, b: &usize| self.wheel_position(*a).x.total_cmp(&self.wheel_position(*b).x);
                [front.min_by(by_x), rear.max_by(by_x)]
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
        if self.units.is_empty()
            && self.axles.len() == 2
            && self.axles.iter().all(|a| !a.is_single())
            && self.rider.is_none()
        {
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
