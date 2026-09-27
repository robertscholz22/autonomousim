//! Powertrains, differentials and brakes of wheeled vehicles.
//!
//! * **Combustion** (the SimpleMap model of Chrono::Vehicle): the engine speed follows the
//!   driveline algebraically (no clutch, no torque converter, no engine inertia). The engine
//!   torque interpolates between the zero- and full-throttle maps in the throttle. An automatic
//!   gearbox shifts up and down at fixed input speeds per gear, with an optional torque
//!   interruption. Open differentials split the torque in fixed shares; the driveline shafts'
//!   inertia is lumped onto the driven wheels.
//!
//!   With a **torque converter** (Chrono's `ChShaftsTorqueConverter`, as in its shafts
//!   powertrains) the engine speed is a state with the engine's inertia. The converter loads
//!   the engine with `(ω_e/K(R))²` at the speed ratio `R = ω_t/ω_e` (turbine over engine) and
//!   passes `TR(R)` times that to the gearbox, so a stalled converter (`R = 0`) multiplies the
//!   engine torque by `TR(0)` (about 2). When the wheels drive the engine (`ω_t > ω_e`: engine
//!   braking) the roles swap and the torque ratio is 1. The gearbox shifts on the turbine
//!   speed.
//! * **Electric**: motors with a torque limit, a power limit and a first-order torque lag,
//!   either torque-commanded or, with a no-load speed, voltage-commanded DC motors (torque
//!   falling linearly from stall at zero speed to zero at the commanded fraction of the no-load
//!   speed; back-EMF brakes a motor running faster than commanded),
//!   each driving one or more wheels (per axle, per side or per wheel). A differential command
//!   (`DriveInput::yaw`) is mixed into left and right motors for skid-steer and diff-drive robots.
//! * **Couplings**: brakes, locked, limited-slip and Torsen differentials and chain drives are
//!   torsional "bristles" between groups of wheels (or between wheels and their carriers): a
//!   stiff spring–damper on the integrated relative rotation whose torque is capped by the
//!   coupling's capacity. A braked wheel therefore holds without creep, a slipping one feels
//!   exactly its capacity, and a locked differential keeps both wheels in step.
//!
//! Gear ratios follow Chrono's convention: output speed over input speed.

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

const RPM: f64 = std::f64::consts::PI / 30.0;

/// Most wheels a vehicle can have, over all its units (eight axles).
pub const MAX_WHEELS: usize = 16;

/// Per-wheel commands, indexed by wheel (`2·axle + side`). When given in
/// [`DriveInput::wheels`], they replace the mixed commands: `drive` those of the electric
/// motors (a motor driving several wheels takes their mean; combustion drives keep the
/// throttle), `brake` the service brake, and `steer` the steering of independently steered
/// axles.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct WheelCommands {
    /// Motor command, `[−1, 1]`.
    pub drive: [f64; MAX_WHEELS],
    /// Service brake, `[0, 1]`.
    pub brake: [f64; MAX_WHEELS],
    /// Steering angle as a fraction of the axle's lock, `[−1, 1]`, positive turns left.
    pub steer: [f64; MAX_WHEELS],
}

/// Driver commands, all normalised.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DriveInput {
    /// Steering, `[−1, 1]`, positive turns left.
    pub steering: f64,
    /// Accelerator, `[0, 1]` for combustion engines; `[−1, 1]` for electric drives (negative
    /// drives backwards).
    pub throttle: f64,
    /// Service brake, `[0, 1]`.
    pub brake: f64,
    /// Parking brake.
    pub parking: bool,
    /// Reverse gear (combustion).
    pub reverse: bool,
    /// Differential drive command, `[−1, 1]`, positive turns left: added to right-side and
    /// subtracted from left-side electric motors.
    pub yaw: f64,
    /// Per-wheel commands, replacing the mixed ones where given.
    pub wheels: Option<WheelCommands>,
    /// Extra service brake per wheel, `[0, 1]`, added to the pedal (traction control and
    /// other brake interventions).
    #[serde(skip_serializing_if = "is_zero")]
    pub wheel_brake: [f64; MAX_WHEELS],
}

fn is_zero_f64(x: &f64) -> bool {
    *x == 0.0
}

fn is_zero(x: &[f64; MAX_WHEELS]) -> bool {
    x.iter().all(|&v| v == 0.0)
}

impl DriveInput {
    /// The same input with every field clamped to its range.
    pub fn clamped(&self) -> Self {
        let c = |x: f64, lo: f64| if x.is_finite() { x.clamp(lo, 1.0) } else { 0.0 };
        Self {
            steering: c(self.steering, -1.0),
            throttle: c(self.throttle, -1.0),
            brake: c(self.brake, 0.0),
            parking: self.parking,
            reverse: self.reverse,
            yaw: c(self.yaw, -1.0),
            wheels: self.wheels.map(|w| WheelCommands {
                drive: w.drive.map(|x| c(x, -1.0)),
                brake: w.brake.map(|x| c(x, 0.0)),
                steer: w.steer.map(|x| c(x, -1.0)),
            }),
            wheel_brake: self.wheel_brake.map(|x| c(x, 0.0)),
        }
    }
}

// ------------------------------------------------------------------------ definitions

/// Piecewise-linear table `[[x, y], …]`, held constant beyond its ends.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "Vec<[f64; 2]>", into = "Vec<[f64; 2]>")]
pub struct LinearTable {
    points: Vec<[f64; 2]>,
}

impl TryFrom<Vec<[f64; 2]>> for LinearTable {
    type Error = String;

    fn try_from(points: Vec<[f64; 2]>) -> Result<Self, String> {
        if points.is_empty() || points.iter().flatten().any(|v| !v.is_finite()) {
            return Err("a table needs at least one finite point".into());
        }
        if points.windows(2).any(|w| w[1][0] <= w[0][0]) {
            return Err("table abscissae must be strictly increasing".into());
        }
        Ok(Self { points })
    }
}

impl From<LinearTable> for Vec<[f64; 2]> {
    fn from(t: LinearTable) -> Self {
        t.points
    }
}

impl LinearTable {
    pub fn eval(&self, x: f64) -> f64 {
        let p = &self.points;
        let k = p.partition_point(|q| q[0] <= x);
        if k == 0 {
            return p[0][1];
        }
        if k == p.len() {
            return p[k - 1][1];
        }
        let ([x0, y0], [x1, y1]) = (p[k - 1], p[k]);
        y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    }

    pub fn points(&self) -> &[[f64; 2]] {
        &self.points
    }
}

// A definition, built once per vehicle type.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PowertrainDef {
    Combustion(CombustionDef),
    Electric(ElectricDef),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CombustionDef {
    pub engine: EngineDef,
    pub gearbox: GearboxDef,
    /// Driven axles.
    pub driven: Vec<usize>,
    /// Torque share of each driven axle (normalised; default equal): the centre differential.
    #[serde(default)]
    pub split: Vec<f64>,
    /// Final drive ratio (output/input speed, e.g. 0.2 for 5:1).
    pub final_drive: f64,
    /// Inertia of the driveline shafts referred to the mean driven-wheel speed (kg·m²), spread
    /// evenly over the driven wheels.
    #[serde(default)]
    pub driveline_inertia: f64,
    #[serde(default)]
    pub axle_differential: DifferentialDef,
    /// Coupling between the driven axles (ignored with one driven axle).
    #[serde(default)]
    pub center_differential: DifferentialDef,
    /// Hydrodynamic torque converter between the engine and the gearbox; without one the
    /// engine is geared rigidly to the wheels.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub torque_converter: Option<TorqueConverterDef>,
}

/// Torque converter maps over the speed ratio `R` = turbine speed / pump (engine) speed, in
/// `[0, 1]`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TorqueConverterDef {
    /// Capacity factor `K(R)` = pump speed / √(pump torque), (rad/s)/√(N·m).
    pub capacity_factor: LinearTable,
    /// Torque ratio `TR(R)` = turbine torque / pump torque.
    pub torque_ratio: LinearTable,
}

impl TorqueConverterDef {
    /// Pump torque (load on the engine) and turbine torque (to the gearbox), N·m, at engine
    /// speed `we` and turbine speed `wt` (rad/s).
    pub fn torques(&self, we: f64, wt: f64) -> (f64, f64) {
        let we = we.max(0.0);
        if wt <= we {
            // The engine drives; a turbine turning backward (rolling back in gear) is stalled.
            let r = if we > 0.0 { (wt / we).max(0.0) } else { 1.0 };
            let pump = (we / self.capacity_factor.eval(r)).powi(2);
            (pump, self.torque_ratio.eval(r) * pump)
        } else {
            // Overrun: the turbine drives the pump, without torque multiplication.
            let t = (wt / self.capacity_factor.eval(we / wt)).powi(2);
            (-t, -t)
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EngineDef {
    /// Above this speed the fuel is cut (zero-throttle torque).
    pub max_rpm: f64,
    /// Torque (N·m) over engine speed (rpm) at full and at zero throttle.
    pub full_throttle: LinearTable,
    pub zero_throttle: LinearTable,
    /// Rotating inertia of the engine and the converter's pump (kg·m²); used, and required,
    /// with a torque converter.
    #[serde(default, skip_serializing_if = "is_zero_f64")]
    pub inertia: f64,
}

impl EngineDef {
    /// Engine torque (N·m) at `rpm` and `throttle` in `[0, 1]`.
    fn torque(&self, rpm: f64, throttle: f64) -> f64 {
        let zero = self.zero_throttle.eval(rpm);
        if rpm > self.max_rpm { zero } else { throttle * self.full_throttle.eval(rpm) + (1.0 - throttle) * zero }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GearboxDef {
    /// Forward ratios from first gear up (output/input speed).
    pub forward: Vec<f64>,
    /// Reverse ratio (negative).
    pub reverse: f64,
    /// Down- and upshift speeds (rpm) of the gearbox input per forward gear (the engine, or the
    /// converter's turbine).
    pub shift: Vec<[f64; 2]>,
    /// Torque interruption per shift (s).
    #[serde(default)]
    pub shift_time: f64,
}

/// Differential or coupling between two outputs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DifferentialDef {
    /// Fixed torque split, free relative rotation.
    #[default]
    Open,
    /// The outputs turn together.
    Locked,
    /// Relative rotation resisted by up to `torque` (N·m) on top of the open split.
    LimitedSlip { torque: f64 },
    /// Torque-sensing (Torsen): the slower output may take up to `bias` (≥ 1) times the torque
    /// of the faster one, so relative rotation is resisted by up to (bias − 1)/(2(bias + 1))
    /// times the torque entering the differential (Chrono's `SimpleDrivelineXWD` axles).
    Torsen { bias: f64 },
}

impl DifferentialDef {
    /// Coupling capacity; `None` for an open differential.
    fn capacity(&self) -> Option<Capacity> {
        match *self {
            Self::Open => None,
            Self::Locked => Some(Capacity::Fixed(f64::INFINITY)),
            Self::LimitedSlip { torque } => Some(Capacity::Fixed(torque)),
            Self::Torsen { bias } => Some(Capacity::TorqueSensing(0.5 * (bias - 1.0) / (bias + 1.0))),
        }
    }

    fn validate(&self) -> Result<(), String> {
        match *self {
            Self::LimitedSlip { torque } if torque.is_nan() || torque < 0.0 => {
                Err("limited-slip torque must be non-negative".into())
            }
            Self::Torsen { bias } if !(bias.is_finite() && bias >= 1.0) => {
                Err("Torsen bias ratio must be finite and at least 1".into())
            }
            _ => Ok(()),
        }
    }
}

/// Capacity of a coupling: fixed (N·m), or a fraction of the drive torque entering it.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Capacity {
    Fixed(f64),
    TorqueSensing(f64),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ElectricDef {
    pub motors: Vec<MotorDef>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MotorDef {
    /// Driven wheels, numbered axle by axle, left before right (`2·axle + side`).
    pub wheels: Vec<usize>,
    /// Which differential command the motor follows.
    #[serde(default)]
    pub side: MotorSide,
    /// Peak torque (N·m) and power (W) at the motor shaft.
    pub max_torque: f64,
    pub max_power: f64,
    /// Reduction ratio (wheel speed / motor speed).
    pub ratio: f64,
    /// No-load motor speed at full command (rad/s): makes the command a voltage, with
    /// `max_torque` the stall torque. Omitted: the command is a torque.
    #[serde(default)]
    pub no_load_speed: Option<f64>,
    /// Torque response time constant (s).
    #[serde(default)]
    pub time_constant: f64,
    /// Rotor inertia (kg·m²), referred to the wheels.
    #[serde(default)]
    pub rotor_inertia: f64,
    /// How the motor's wheels share its torque: `open` (equal torque) or `locked` (a chain or
    /// shaft: equal speed).
    #[serde(default)]
    pub coupling: DifferentialDef,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MotorSide {
    #[default]
    Both,
    Left,
    Right,
}

impl PowertrainDef {
    /// Check the definition against a vehicle with `axles` axles.
    pub fn validate(&self, axles: usize) -> Result<(), String> {
        let pos = |x: f64| x > 0.0 && x.is_finite();
        let nonneg = |x: f64| x >= 0.0 && x.is_finite();
        match self {
            PowertrainDef::Combustion(c) => {
                if c.driven.is_empty() || c.driven.iter().any(|&a| a >= axles) {
                    return Err(format!("driven axles {:?} must be among the {axles} axles", c.driven));
                }
                if !c.split.is_empty() && (c.split.len() != c.driven.len() || c.split.iter().any(|&s| !pos(s))) {
                    return Err("split needs one positive share per driven axle".into());
                }
                if !pos(c.final_drive) || !nonneg(c.driveline_inertia) {
                    return Err("final_drive must be positive and driveline_inertia non-negative".into());
                }
                if !pos(c.engine.max_rpm) {
                    return Err("engine max_rpm must be positive".into());
                }
                let g = &c.gearbox;
                if g.forward.is_empty() || g.forward.iter().any(|&r| !pos(r)) || !pos(-g.reverse) {
                    return Err("gear ratios must be positive (reverse negative)".into());
                }
                if g.shift.len() != g.forward.len() || g.shift.iter().any(|s| !pos(s[1] - s[0])) {
                    return Err("need one [down, up] shift pair per forward gear with down < up".into());
                }
                if !nonneg(g.shift_time) {
                    return Err("shift_time must be non-negative".into());
                }
                c.axle_differential.validate()?;
                c.center_differential.validate()?;
                if let Some(tc) = &c.torque_converter {
                    if !pos(c.engine.inertia) {
                        return Err("a torque converter needs a positive engine inertia".into());
                    }
                    let [first, .., last] = tc.capacity_factor.points() else {
                        return Err("torque converter maps need at least two points".into());
                    };
                    if first[0] > 0.0 || last[0] < 1.0 || tc.torque_ratio.points().len() < 2 {
                        return Err("torque converter maps must cover speed ratios 0 to 1".into());
                    }
                    if tc.capacity_factor.points().iter().any(|p| !pos(p[1]))
                        || tc.torque_ratio.points().iter().any(|p| !pos(p[1]))
                    {
                        return Err("torque converter capacity factors and torque ratios must be positive".into());
                    }
                }
            }
            PowertrainDef::Electric(e) => {
                if e.motors.is_empty() {
                    return Err("an electric powertrain needs motors".into());
                }
                for m in &e.motors {
                    if m.wheels.is_empty() || m.wheels.iter().any(|&w| w >= 2 * axles) {
                        return Err(format!("motor wheels {:?} out of range", m.wheels));
                    }
                    if !(pos(m.max_torque) && pos(m.max_power) && pos(m.ratio))
                        || m.no_load_speed.is_some_and(|w| !pos(w))
                    {
                        return Err("motor torque, power and ratio must be positive".into());
                    }
                    if !(m.time_constant >= 0.0 && m.rotor_inertia >= 0.0) {
                        return Err("motor time constant and inertia must be non-negative".into());
                    }
                    m.coupling.validate()?;
                }
                let mut seen = vec![false; 2 * axles];
                for w in e.motors.iter().flat_map(|m| &m.wheels) {
                    if std::mem::replace(&mut seen[*w], true) {
                        return Err(format!("wheel {w} is driven by two motors"));
                    }
                }
            }
        }
        Ok(())
    }

    /// Extra spin inertia (kg·m²) the driveline adds to each wheel.
    pub fn wheel_inertia(&self, wheels: usize) -> Vec<f64> {
        let mut out = vec![0.0; wheels];
        match self {
            PowertrainDef::Combustion(c) => {
                let n = 2 * c.driven.len();
                for &a in &c.driven {
                    for w in [2 * a, 2 * a + 1] {
                        out[w] += c.driveline_inertia / n as f64;
                    }
                }
            }
            PowertrainDef::Electric(e) => {
                for m in &e.motors {
                    let share = m.rotor_inertia / (m.ratio * m.ratio) / m.wheels.len() as f64;
                    for &w in &m.wheels {
                        out[w] += share;
                    }
                }
            }
        }
        out
    }
}

// ------------------------------------------------------------------------ couplings

type Wheels = SmallVec<[usize; 4]>;

/// A torsional bristle between the mean rotation of wheel group `a` and that of group `b`
/// (empty `b`: the wheels' carriers, as for a brake). A geared coupling (ratio `ρ`) holds
/// `ω_a = ρ·ω_b`: its torque `t` on `a` comes with `−ρ·t` on `b`, so it only dissipates
/// (`t·(ω_a − ρ·ω_b) ≤ 0`).
#[derive(Clone, Debug, PartialEq)]
pub struct Coupling {
    a: Wheels,
    b: Wheels,
    ratio: f64,
    /// Bristle stiffness (N·m/rad) and damping (N·m·s/rad) on the relative rotation.
    stiffness: f64,
    damping: f64,
    /// Integrated relative rotation (rad).
    deflection: f64,
    /// Torque on group `a` in the last step (N·m); `b` receives the opposite.
    pub torque: f64,
}

/// Bristle frequency times the step and damping ratio: stiff enough to lock within a few
/// steps, soft enough for explicit integration together with the tyre.
const BRISTLE_OMEGA_DT: f64 = 0.3;
const BRISTLE_ZETA: f64 = 0.7;

impl Coupling {
    /// A coupling for wheels with spin inertias `inertia`, stepped at `dt`.
    pub fn new(a: &[usize], b: &[usize], inertia: &[f64], dt: f64) -> Self {
        Self::geared(a, b, 1.0, inertia, dt)
    }

    /// A coupling holding group `a`'s mean speed at `ratio` times group `b`'s.
    pub fn geared(a: &[usize], b: &[usize], ratio: f64, inertia: &[f64], dt: f64) -> Self {
        let sum = |g: &[usize]| g.iter().map(|&w| inertia[w]).sum::<f64>();
        let (ia, ib) = (sum(a), sum(b));
        let i_eff = if b.is_empty() { ia } else { ia * ib / (ib + ratio * ratio * ia) };
        let omega = BRISTLE_OMEGA_DT / dt;
        Self {
            a: a.into(),
            b: b.into(),
            ratio,
            stiffness: i_eff * omega * omega,
            damping: 2.0 * BRISTLE_ZETA * i_eff * omega,
            deflection: 0.0,
            torque: 0.0,
        }
    }

    fn mean(group: &[usize], spin: &[f64]) -> f64 {
        if group.is_empty() { 0.0 } else { group.iter().map(|&w| spin[w]).sum::<f64>() / group.len() as f64 }
    }

    /// Advance by `dt` with capacity `limit` (N·m) and add the torques to `out` (per wheel).
    pub fn step(&mut self, limit: f64, spin: &[f64], dt: f64, out: &mut [f64]) {
        if limit.is_nan() || limit <= 0.0 {
            self.deflection = 0.0;
            self.torque = 0.0;
            return;
        }
        let rel = Self::mean(&self.a, spin) - self.ratio * Self::mean(&self.b, spin);
        let s = self.deflection + rel * dt;
        let trial = -self.stiffness * s - self.damping * rel;
        let t = if trial.abs() > limit {
            let t = limit.copysign(trial);
            self.deflection = -t / self.stiffness;
            t
        } else {
            self.deflection = s;
            trial
        };
        self.torque = t;
        for &w in &self.a {
            out[w] += t / self.a.len() as f64;
        }
        for &w in &self.b {
            out[w] -= self.ratio * t / self.b.len() as f64;
        }
    }

    pub fn reset(&mut self) {
        self.deflection = 0.0;
        self.torque = 0.0;
    }
}

// ------------------------------------------------------------------------ runtime

/// Observable powertrain state.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PowertrainStatus {
    /// Selected gear: 1… forward, −1 reverse; 0 for electric drives.
    pub gear: i32,
    /// Engine speed (rad/s) and torque (N·m); for electric drives, the first motor's.
    pub engine_speed: f64,
    pub engine_torque: f64,
}

#[derive(Clone, Debug)]
struct Motor {
    wheels: Wheels,
    sign: f64,
    max_torque: f64,
    max_power: f64,
    ratio: f64,
    no_load_speed: Option<f64>,
    time_constant: f64,
    torque: f64,
}

/// Per-instance powertrain: turns driver input and wheel speeds into wheel torques.
#[derive(Clone, Debug)]
pub struct Powertrain {
    def: PowertrainDef,
    /// Driven wheels and their torque shares (combustion).
    shares: Vec<(usize, f64)>,
    motors: Vec<Motor>,
    couplings: Vec<(Coupling, Capacity)>,
    gear: i32,
    shift_timer: f64,
    /// Engine speed (rad/s), a state with a torque converter.
    engine_speed: f64,
    status: PowertrainStatus,
}

impl Powertrain {
    /// `inertia`: total spin inertia per wheel (including [`PowertrainDef::wheel_inertia`]).
    pub fn new(def: &PowertrainDef, inertia: &[f64], dt: f64) -> Self {
        let mut shares = Vec::new();
        let mut motors = Vec::new();
        let mut couplings = Vec::new();
        match def {
            PowertrainDef::Combustion(c) => {
                let total: f64 = if c.split.is_empty() { c.driven.len() as f64 } else { c.split.iter().sum() };
                for (k, &a) in c.driven.iter().enumerate() {
                    let share = c.split.get(k).copied().unwrap_or(1.0) / total;
                    shares.push((2 * a, 0.5 * share));
                    shares.push((2 * a + 1, 0.5 * share));
                    if let Some(cap) = c.axle_differential.capacity() {
                        couplings.push((Coupling::new(&[2 * a], &[2 * a + 1], inertia, dt), cap));
                    }
                }
                if let Some(cap) = c.center_differential.capacity() {
                    for pair in c.driven.windows(2) {
                        let (f, r) = ([2 * pair[0], 2 * pair[0] + 1], [2 * pair[1], 2 * pair[1] + 1]);
                        couplings.push((Coupling::new(&f, &r, inertia, dt), cap));
                    }
                }
            }
            PowertrainDef::Electric(e) => {
                for m in &e.motors {
                    motors.push(Motor {
                        wheels: m.wheels.as_slice().into(),
                        sign: match m.side {
                            MotorSide::Both => 0.0,
                            MotorSide::Left => -1.0,
                            MotorSide::Right => 1.0,
                        },
                        max_torque: m.max_torque,
                        max_power: m.max_power,
                        ratio: m.ratio,
                        no_load_speed: m.no_load_speed,
                        time_constant: m.time_constant,
                        torque: 0.0,
                    });
                    if let Some(cap) = m.coupling.capacity() {
                        for pair in m.wheels.windows(2) {
                            couplings.push((Coupling::new(&pair[..1], &pair[1..], inertia, dt), cap));
                        }
                    }
                }
            }
        }
        let mut p = Self {
            def: def.clone(),
            shares,
            motors,
            couplings,
            gear: 0,
            shift_timer: 0.0,
            engine_speed: 0.0,
            status: PowertrainStatus::default(),
        };
        p.reset(&vec![0.0; inertia.len()]);
        p
    }

    /// Clear the transient state; pick the gear that suits the wheel speeds.
    pub fn reset(&mut self, spin: &[f64]) {
        for (c, _) in &mut self.couplings {
            c.reset();
        }
        for m in &mut self.motors {
            m.torque = 0.0;
        }
        self.shift_timer = 0.0;
        self.gear = 0;
        if let PowertrainDef::Combustion(c) = &self.def {
            self.gear = 1;
            // Highest gear whose engine speed stays above that gear's downshift point.
            let shaft = self.driveshaft_speed(spin, c);
            for (k, &ratio) in c.gearbox.forward.iter().enumerate() {
                if shaft / ratio / RPM >= c.gearbox.shift[k][0] {
                    self.gear = k as i32 + 1;
                }
            }
            // The converter turning as one.
            self.engine_speed = (shaft / c.gearbox.forward[(self.gear - 1) as usize]).max(0.0);
        }
        self.status = PowertrainStatus { gear: self.gear, ..Default::default() };
    }

    fn driveshaft_speed(&self, spin: &[f64], c: &CombustionDef) -> f64 {
        self.shares.iter().map(|&(w, s)| s * spin[w]).sum::<f64>() / c.final_drive
    }

    pub fn status(&self) -> PowertrainStatus {
        self.status
    }

    /// Show `status` (playback of a recording) until the next step.
    pub fn set_status(&mut self, status: PowertrainStatus) {
        self.status = status;
    }

    /// Engage forward gear `gear` (1…) or reverse (−1) of a combustion drive; the automatic
    /// shifting takes over from the next step.
    pub fn set_gear(&mut self, gear: i32) {
        if let PowertrainDef::Combustion(c) = &self.def {
            self.gear = if gear < 0 { -1 } else { gear.clamp(1, c.gearbox.forward.len() as i32) };
            self.status.gear = self.gear;
        }
    }

    /// Add this step's drive torques (N·m, per wheel, about the spin axes) to `out` and advance
    /// the gearbox, motors and couplings by `dt`.
    pub fn step(&mut self, input: &DriveInput, spin: &[f64], dt: f64, out: &mut [f64]) {
        let mut drive = [0.0; MAX_WHEELS];
        match &self.def {
            PowertrainDef::Combustion(c) => {
                let g = &c.gearbox;
                let shaft = self.driveshaft_speed(spin, c);
                // Gear selection (instantaneous), then a torque gap of `shift_time`.
                let previous = self.gear;
                if input.reverse {
                    self.gear = -1;
                } else if self.gear < 1 {
                    self.gear = 1;
                } else {
                    let k = (self.gear - 1) as usize;
                    let rpm = shaft / g.forward[k] / RPM;
                    if rpm > g.shift[k][1] && k + 1 < g.forward.len() {
                        self.gear += 1;
                    } else if rpm < g.shift[k][0] && k > 0 {
                        self.gear -= 1;
                    }
                }
                if self.gear != previous && previous != 0 {
                    self.shift_timer = g.shift_time;
                }
                let ratio = if self.gear < 0 { g.reverse } else { g.forward[(self.gear - 1) as usize] };
                let input_speed = shaft / ratio;
                let throttle = input.throttle.clamp(0.0, 1.0);
                let e = &c.engine;
                let shifting = self.shift_timer > 0.0;
                if shifting {
                    self.shift_timer -= dt;
                }
                let (engine_speed, engine_torque, output) = match &c.torque_converter {
                    None => {
                        let torque = e.torque(input_speed / RPM, throttle);
                        (input_speed, torque, if shifting { 0.0 } else { torque })
                    }
                    Some(tc) => {
                        let we = self.engine_speed;
                        let torque = e.torque(we / RPM, throttle);
                        // In a shift the turbine runs free: no load, no output.
                        let (pump, turbine) = if shifting { (0.0, 0.0) } else { tc.torques(we, input_speed) };
                        self.engine_speed = (we + dt * (torque - pump) / e.inertia).max(0.0);
                        (we, torque, turbine)
                    }
                };
                let wheel = output / (ratio * c.final_drive);
                for &(w, s) in &self.shares {
                    drive[w] += wheel * s;
                }
                self.status = PowertrainStatus {
                    gear: self.gear,
                    engine_speed,
                    engine_torque: if shifting && c.torque_converter.is_none() { 0.0 } else { engine_torque },
                };
            }
            PowertrainDef::Electric(_) => {
                for (k, m) in self.motors.iter_mut().enumerate() {
                    let n = m.wheels.len() as f64;
                    let speed = m.wheels.iter().map(|&w| spin[w]).sum::<f64>() / n / m.ratio;
                    let command = match &input.wheels {
                        Some(wc) => m.wheels.iter().map(|&w| wc.drive[w]).sum::<f64>() / n,
                        None => (input.throttle + m.sign * input.yaw).clamp(-1.0, 1.0),
                    };
                    let mut demand = match m.no_load_speed {
                        Some(w0) => (m.max_torque * (command - speed / w0)).clamp(-m.max_torque, m.max_torque),
                        None => command * m.max_torque,
                    };
                    if demand.abs() * speed.abs() > m.max_power {
                        demand = (m.max_power / speed.abs()).copysign(demand);
                    }
                    let blend = if m.time_constant > 0.0 { 1.0 - (-dt / m.time_constant).exp() } else { 1.0 };
                    m.torque += (demand - m.torque) * blend;
                    for &w in &m.wheels {
                        drive[w] += m.torque / m.ratio / n;
                    }
                    if k == 0 {
                        self.status = PowertrainStatus { gear: 0, engine_speed: speed, engine_torque: m.torque };
                    }
                }
            }
        }
        for (o, d) in out.iter_mut().zip(drive) {
            *o += d;
        }
        for (c, cap) in &mut self.couplings {
            let limit = match *cap {
                Capacity::Fixed(t) => t,
                Capacity::TorqueSensing(f) => f * c.a.iter().chain(&c.b).map(|&w| drive[w]).sum::<f64>().abs(),
            };
            c.step(limit, spin, dt, out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_table_interpolates_and_clamps() {
        let t = LinearTable::try_from(vec![[0.0, 1.0], [2.0, 3.0], [4.0, 3.0]]).unwrap();
        assert_eq!(t.eval(-1.0), 1.0);
        assert_eq!(t.eval(1.0), 2.0);
        assert_eq!(t.eval(2.0), 3.0);
        assert_eq!(t.eval(9.0), 3.0);
        assert!(LinearTable::try_from(vec![[1.0, 0.0], [1.0, 1.0]]).is_err());
    }

    #[test]
    fn brake_coupling_holds_then_slips_at_capacity() {
        let dt = 1e-3;
        let inertia = [1.0];
        let mut c = Coupling::new(&[0], &[], &inertia, dt);
        // A wheel pushed by 50 N·m against a 100 N·m brake comes to rest and stays there.
        let (mut w, mut angle) = (2.0, 0.0);
        for _ in 0..2000 {
            let mut t = [50.0];
            c.step(100.0, &[w], dt, &mut t);
            w += t[0] / inertia[0] * dt;
            angle += w * dt;
        }
        let held = angle;
        for _ in 0..2000 {
            let mut t = [50.0];
            c.step(100.0, &[w], dt, &mut t);
            w += t[0] / inertia[0] * dt;
            angle += w * dt;
        }
        assert!((angle - held).abs() < 1e-9 && w.abs() < 1e-9, "creep {} rad at {w} rad/s", angle - held);
        // Against 150 N·m it slips with exactly the capacity.
        let w0 = w;
        for _ in 0..1000 {
            let mut t = [150.0];
            c.step(100.0, &[w], dt, &mut t);
            w += t[0] / inertia[0] * dt;
        }
        assert_eq!(c.torque, -100.0);
        assert!((w - w0 - 50.0).abs() < 0.5, "{w}");
    }

    #[test]
    fn torsen_splits_torque_up_to_the_bias_ratio() {
        let def: PowertrainDef = toml::from_str(
            r#"type = "electric"
               motors = [{ wheels = [0, 1], max_torque = 100.0, max_power = 1e9, ratio = 1.0,
                 time_constant = 0.0, coupling = { torsen = { bias = 3.0 } } }]"#,
        )
        .unwrap();
        def.validate(1).unwrap();
        let dt = 1e-3;
        let mut p = Powertrain::new(&def, &[1.0, 1.0], dt);
        let input = DriveInput { throttle: 1.0, ..Default::default() };
        // Wheel 1 is held by an 80 N·m load, wheel 0 runs free: the slow wheel gets 3/4 of the torque.
        let mut w = [0.0; 2];
        let mut t = [0.0; 2];
        for _ in 0..3000 {
            t = [0.0; 2];
            p.step(&input, &w, dt, &mut t);
            w[0] += t[0] * dt;
            w[1] += (t[1] - 80.0).max(-w[1] / dt) * dt;
        }
        assert!((t[1] - 75.0).abs() < 1e-6 && (t[0] - 25.0).abs() < 1e-6, "{t:?}");
        assert!(w[0] > 50.0 && w[1] < 1e-3, "{w:?}");
    }

    #[test]
    fn locked_coupling_equalises_speeds_and_conserves_momentum() {
        let dt = 1e-3;
        let inertia = [1.0, 3.0];
        let mut c = Coupling::new(&[0], &[1], &inertia, dt);
        let mut w = [4.0, 0.0];
        for _ in 0..200 {
            let mut t = [0.0; 2];
            c.step(f64::INFINITY, &w, dt, &mut t);
            for k in 0..2 {
                w[k] += t[k] / inertia[k] * dt;
            }
        }
        assert!((w[0] - w[1]).abs() < 1e-6, "{w:?}");
        assert!((w[0] + 3.0 * w[1] - 4.0).abs() < 1e-12);
    }

    /// A geared coupling holds `ω_a = ρ·ω_b`, conserves `I_a·ω_a·ρ + I_b·ω_b` (its torques are
    /// `t` and `−ρ·t`) and dissipates energy.
    #[test]
    fn geared_coupling_holds_its_ratio() {
        let (dt, inertia) = (1e-3, [2.0, 5.0]);
        let mut c = Coupling::geared(&[0], &[1], 0.6, &inertia, dt);
        let mut w = [4.0, 4.0];
        let energy = |w: &[f64; 2]| 0.5 * (inertia[0] * w[0] * w[0] + inertia[1] * w[1] * w[1]);
        let momentum = |w: &[f64; 2]| 0.6 * inertia[0] * w[0] + inertia[1] * w[1];
        let (e0, m0) = (energy(&w), momentum(&w));
        for _ in 0..300 {
            let mut t = [0.0; 2];
            c.step(f64::INFINITY, &w, dt, &mut t);
            for k in 0..2 {
                w[k] += t[k] / inertia[k] * dt;
            }
        }
        assert!((w[0] - 0.6 * w[1]).abs() < 1e-6, "{w:?}");
        assert!((momentum(&w) - m0).abs() < 1e-9 && energy(&w) < e0);
    }

    /// Through the APC's converter: held at standstill at full throttle, the engine settles
    /// where the stalled pump takes its torque, `(ω/K(0))² = T(ω)`, and the wheels get `TR(0)`
    /// times it through the gears. Coasting in gear at zero throttle, the wheels drive the
    /// engine (engine braking) without torque multiplication.
    #[test]
    fn torque_converter_stalls_and_overruns() {
        let def = crate::presets::wheeled("tracked_apc").unwrap();
        let PowertrainDef::Combustion(c) = &def.powertrain else { panic!() };
        let tc = c.torque_converter.as_ref().unwrap();
        let n = def.num_wheels();
        let dt = 1e-3;
        let mut p = Powertrain::new(&def.powertrain, &def.spin_inertia(), dt);
        let full = DriveInput { throttle: 1.0, ..Default::default() };
        let spin = vec![0.0; n];
        let mut out = vec![0.0; n];
        for _ in 0..3000 {
            out.fill(0.0);
            p.step(&full, &spin, dt, &mut out);
        }
        let we = p.status().engine_speed;
        let torque = c.engine.full_throttle.eval(we / RPM);
        assert!(((we / tc.capacity_factor.eval(0.0)).powi(2) / torque - 1.0).abs() < 1e-3, "stall at {we} rad/s");
        let wheels: f64 = out.iter().sum();
        let expected = tc.torque_ratio.eval(0.0) * torque / (c.gearbox.forward[0] * c.final_drive);
        assert!((wheels / expected - 1.0).abs() < 1e-3, "{wheels} vs {expected} N·m");

        // Rolling at a turbine speed of 150 rad/s in first gear, the engine idle.
        let turbine = 150.0;
        let spin = vec![turbine * c.gearbox.forward[0] * c.final_drive; n];
        p.reset(&spin);
        p.set_gear(1);
        let coast = DriveInput::default();
        for _ in 0..20 {
            out.fill(0.0);
            p.step(&coast, &spin, dt, &mut out);
        }
        let we = p.status().engine_speed;
        assert!(we < turbine);
        let (pump, turbine_torque) = tc.torques(we, turbine);
        assert!(pump < 0.0 && pump == turbine_torque);
        let wheels: f64 = out.iter().sum();
        assert!(wheels < 0.0 && (wheels * c.gearbox.forward[0] * c.final_drive / pump - 1.0).abs() < 0.05);
    }
}
