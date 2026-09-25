//! Observation specs: named terms, each scaled and optionally clipped, concatenated into one
//! `f32` vector per agent. The same spec drives Python training, the viewer and (later) ROS,
//! so a trained policy sees identical inputs everywhere.
//!
//! | Term | Dim | Value |
//! |---|---|---|
//! | `goal_rel_world` / `goal_rel_body` / `goal_rel_heading` | 3 | goal − position in the world, body or heading frame (m) |
//! | `goal_yaw` | 2 | sin, cos of goal heading − heading |
//! | `position` | 3 | world position (m) |
//! | `height` / `agl` | 1 | z (m) / height above the ground or water surface (m) |
//! | `rot6d` / `quat` / `gravity_body` / `yaw` | 6 / 4 / 3 / 2 | attitude: first two rotation-matrix columns; quaternion (x, y, z, w with w ≥ 0); world down in the body frame; sin, cos of heading |
//! | `lin_vel_world` / `lin_vel_body` / `lin_vel_heading` | 3 | velocity (m/s) |
//! | `ang_vel_body` | 3 | body rates (rad/s) |
//! | `speed` / `sideslip` | 1 | forward speed (body x, m/s) / sideslip angle atan2(v_y, max(\|v_x\|, 1 m/s)) (rad) |
//! | `pitch_roll` | 2 | pitch and roll (rad; Z-Y-X Euler angles) |
//! | `wheel_speeds` / `wheel_slip` | wheels | ground vehicles: wheel spin × tyre radius (m/s) / longitudinal slip κ |
//! | `steering` | 1 | ground vehicles: steering angle of the equivalent bicycle (rad) |
//! | `gear_rpm` | 2 | ground vehicles: gear (1… forward, −1 reverse, 0 electric) and engine (first motor) speed (1000 rpm) |
//! | `motor_speeds` | rotors | rotor speeds mapped to [−1, 1] over their range |
//! | `last_action` | action | the action held during the last step |
//! | `clearance` | 1 | distance to the nearest terrain or solid obstacle (ground vehicles: solid obstacle), up to 20 m (costs ~0.6 µs) |
//! | `imu` / `imu_accel` / `imu_gyro` | 6 / 3 / 3 | IMU reading (sensor frame): specific force (m/s²) then rates (rad/s) |
//! | `gps_position` / `gps_velocity` / `gps_goal_rel_world` | 3 | GPS fix (ENU; m, m/s); goal − GPS position |
//! | `baro_altitude` | 1 | pressure altitude (m) |
//! | `mag` | 3 | magnetic field (sensor frame, µT) |
//! | `range` | 1 | rangefinder distance / max range (no return: 1) |
//! | `lidar` / `lidar_log` | beams | range / max range, or ln(1 + r)/ln(1 + max) (no return: 1) |
//! | `neighbors` | 7 × `count` | the `count` (default 3) nearest other active agents with centres within `range` (default 20 m), nearest first (ties by agent index): position and velocity relative to this agent in the heading frame (m, m/s), then 1; empty slots are all 0. The 1 is neither scaled nor clipped |
//! | `nearest_agent` | 1 | distance between this agent's colliders and the nearest other active agent's, up to `range` (default 20 m) |
//!
//! Sensor terms name their sensor (`sensor = "imu"`) and read zeros until its first reading
//! arrives. Non-finite values are written as 0.

use crate::interaction::{AgentGrid, AgentShape};
use crate::scenario::Goal;
use autonomousim_core::math::quat::{from_yaw, rot6d, wrap_angle, yaw};
use autonomousim_sensors::{BodyKinematics, Sensor, SensorConfig, SensorSpec};
use autonomousim_vehicles::ground::Wheeled;
use autonomousim_world::StaticWorld;
use glam::{DQuat, DVec3, EulerRot};
use serde::{Deserialize, Serialize};

/// Largest distance the `clearance` term looks (m).
pub const CLEARANCE_RANGE: f64 = 20.0;

/// Default `range` of the agent terms (m).
pub const NEIGHBOR_RANGE: f64 = 20.0;

/// Default and largest `count` of the `neighbors` term.
pub const NEIGHBOR_COUNT: usize = 3;
pub const MAX_NEIGHBORS: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TermKind {
    GoalRelWorld,
    GoalRelBody,
    GoalRelHeading,
    GoalYaw,
    Position,
    Height,
    Agl,
    Rot6d,
    Quat,
    GravityBody,
    Yaw,
    LinVelWorld,
    LinVelBody,
    LinVelHeading,
    AngVelBody,
    Speed,
    Sideslip,
    PitchRoll,
    WheelSpeeds,
    WheelSlip,
    Steering,
    GearRpm,
    MotorSpeeds,
    LastAction,
    Clearance,
    Imu,
    ImuAccel,
    ImuGyro,
    GpsPosition,
    GpsVelocity,
    GpsGoalRelWorld,
    BaroAltitude,
    Mag,
    Range,
    Lidar,
    LidarLog,
    Neighbors,
    NearestAgent,
}

impl TermKind {
    /// Sensor kind the term reads, if any.
    fn sensor_kind(self) -> Option<&'static str> {
        use TermKind::*;
        match self {
            Imu | ImuAccel | ImuGyro => Some("imu"),
            GpsPosition | GpsVelocity | GpsGoalRelWorld => Some("gps"),
            BaroAltitude => Some("baro"),
            Mag => Some("mag"),
            Range => Some("rangefinder"),
            Lidar | LidarLog => Some("lidar"),
            _ => None,
        }
    }

    /// Whether the term reads a ground vehicle's wheels, steering or powertrain.
    fn needs_wheels(self) -> bool {
        use TermKind::*;
        matches!(self, WheelSpeeds | WheelSlip | Steering | GearRpm)
    }
}

fn one() -> f64 {
    1.0
}

/// One observation term.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObsTerm {
    pub term: TermKind,
    /// Multiplies the value.
    #[serde(default = "one")]
    pub scale: f64,
    /// Clips the scaled value to `[−clip, clip]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clip: Option<f64>,
    /// Sensor name, for sensor terms.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sensor: Option<String>,
    /// Agents in the `neighbors` term.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub count: Option<usize>,
    /// Range of the agent terms (m).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub range: Option<f64>,
}

impl ObsTerm {
    pub fn new(term: TermKind, scale: f64) -> Self {
        Self { term, scale, clip: None, sensor: None, count: None, range: None }
    }

    pub fn clip(mut self, clip: f64) -> Self {
        self.clip = Some(clip);
        self
    }

    pub fn sensor(mut self, name: &str) -> Self {
        self.sensor = Some(name.into());
        self
    }

    pub fn count(mut self, count: usize) -> Self {
        self.count = Some(count);
        self
    }

    pub fn range(mut self, range: f64) -> Self {
        self.range = Some(range);
        self
    }
}

/// Hover observation (19 values for a quadrotor in `ctbr`): position error, attitude,
/// velocity, rates and the last action.
pub fn default_obs() -> Vec<ObsTerm> {
    vec![
        ObsTerm::new(TermKind::GoalRelWorld, 0.5).clip(5.0),
        ObsTerm::new(TermKind::Rot6d, 1.0),
        ObsTerm::new(TermKind::LinVelWorld, 0.5).clip(5.0),
        ObsTerm::new(TermKind::AngVelBody, 0.1).clip(5.0),
        ObsTerm::new(TermKind::LastAction, 1.0),
    ]
}

#[derive(Clone, Debug)]
struct Compiled {
    kind: TermKind,
    name: String,
    sensor: usize,
    dim: usize,
    scale: f64,
    clip: f64,
    /// Maximum range of a range sensor or of the agent terms (m).
    max_range: f64,
    /// Agents in the `neighbors` term.
    count: usize,
}

/// An observation spec resolved against a group's sensors and action size.
#[derive(Clone, Debug)]
pub struct CompiledObs {
    terms: Vec<Compiled>,
    dim: usize,
}

/// Everything an agent's observation may read.
pub struct ObsInput<'a> {
    pub kin: &'a BodyKinematics,
    pub goal: Goal,
    pub agl: f64,
    /// Rotor speeds and their range (rad/s).
    pub motors: &'a [f64],
    pub motor_range: (f64, f64),
    pub last_action: &'a [f64],
    /// The vehicle, when it is a ground vehicle.
    pub wheeled: Option<&'a Wheeled>,
    pub sensors: &'a [Sensor],
    pub world: &'a StaticWorld,
    /// Shapes of all agents in the world, and this agent's index among them.
    pub agents: &'a [AgentShape],
    pub me: usize,
    /// Neighbour index over `agents`.
    pub grid: &'a AgentGrid,
}

impl CompiledObs {
    pub fn new(
        terms: &[ObsTerm],
        sensors: &[SensorSpec],
        act_dim: usize,
        num_rotors: usize,
        num_wheels: usize,
    ) -> Result<Self, String> {
        let mut out = Vec::with_capacity(terms.len());
        let mut dim = 0;
        for t in terms {
            if !t.scale.is_finite() || t.clip.is_some_and(|c| c.is_nan() || c <= 0.0) {
                return Err(format!("invalid scale or clip in {t:?}"));
            }
            let (sensor, spec) = match (t.term.sensor_kind(), &t.sensor) {
                (Some(kind), Some(name)) => {
                    let i = sensors
                        .iter()
                        .position(|s| &s.name == name)
                        .ok_or_else(|| format!("observation {:?} reads unknown sensor {name:?}", t.term))?;
                    if sensors[i].config.kind() != kind {
                        return Err(format!("observation {:?} needs a {kind} sensor, {name:?} is not", t.term));
                    }
                    (i, Some(&sensors[i].config))
                }
                (Some(kind), None) => return Err(format!("observation {:?} needs `sensor` (a {kind})", t.term)),
                (None, Some(_)) => return Err(format!("observation {:?} does not read a sensor", t.term)),
                (None, None) => (usize::MAX, None),
            };
            if t.term.needs_wheels() && num_wheels == 0 {
                return Err(format!("observation {:?} needs a ground vehicle", t.term));
            }
            let agent_term = matches!(t.term, TermKind::Neighbors | TermKind::NearestAgent);
            if (t.count.is_some() && t.term != TermKind::Neighbors) || (t.range.is_some() && !agent_term) {
                return Err(format!("observation {:?} takes no count or range", t.term));
            }
            let count = t.count.unwrap_or(NEIGHBOR_COUNT);
            let range = t.range.unwrap_or(NEIGHBOR_RANGE);
            if agent_term && (!(1..=MAX_NEIGHBORS).contains(&count) || !(range > 0.0 && range.is_finite())) {
                return Err(format!("observation {:?}: count must be 1–{MAX_NEIGHBORS} and range positive", t.term));
            }
            let (d, max_range) = match (t.term, spec) {
                (TermKind::GoalYaw | TermKind::Yaw | TermKind::PitchRoll | TermKind::GearRpm, _) => (2, 0.0),
                (
                    TermKind::Height
                    | TermKind::Agl
                    | TermKind::Clearance
                    | TermKind::BaroAltitude
                    | TermKind::Speed
                    | TermKind::Sideslip
                    | TermKind::Steering,
                    _,
                ) => (1, 0.0),
                (TermKind::NearestAgent, _) => (1, range),
                (TermKind::Neighbors, _) => (7 * count, range),
                (TermKind::WheelSpeeds | TermKind::WheelSlip, _) => (num_wheels, 0.0),
                (TermKind::Rot6d | TermKind::Imu, _) => (6, 0.0),
                (TermKind::Quat, _) => (4, 0.0),
                (TermKind::MotorSpeeds, _) if num_rotors == 0 => {
                    return Err("observation motor_speeds needs a vehicle with rotors".into());
                }
                (TermKind::MotorSpeeds, _) => (num_rotors, 0.0),
                (TermKind::LastAction, _) => (act_dim, 0.0),
                (TermKind::Range, Some(SensorConfig::Rangefinder(c))) => (1, c.max_range),
                (TermKind::Lidar | TermKind::LidarLog, Some(SensorConfig::Lidar(c))) => {
                    (c.pattern.directions().len(), c.max_range)
                }
                _ => (3, 0.0),
            };
            out.push(Compiled {
                kind: t.term,
                name: serde_json::to_value(t.term).ok().and_then(|v| v.as_str().map(String::from)).unwrap_or_default(),
                sensor,
                dim: d,
                scale: t.scale,
                clip: t.clip.unwrap_or(f64::INFINITY),
                max_range,
                count,
            });
            dim += d;
        }
        Ok(Self { terms: out, dim })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// `(term name, offset, length)` of each term.
    pub fn layout(&self) -> Vec<(String, usize, usize)> {
        let mut off = 0;
        self.terms
            .iter()
            .map(|t| {
                let e = (t.name.clone(), off, t.dim);
                off += t.dim;
                e
            })
            .collect()
    }

    /// Write the observation into `out` (length [`dim`](Self::dim)).
    pub fn write(&self, inp: &ObsInput, out: &mut [f32]) {
        assert_eq!(out.len(), self.dim, "observation buffer length");
        let k = inp.kin;
        let q = k.attitude;
        let heading = from_yaw(yaw(q));
        let rel = inp.goal.position - k.position;
        let mut off = 0;
        for t in &self.terms {
            let dst = &mut out[off..off + t.dim];
            off += t.dim;
            let put3 = |dst: &mut [f32], v: DVec3| put(dst, &v.to_array(), t);
            match t.kind {
                TermKind::GoalRelWorld => put3(dst, rel),
                TermKind::GoalRelBody => put3(dst, q.inverse() * rel),
                TermKind::GoalRelHeading => put3(dst, heading.inverse() * rel),
                TermKind::GoalYaw => {
                    let e = wrap_angle(inp.goal.yaw - yaw(q));
                    put(dst, &[e.sin(), e.cos()], t)
                }
                TermKind::Position => put3(dst, k.position),
                TermKind::Height => put(dst, &[k.position.z], t),
                TermKind::Agl => put(dst, &[inp.agl], t),
                TermKind::Rot6d => put(dst, &rot6d(q), t),
                TermKind::Quat => {
                    let c = if q.w < 0.0 { -q } else { q };
                    put(dst, &c.to_array(), t)
                }
                TermKind::GravityBody => put3(dst, q.inverse() * DVec3::NEG_Z),
                TermKind::Yaw => {
                    let y = yaw(q);
                    put(dst, &[y.sin(), y.cos()], t)
                }
                TermKind::LinVelWorld => put3(dst, k.velocity),
                TermKind::LinVelBody => put3(dst, q.inverse() * k.velocity),
                TermKind::LinVelHeading => put3(dst, heading.inverse() * k.velocity),
                TermKind::AngVelBody => put3(dst, k.rates),
                TermKind::Speed => put(dst, &[(q.inverse() * k.velocity).x], t),
                TermKind::Sideslip => {
                    let v = q.inverse() * k.velocity;
                    put(dst, &[v.y.atan2(v.x.abs().max(1.0))], t)
                }
                TermKind::PitchRoll => {
                    let (_, pitch, roll) = q.to_euler(EulerRot::ZYX);
                    put(dst, &[pitch, roll], t)
                }
                TermKind::WheelSpeeds | TermKind::WheelSlip | TermKind::Steering | TermKind::GearRpm => {
                    let v = inp.wheeled.expect("ground terms are checked when the spec is compiled");
                    match t.kind {
                        TermKind::WheelSpeeds => {
                            for (w, (d, s)) in dst.iter_mut().zip(v.wheels()).enumerate() {
                                *d = value(s.spin * v.def().tire(w / 2).radius(), t);
                            }
                        }
                        TermKind::WheelSlip => {
                            for (d, s) in dst.iter_mut().zip(v.wheels()) {
                                *d = value(s.tire.kappa, t);
                            }
                        }
                        TermKind::Steering => put(dst, &[v.steering_angle()], t),
                        _ => {
                            let p = v.powertrain();
                            let krpm = p.engine_speed * 60.0 / std::f64::consts::TAU / 1000.0;
                            put(dst, &[f64::from(p.gear), krpm], t)
                        }
                    }
                }
                TermKind::MotorSpeeds => {
                    let (lo, hi) = inp.motor_range;
                    for (d, &w) in dst.iter_mut().zip(inp.motors) {
                        *d = value(2.0 * (w - lo) / (hi - lo).max(1e-9) - 1.0, t);
                    }
                }
                TermKind::LastAction => put(dst, inp.last_action, t),
                TermKind::Clearance => {
                    let c = if inp.wheeled.is_some() {
                        inp.world.obstacle_clearance(k.position, CLEARANCE_RANGE)
                    } else {
                        inp.world.clearance(k.position, CLEARANCE_RANGE)
                    };
                    put(dst, &[c], t)
                }
                TermKind::NearestAgent => {
                    let c = if inp.agents.is_empty() {
                        t.max_range
                    } else {
                        inp.grid.clearance(inp.agents, inp.me, t.max_range)
                    };
                    put(dst, &[c], t)
                }
                TermKind::Neighbors => {
                    dst.fill(0.0);
                    if inp.agents.is_empty() {
                        continue;
                    }
                    let mut near = Vec::with_capacity(t.count);
                    inp.grid.nearest(inp.agents, inp.me, t.max_range, t.count, &mut near);
                    let to_heading = heading.inverse();
                    for (slot, &(_, j)) in dst.as_chunks_mut::<7>().0.iter_mut().zip(&near) {
                        let o = &inp.agents[j];
                        put(&mut slot[0..3], &(to_heading * (o.center - k.position)).to_array(), t);
                        put(&mut slot[3..6], &(to_heading * (o.velocity - k.velocity)).to_array(), t);
                        slot[6] = 1.0;
                    }
                }
                _ => write_sensor(t, &inp.sensors[t.sensor], inp.goal, dst),
            }
        }
    }
}

fn write_sensor(t: &Compiled, sensor: &Sensor, goal: Goal, dst: &mut [f32]) {
    let v3 = |dst: &mut [f32], v: Option<DVec3>| match v {
        Some(v) => put(dst, &v.to_array(), t),
        None => dst.fill(0.0),
    };
    match (t.kind, sensor) {
        (TermKind::Imu, Sensor::Imu(s)) => {
            let (a, g) = dst.split_at_mut(3);
            v3(a, s.latest().map(|r| r.value.accel));
            v3(g, s.latest().map(|r| r.value.gyro));
        }
        (TermKind::ImuAccel, Sensor::Imu(s)) => v3(dst, s.latest().map(|r| r.value.accel)),
        (TermKind::ImuGyro, Sensor::Imu(s)) => v3(dst, s.latest().map(|r| r.value.gyro)),
        (TermKind::GpsPosition, Sensor::Gps(s)) => v3(dst, s.latest().map(|r| r.value.position)),
        (TermKind::GpsVelocity, Sensor::Gps(s)) => v3(dst, s.latest().map(|r| r.value.velocity)),
        (TermKind::GpsGoalRelWorld, Sensor::Gps(s)) => v3(dst, s.latest().map(|r| goal.position - r.value.position)),
        (TermKind::BaroAltitude, Sensor::Baro(s)) => match s.latest() {
            Some(r) => put(dst, &[r.value.altitude], t),
            None => dst.fill(0.0),
        },
        (TermKind::Mag, Sensor::Mag(s)) => v3(dst, s.latest().map(|r| r.value.field * 1e6)),
        (TermKind::Range, Sensor::Rangefinder(s)) => {
            let r = s.latest().and_then(|r| r.value.range).map_or(1.0, |r| r / t.max_range);
            put(dst, &[r], t)
        }
        (TermKind::Lidar | TermKind::LidarLog, Sensor::Lidar(s)) => match s.latest() {
            Some(scan) => {
                let log = t.kind == TermKind::LidarLog;
                let norm = if log { 1.0 / t.max_range.ln_1p() } else { 1.0 / t.max_range };
                for (d, &r) in dst.iter_mut().zip(&scan.ranges) {
                    let r = f64::from(r);
                    let x = if !r.is_finite() {
                        1.0
                    } else if log {
                        r.ln_1p() * norm
                    } else {
                        r * norm
                    };
                    *d = value(x, t);
                }
            }
            None => dst.fill(0.0),
        },
        _ => unreachable!("sensor kinds are checked when the spec is compiled"),
    }
}

#[inline]
fn value(x: f64, t: &Compiled) -> f32 {
    let y = (x * t.scale).clamp(-t.clip, t.clip);
    if y.is_finite() { y as f32 } else { 0.0 }
}

#[inline]
fn put(dst: &mut [f32], src: &[f64], t: &Compiled) {
    for (d, &x) in dst.iter_mut().zip(src) {
        *d = value(x, t);
    }
}

/// Canonical quaternion sign (w ≥ 0), as the `quat` term writes it.
pub fn canonical(q: DQuat) -> DQuat {
    if q.w < 0.0 { -q } else { q }
}

#[cfg(test)]
mod tests {
    use super::*;
    use autonomousim_sensors::{ImuConfig, LidarConfig};
    use autonomousim_world::testworlds;

    #[test]
    fn terms_resolve_and_write() {
        let sensors = vec![
            SensorSpec { name: "imu".into(), config: SensorConfig::Imu(ImuConfig::ideal()) },
            SensorSpec { name: "lidar".into(), config: SensorConfig::Lidar(LidarConfig::rl64()) },
        ];
        let terms = vec![
            ObsTerm::new(TermKind::GoalRelBody, 1.0),
            ObsTerm::new(TermKind::GoalYaw, 1.0),
            ObsTerm::new(TermKind::Quat, 1.0),
            ObsTerm::new(TermKind::LinVelHeading, 2.0).clip(1.0),
            ObsTerm::new(TermKind::MotorSpeeds, 1.0),
            ObsTerm::new(TermKind::LastAction, 1.0),
            ObsTerm::new(TermKind::ImuAccel, 1.0).sensor("imu"),
            ObsTerm::new(TermKind::LidarLog, 1.0).sensor("lidar"),
        ];
        let obs = CompiledObs::new(&terms, &sensors, 4, 4, 0).unwrap();
        assert_eq!(obs.dim(), 3 + 2 + 4 + 3 + 4 + 4 + 3 + 64);
        let layout = obs.layout();
        assert_eq!(layout[1], ("goal_yaw".to_string(), 3, 2));
        assert_eq!(layout.last().unwrap().1, obs.dim() - 64);

        let world = testworlds::flat(50.0);
        let yaw0 = 0.5;
        let kin = BodyKinematics {
            position: DVec3::new(1.0, 2.0, 3.0),
            attitude: DQuat::from_rotation_z(yaw0),
            velocity: DVec3::new(0.3, 0.0, 0.0),
            ..Default::default()
        };
        let goal = Goal { position: DVec3::new(1.0, 4.0, 3.0), yaw: yaw0 + 0.25 };
        let motors = [100.0, 200.0, 300.0, f64::NAN];
        let built: Vec<Sensor> = sensors
            .iter()
            .map(|s| {
                Sensor::new(
                    &s.config,
                    &autonomousim_core::time::Clock::new(500),
                    autonomousim_core::rng::Seed::from_u64(0),
                )
                .unwrap()
            })
            .collect();
        let inp = ObsInput {
            kin: &kin,
            goal,
            agl: 3.0,
            motors: &motors,
            motor_range: (100.0, 300.0),
            last_action: &[0.1, -0.2, 0.3, 2.0],
            wheeled: None,
            sensors: &built,
            world: &world,
            agents: &[],
            me: 0,
            grid: &AgentGrid::default(),
        };
        let mut out = vec![f32::NAN; obs.dim()];
        obs.write(&inp, &mut out);
        let body = DQuat::from_rotation_z(yaw0).inverse() * DVec3::new(0.0, 2.0, 0.0);
        let close = |a: &[f32], b: &[f64]| a.iter().zip(b).all(|(x, y)| (f64::from(*x) - y).abs() < 1e-6);
        assert!(close(&out[0..3], &body.to_array()));
        assert!(close(&out[3..5], &[0.25f64.sin(), 0.25f64.cos()]));
        let v = DQuat::from_rotation_z(yaw0).inverse() * DVec3::new(0.6, 0.0, 0.0);
        assert!(close(&out[9..12], &v.clamp(DVec3::splat(-1.0), DVec3::ONE).to_array()));
        assert!(close(&out[12..16], &[-1.0, 0.0, 1.0, 0.0]), "{:?}", &out[12..16]);
        assert!(close(&out[16..20], &[0.1, -0.2, 0.3, 2.0]));
        // No readings yet: zeros.
        assert!(out[20..].iter().all(|x| *x == 0.0));

        // Errors: unknown sensor, wrong kind, missing name, sensor on a state term.
        for bad in [
            ObsTerm::new(TermKind::ImuGyro, 1.0).sensor("gps"),
            ObsTerm::new(TermKind::Lidar, 1.0).sensor("imu"),
            ObsTerm::new(TermKind::Mag, 1.0),
            ObsTerm::new(TermKind::Agl, 1.0).sensor("imu"),
            ObsTerm::new(TermKind::Agl, 1.0).clip(0.0),
            ObsTerm::new(TermKind::WheelSpeeds, 1.0),
            ObsTerm::new(TermKind::GearRpm, 1.0),
        ] {
            assert!(CompiledObs::new(&[bad], &sensors, 4, 4, 0).is_err());
        }
    }
}
