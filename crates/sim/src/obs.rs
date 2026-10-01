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
//! | `air_data` | 3 | true airspeed (m/s), angle of attack and sideslip (rad) relative to the air at the vehicle (wind, gusts and turbulence; see [`AirFlow`](autonomousim_vehicles::aero::AirFlow)) |
//! | `wind_body` | 3 | air velocity at the vehicle in the body frame (m/s) |
//! | `pitch_roll` | 2 | pitch and roll (rad; Z-Y-X Euler angles) |
//! | `lean` | 2 | roll (rad, positive right; Z-Y-X Euler angles, so about the pitched heading) and its rate (rad/s) |
//! | `wheel_speeds` / `wheel_slip` | wheels | ground vehicles: wheel spin × tyre radius (m/s) / longitudinal slip κ |
//! | `steering` | 1 / 2 | ground vehicles: steering angle of the equivalent bicycle (rad); with a free steering head (two-wheelers) its angle (rad, positive left) and rate (rad/s) |
//! | `rider_lean` | 2 | ground vehicles: the rider's upper-body lean relative to the frame (rad, positive right) and its rate (rad/s); 0 without a rider |
//! | `feet` | 1 | ground vehicles: 1 while a two-wheeler's feet are down, else 0 |
//! | `gear_rpm` | 2 | ground vehicles: gear (1… forward, −1 reverse, 0 electric) and engine (first motor) speed (1000 rpm) |
//! | `motor_speeds` | rotors | rotor speeds mapped to [−1, 1] over their range |
//! | `last_action` | action | the action held during the last step |
//! | `clearance` | 1 | distance to the nearest terrain or solid obstacle (ground vehicles: solid obstacle), up to 20 m (costs ~0.6 µs) |
//! | `imu` / `imu_accel` / `imu_gyro` | 6 / 3 / 3 | IMU reading (sensor frame): specific force (m/s²) then rates (rad/s) |
//! | `gps_position` / `gps_velocity` / `gps_goal_rel_world` | 3 | GPS fix (ENU; m, m/s); goal − GPS position |
//! | `baro_altitude` | 1 | pressure altitude (m) |
//! | `pitot` | 1 | indicated airspeed (m/s) |
//! | `mag` | 3 | magnetic field (sensor frame, µT) |
//! | `range` | 1 | rangefinder distance / max range (no return: 1) |
//! | `lidar` / `lidar_log` | beams | range / max range, or ln(1 + r)/ln(1 + max) (no return: 1) |
//! | `neighbors` | 7 × `count` | the `count` (default 3) nearest other active agents with centres within `range` (default 20 m), nearest first (ties by agent index): position and velocity relative to this agent in the heading frame (m, m/s), then 1; empty slots are all 0. The 1 is neither scaled nor clipped |
//! | `road` | 6 | lateral offset from the lane centre (m, + left), lane heading − heading (rad), lane curvature 5, 10, 20 and 40 m ahead (1/m, + left) |
//! | `route` | 8 | lane centre 5, 10, 20 and 40 m ahead in the heading frame (x, y; m) |
//! | `on_road` | 1 | 1 on a road's surface, else 0 |
//! | `road_class` | 3 | the class of the road whose surface the agent is on, one-hot: paved, gravel, track; all 0 off the road |
//! | `articulation` | 4 | ground vehicles: yaw of the first two trailers (or dollies) relative to the unit ahead (rad, positive pointing left), then their rates (rad/s); 0 without |
//! | `sinkage` | 1 | ground vehicles: mean sinkage of the loaded track patches into soft soil (m); 0 on rigid ground and for tyres |
//! | `trailer_goal` | 4 | goal − tail of the last unit, in that unit's heading frame (x, y; m), then sin, cos of goal heading − the unit's heading (see `Wheeled::tail_pose`) |
//! | `nearest_agent` | 1 | distance between this agent's colliders and the nearest other active agent's, up to `range` (default 20 m) |
//!
//! | `signal` | 4 | ground vehicles on urban maps: the light (one-hot green, amber, red) for the movement ahead (the one nearest the route, else straight on) at the end of the lane followed, if that is a signalized junction, then the distance to its stop line (m); all 0 without |
//! | `traffic` | 15 × `count` | ground vehicles: the `count` (default 8) nearest other road users with centres within `range` (default 50 m), nearest first (ties by agent index, pedestrians after the agents): the other active ground vehicles, and the pedestrians crossing a carriageway (or standing on it, hit). Per slot: footprint centre relative to this agent and velocity, in the heading frame (m, m/s), sin, cos of its heading − the heading, its length and width (m; a pedestrian's diameter), its lane relative to the lane followed (one-hot: same, a lane to the left, to the right, any other or none; pedestrians: none), then its kind (one-hot: vehicle, two-wheeler, pedestrian); empty slots are all 0. The one-hots are neither scaled nor clipped |
//! | `pedestrians` | 6 × `count` | ground vehicles: the `count` (default 8) nearest pedestrians within `range` (default 30 m), nearest first (ties by index): position relative to this agent and velocity, in the heading frame (m, m/s), 1 if on a carriageway (else 0), then 1; empty slots are all 0. The last two are neither scaled nor clipped |
//! | `lane_route` | 6 | ground vehicles on urban maps: the lane changes to the lane the next movement leaves from (+ left; the movement ending nearest the route, else straight on from the lane followed), the distance to the end of the lane followed (m), and the movement's turn (one-hot: left, straight, right, U-turn; neither scaled nor clipped); all 0 without a lane |
//! | `lanes` | 12 | ground vehicles on urban maps: the lane followed 5, 10, 20 and 40 m ahead in the heading frame (x, y; m; through the connectors `signal` picks), its speed limit (m/s), 1 if a lane to its left / right of the same direction exists (else 0), the offset from its centre (m, + left); all 0 without a lane |
//!
//! `road` and `route` follow the lane of the agent's route (`route` goals), or else the lane
//! of the nearest road within 30 m in the direction closer to the heading (keep right; see
//! [`lane`](crate::lane)); with neither they read 0.
//!
//! Sensor terms name their sensor (`sensor = "imu"`) and read zeros until its first reading
//! arrives. Non-finite values are written as 0.
//!
//! **Images**: `camera` terms (`{ term = "camera", sensor = "down", output = "rgb" }`) are not
//! part of the vector. They fill a separate `u8` image of `[height, width, channels]` per
//! agent, their channels concatenated in the order of the terms; all the camera terms of a
//! group must read images of the same size.
//!
//! | `output` | Channels | Value |
//! |---|---|---|
//! | `rgb` | 3 | sRGB colour |
//! | `depth` | 1 | depth along the optical axis over `range` (default 100 m), 0–255; 255 where nothing was hit or beyond `range` |
//! | `semantic` | 1 | semantic class id (`render::SemanticClass`) |
//!
//! Images read zeros until the camera's first frame, which is rendered at the reset (see
//! [`Cameras`](crate::camera::Cameras)).

use crate::interaction::{AgentGrid, AgentShape};
use crate::lane::Follow;
use crate::pedestrians::Pedestrian;
use crate::scenario::Goal;
use crate::traffic::{RoadTrack, Signals, next_connector, next_movement, walk};
use autonomousim_core::math::quat::{from_yaw, rot6d, wrap_angle, yaw};
use autonomousim_sensors::{BodyKinematics, Sensor, SensorConfig, SensorSpec};
use autonomousim_vehicles::aero::AirData;
use autonomousim_vehicles::ground::Wheeled;
use autonomousim_world::lanes::{LaneGraph, Turn};
use autonomousim_world::{Polyline, StaticWorld};
use glam::{DQuat, DVec2, DVec3, EulerRot};
use serde::{Deserialize, Serialize};

/// Largest distance the `clearance` term looks (m).
pub const CLEARANCE_RANGE: f64 = 20.0;

/// Default `range` of the agent terms (m).
pub const NEIGHBOR_RANGE: f64 = 20.0;

/// Default and largest `count` of the `neighbors` term.
pub const NEIGHBOR_COUNT: usize = 3;
pub const MAX_NEIGHBORS: usize = 16;

/// Default `count` and `range` (m) of the `traffic` term.
pub const TRAFFIC_COUNT: usize = 8;
pub const TRAFFIC_RANGE: f64 = 50.0;

/// Width of a road user's slot in the `traffic` term.
const TRAFFIC_SLOT: usize = 15;

/// Default `count` and `range` (m) of the `pedestrians` term.
pub const PEDESTRIAN_COUNT: usize = 8;
pub const PEDESTRIAN_RANGE: f64 = 30.0;

/// Width of a pedestrian's slot in the `pedestrians` term.
const PEDESTRIAN_SLOT: usize = 6;

/// What a road user seen by the `traffic` term is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeenKind {
    Vehicle,
    TwoWheeler,
    Pedestrian,
}

/// Another road user as the `traffic` term sees it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Seen {
    pub kind: SeenKind,
    /// Centre of its footprint (m, world) and its velocity (m/s, world).
    pub center: DVec2,
    pub velocity: DVec2,
    pub heading: f64,
    pub length: f64,
    pub width: f64,
    /// The lane it follows.
    pub lane: Option<u32>,
}

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
    Road,
    Route,
    OnRoad,
    RoadClass,
    Articulation,
    TrailerGoal,
    Sinkage,
    Camera,
    Lean,
    RiderLean,
    Feet,
    AirData,
    WindBody,
    Pitot,
    Signal,
    Lanes,
    Traffic,
    Pedestrians,
    LaneRoute,
}

/// Distances ahead at which the `road` and `route` terms look (m).
pub const LOOKAHEAD: [f64; 4] = [5.0, 10.0, 20.0, 40.0];

impl TermKind {
    /// Sensor kind the term reads, if any.
    fn sensor_kind(self) -> Option<&'static str> {
        use TermKind::*;
        match self {
            Imu | ImuAccel | ImuGyro => Some("imu"),
            GpsPosition | GpsVelocity | GpsGoalRelWorld => Some("gps"),
            BaroAltitude => Some("baro"),
            Pitot => Some("pitot"),
            Mag => Some("mag"),
            Range => Some("rangefinder"),
            Lidar | LidarLog => Some("lidar"),
            Camera => Some("camera"),
            _ => None,
        }
    }

    /// Whether the term reads a ground vehicle's wheels, steering or powertrain.
    fn needs_wheels(self) -> bool {
        use TermKind::*;
        matches!(
            self,
            WheelSpeeds
                | WheelSlip
                | Steering
                | GearRpm
                | Articulation
                | TrailerGoal
                | Sinkage
                | RiderLean
                | Feet
                | Signal
                | Lanes
                | LaneRoute
        )
    }
}

fn one() -> f64 {
    1.0
}

/// Image a `camera` term reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CameraOutput {
    Rgb,
    Depth,
    Semantic,
}

impl CameraOutput {
    pub fn channels(self) -> usize {
        match self {
            CameraOutput::Rgb => 3,
            CameraOutput::Depth | CameraOutput::Semantic => 1,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            CameraOutput::Rgb => "rgb",
            CameraOutput::Depth => "depth",
            CameraOutput::Semantic => "semantic",
        }
    }
}

/// Default `range` of the depth image (m).
pub const DEPTH_RANGE: f64 = 100.0;

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
    /// Range of the agent terms and of the depth image (m).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub range: Option<f64>,
    /// Image of a `camera` term.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<CameraOutput>,
}

impl ObsTerm {
    pub fn new(term: TermKind, scale: f64) -> Self {
        Self { term, scale, clip: None, sensor: None, count: None, range: None, output: None }
    }

    /// A `camera` term reading `output` of camera `sensor`.
    pub fn camera(sensor: &str, output: CameraOutput) -> Self {
        Self { output: Some(output), ..Self::new(TermKind::Camera, 1.0).sensor(sensor) }
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

#[derive(Clone, Debug)]
struct ImageTerm {
    /// `<sensor>/<output>`.
    name: String,
    sensor: usize,
    output: CameraOutput,
    range: f64,
}

/// An observation spec resolved against a group's sensors and action size.
#[derive(Clone, Debug)]
pub struct CompiledObs {
    terms: Vec<Compiled>,
    dim: usize,
    images: Vec<ImageTerm>,
    /// `[height, width, channels]` of the image, if there are camera terms.
    image_shape: Option<[usize; 3]>,
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
    /// The lane line of the agent's route, if it has one.
    pub route: Option<&'a Polyline>,
    /// Where the agent is on the roads, the signals and the time (s).
    pub track: &'a RoadTrack,
    /// Per agent, the ground vehicles as the `traffic` term sees them (empty unless a term
    /// needs them; see [`CompiledObs::needs_traffic`]).
    pub vehicles: &'a [Option<Seen>],
    /// The pedestrians in the world.
    pub pedestrians: &'a [Pedestrian],
    pub signals: &'a Signals,
    pub time: f64,
}

impl CompiledObs {
    pub fn new(
        terms: &[ObsTerm],
        sensors: &[SensorSpec],
        act_dim: usize,
        num_rotors: usize,
        num_wheels: usize,
        steering_head: bool,
    ) -> Result<Self, String> {
        let mut out = Vec::with_capacity(terms.len());
        let mut dim = 0;
        let mut images = Vec::new();
        let mut image_shape: Option<[usize; 3]> = None;
        for t in terms {
            if !t.scale.is_finite() || t.clip.is_some_and(|c| c.is_nan() || c <= 0.0) {
                return Err(format!("invalid scale or clip in {t:?}"));
            }
            if (t.term == TermKind::Camera) != t.output.is_some() {
                return Err(format!("`output` goes with `camera` terms, and they need it: {t:?}"));
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
            if let (Some(output), Some(SensorConfig::Camera(c))) = (t.output, spec) {
                let range = t.range.unwrap_or(DEPTH_RANGE);
                if t.scale != 1.0 || t.clip.is_some() || t.count.is_some() {
                    return Err(format!("camera terms take no scale, clip or count: {t:?}"));
                }
                if (t.range.is_some() && output != CameraOutput::Depth) || !(range > 0.0 && range.is_finite()) {
                    return Err(format!("only depth images take a (positive) range: {t:?}"));
                }
                let (h, w) = (c.height as usize, c.width as usize);
                let shape = image_shape.get_or_insert([h, w, 0]);
                if shape[..2] != [h, w] {
                    return Err(format!("camera terms of a group must have one image size: {t:?} is {w}×{h}"));
                }
                shape[2] += output.channels();
                let name = format!("{}/{}", sensors[sensor].name, output.name());
                images.push(ImageTerm { name, sensor, output, range });
                continue;
            }
            if t.term.needs_wheels() && num_wheels == 0 {
                return Err(format!("observation {:?} needs a ground vehicle", t.term));
            }
            let agent_term = matches!(
                t.term,
                TermKind::Neighbors | TermKind::NearestAgent | TermKind::Traffic | TermKind::Pedestrians
            );
            let counted = matches!(t.term, TermKind::Neighbors | TermKind::Traffic | TermKind::Pedestrians);
            if (t.count.is_some() && !counted) || (t.range.is_some() && !agent_term) {
                return Err(format!("observation {:?} takes no count or range", t.term));
            }
            let (count, range) = match t.term {
                TermKind::Traffic => (TRAFFIC_COUNT, TRAFFIC_RANGE),
                TermKind::Pedestrians => (PEDESTRIAN_COUNT, PEDESTRIAN_RANGE),
                _ => (NEIGHBOR_COUNT, NEIGHBOR_RANGE),
            };
            let (count, range) = (t.count.unwrap_or(count), t.range.unwrap_or(range));
            if agent_term && (!(1..=MAX_NEIGHBORS).contains(&count) || !(range > 0.0 && range.is_finite())) {
                return Err(format!("observation {:?}: count must be 1–{MAX_NEIGHBORS} and range positive", t.term));
            }
            let (d, max_range) = match (t.term, spec) {
                (TermKind::GoalYaw | TermKind::Yaw | TermKind::PitchRoll | TermKind::GearRpm, _) => (2, 0.0),
                (TermKind::Lean | TermKind::RiderLean, _) => (2, 0.0),
                (TermKind::Steering, _) if steering_head => (2, 0.0),
                (
                    TermKind::Height
                    | TermKind::Agl
                    | TermKind::Clearance
                    | TermKind::BaroAltitude
                    | TermKind::Pitot
                    | TermKind::Speed
                    | TermKind::Sideslip
                    | TermKind::Steering
                    | TermKind::Sinkage
                    | TermKind::Feet
                    | TermKind::OnRoad,
                    _,
                ) => (1, 0.0),
                (TermKind::Road, _) => (2 + LOOKAHEAD.len(), 0.0),
                (TermKind::Signal, _) => (4, 0.0),
                (TermKind::Lanes, _) => (2 * LOOKAHEAD.len() + 4, 0.0),
                (TermKind::RoadClass, _) => (3, 0.0),
                (TermKind::Articulation | TermKind::TrailerGoal, _) => (4, 0.0),
                (TermKind::Route, _) => (2 * LOOKAHEAD.len(), 0.0),
                (TermKind::NearestAgent, _) => (1, range),
                (TermKind::Neighbors, _) => (7 * count, range),
                (TermKind::Traffic, _) => (TRAFFIC_SLOT * count, range),
                (TermKind::Pedestrians, _) => (PEDESTRIAN_SLOT * count, range),
                (TermKind::LaneRoute, _) => (6, 0.0),
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
        Ok(Self { terms: out, dim, images, image_shape })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Whether it reads the other vehicles ([`ObsInput::vehicles`]).
    pub fn needs_traffic(&self) -> bool {
        self.terms.iter().any(|t| t.kind == TermKind::Traffic)
    }

    /// `[height, width, channels]` of the image, if there are camera terms.
    pub fn image_shape(&self) -> Option<[usize; 3]> {
        self.image_shape
    }

    /// Bytes of the image (0 without camera terms).
    pub fn image_len(&self) -> usize {
        self.image_shape.map_or(0, |[h, w, c]| h * w * c)
    }

    /// Write the image into `out` (length [`image_len`](Self::image_len)), interleaving the
    /// channels of the terms per pixel.
    pub fn write_image(&self, sensors: &[Sensor], out: &mut [u8]) {
        assert_eq!(out.len(), self.image_len(), "image buffer length");
        let Some([_, _, channels]) = self.image_shape else { return };
        let mut first = 0;
        for t in &self.images {
            let n = t.output.channels();
            let Sensor::Camera(cam) = &sensors[t.sensor] else {
                unreachable!("sensor kinds are checked when the spec is compiled")
            };
            let pixels = out.chunks_exact_mut(channels).map(|p| &mut p[first..first + n]);
            match cam.latest().map(|f| &f.value) {
                None => pixels.for_each(|p| p.fill(0)),
                Some(img) => match t.output {
                    CameraOutput::Rgb => pixels.zip(img.rgb.as_chunks::<3>().0).for_each(|(p, c)| p.copy_from_slice(c)),
                    CameraOutput::Semantic => pixels.zip(&img.class).for_each(|(p, c)| p[0] = *c),
                    CameraOutput::Depth => {
                        let k = 255.0 / t.range as f32;
                        pixels
                            .zip(&img.depth)
                            .for_each(|(p, &d)| p[0] = if d > 0.0 { (d * k).round().min(255.0) as u8 } else { 255 })
                    }
                },
            }
            first += n;
        }
    }

    /// The camera terms as `(name, first channel, channels)`, named `<sensor>/<output>`.
    pub fn image_layout(&self) -> Vec<(String, usize, usize)> {
        let mut off = 0;
        self.images
            .iter()
            .map(|t| {
                let n = t.output.channels();
                off += n;
                (t.name.clone(), off - n, n)
            })
            .collect()
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
                TermKind::AirData => {
                    let air = AirData { wind: k.wind, ..AirData::default() };
                    let f = air.flow(q, k.velocity, k.rates);
                    put(dst, &[f.airspeed, f.alpha, f.beta], t)
                }
                TermKind::WindBody => put3(dst, q.inverse() * k.wind),
                TermKind::Sideslip => {
                    let v = q.inverse() * k.velocity;
                    put(dst, &[v.y.atan2(v.x.abs().max(1.0))], t)
                }
                TermKind::PitchRoll => {
                    let (_, pitch, roll) = q.to_euler(EulerRot::ZYX);
                    put(dst, &[pitch, roll], t)
                }
                TermKind::Lean => {
                    // The yaw–pitch–roll angles' roll and its rate from the body rates.
                    let (_, pitch, roll) = q.to_euler(EulerRot::ZYX);
                    let w = k.rates;
                    let rate = w.x + (w.y * roll.sin() + w.z * roll.cos()) * pitch.tan();
                    put(dst, &[roll, rate], t)
                }
                TermKind::WheelSpeeds
                | TermKind::WheelSlip
                | TermKind::Steering
                | TermKind::GearRpm
                | TermKind::Articulation
                | TermKind::TrailerGoal
                | TermKind::Sinkage
                | TermKind::RiderLean
                | TermKind::Feet => {
                    let v = inp.wheeled.expect("ground terms are checked when the spec is compiled");
                    match t.kind {
                        TermKind::Articulation => {
                            let mut a = [0.0; 4];
                            for (k, (angle, rate)) in v.articulations().take(2).enumerate() {
                                (a[k], a[2 + k]) = (angle, rate);
                            }
                            put(dst, &a, t)
                        }
                        TermKind::TrailerGoal => {
                            let tail = v.tail_pose();
                            let y = yaw(tail.rot);
                            let r = from_yaw(y).inverse() * (inp.goal.position - tail.pos);
                            let e = wrap_angle(inp.goal.yaw - y);
                            put(dst, &[r.x, r.y, e.sin(), e.cos()], t)
                        }
                        TermKind::WheelSpeeds => {
                            for (w, (d, s)) in dst.iter_mut().zip(v.wheels()).enumerate() {
                                *d = value(s.spin * v.def().wheel_tire(w).radius(), t);
                            }
                        }
                        TermKind::WheelSlip => {
                            for (d, s) in dst.iter_mut().zip(v.wheels()) {
                                *d = value(s.tire.kappa, t);
                            }
                        }
                        TermKind::Steering => match v.steering_head() {
                            Some((angle, rate)) => put(dst, &[angle, rate], t),
                            None => put(dst, &[v.steering_angle()], t),
                        },
                        TermKind::RiderLean => {
                            let (angle, rate) = v.rider_lean();
                            put(dst, &[angle, rate], t)
                        }
                        TermKind::Feet => put(dst, &[f64::from(u8::from(v.feet_down()))], t),
                        TermKind::Sinkage => put(dst, &[v.sinkage()], t),
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
                TermKind::Road | TermKind::Route => {
                    let y = yaw(q);
                    let Some(f) = Follow::new(inp.route, inp.world, k.position.truncate(), y) else {
                        dst.fill(0.0);
                        continue;
                    };
                    if t.kind == TermKind::Road {
                        let mut v = [0.0; 2 + LOOKAHEAD.len()];
                        v[0] = f.offset;
                        v[1] = wrap_angle(f.heading(0.0) - y);
                        for (d, &a) in v[2..].iter_mut().zip(&LOOKAHEAD) {
                            *d = f.curvature(a);
                        }
                        put(dst, &v, t)
                    } else {
                        let mut v = [0.0; 2 * LOOKAHEAD.len()];
                        let to_heading = heading.inverse();
                        for (d, &a) in v.as_chunks_mut::<2>().0.iter_mut().zip(&LOOKAHEAD) {
                            let r = to_heading * (f.point(a) - k.position.truncate()).extend(0.0);
                            *d = [r.x, r.y];
                        }
                        put(dst, &v, t)
                    }
                }
                TermKind::Signal => {
                    let mut v = [0.0; 4];
                    let lanes = inp.world.roads().lanes();
                    if let Some(l) = inp.track.lane
                        && inp.track.past_end < 0.0
                        && lanes.junction_controller(lanes.lanes()[l as usize].to_node).is_some()
                        && let Some(c) = next_connector(lanes, l, inp.route)
                    {
                        v[inp.signals.light(lanes, c, inp.time) as usize] = 1.0;
                        v[3] = -inp.track.past_end;
                    }
                    put(dst, &v, t)
                }
                TermKind::Lanes => {
                    let Some(l) = inp.track.lane else {
                        dst.fill(0.0);
                        continue;
                    };
                    let lanes = inp.world.roads().lanes();
                    let lane = &lanes.lanes()[l as usize];
                    let mut v = [0.0; 2 * LOOKAHEAD.len() + 4];
                    let to_heading = heading.inverse();
                    let (pts, rest) = v.split_at_mut(2 * LOOKAHEAD.len());
                    for (d, &a) in pts.as_chunks_mut::<2>().0.iter_mut().zip(&LOOKAHEAD) {
                        let p = walk(lanes, l, inp.track.station, a, inp.route);
                        let r = to_heading * (p - k.position.truncate()).extend(0.0);
                        *d = [r.x, r.y];
                    }
                    rest[0] = lane.speed;
                    rest[1] = f64::from(u8::from(lane.left.is_some()));
                    rest[2] = f64::from(u8::from(lane.right.is_some()));
                    rest[3] = inp.track.offset;
                    put(dst, &v, t)
                }
                TermKind::Traffic => {
                    dst.fill(0.0);
                    let me = k.position.truncate();
                    let mut near: Vec<(f64, usize)> = (inp.vehicles.iter().enumerate())
                        .filter(|&(j, _)| j != inp.me)
                        .filter_map(|(j, s)| s.map(|s| (s.center.distance(me), j)))
                        .filter(|&(d, _)| d <= t.max_range)
                        .collect();
                    near.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                    let lanes = inp.world.roads().lanes();
                    let y = yaw(q);
                    let to_heading = heading.inverse();
                    for (slot, &(_, j)) in dst.as_chunks_mut::<TRAFFIC_SLOT>().0.iter_mut().zip(&near) {
                        let o = inp.vehicles[j].expect("listed");
                        let r = to_heading * (o.center - me).extend(0.0);
                        let v = to_heading * o.velocity.extend(0.0);
                        let h = wrap_angle(o.heading - y);
                        put(&mut slot[0..8], &[r.x, r.y, v.x, v.y, h.sin(), h.cos(), o.length, o.width], t);
                        slot[8 + lane_relation(lanes, inp.track.lane, o.lane)] = 1.0;
                        slot[12 + o.kind as usize] = 1.0;
                    }
                }
                TermKind::Pedestrians => {
                    dst.fill(0.0);
                    let me = k.position.truncate();
                    let r2 = t.max_range * t.max_range;
                    let mut near: Vec<(f64, usize)> = (inp.pedestrians.iter().enumerate())
                        .map(|(j, p)| (p.pos.distance_squared(me), j))
                        .filter(|&(d, _)| d <= r2)
                        .collect();
                    near.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                    let to_heading = heading.inverse();
                    let roads = inp.world.roads();
                    for (slot, &(_, j)) in dst.as_chunks_mut::<PEDESTRIAN_SLOT>().0.iter_mut().zip(&near) {
                        let p = &inp.pedestrians[j];
                        let r = to_heading * (p.pos - me).extend(0.0);
                        let v = to_heading * p.vel.extend(0.0);
                        put(&mut slot[0..4], &[r.x, r.y, v.x, v.y], t);
                        slot[4] = f32::from(u8::from(roads.on_road(p.pos).is_some()));
                        slot[5] = 1.0;
                    }
                }
                TermKind::LaneRoute => {
                    let mut v = [0.0; 6];
                    let lanes = inp.world.roads().lanes();
                    if let Some(l) = inp.track.lane
                        && let Some((c, changes)) = next_movement(lanes, l, inp.route)
                    {
                        v[0] = f64::from(changes);
                        v[1] = (lanes.lanes()[l as usize].line.length() - inp.track.station).max(0.0);
                        v[2 + match lanes.connectors()[c as usize].turn {
                            Turn::Left => 0,
                            Turn::Straight => 1,
                            Turn::Right => 2,
                            Turn::UTurn => 3,
                        }] = 1.0;
                    }
                    put(dst, &v[..2], t);
                    for (d, x) in dst[2..].iter_mut().zip(&v[2..]) {
                        *d = *x as f32;
                    }
                }
                TermKind::OnRoad => {
                    let on = inp.world.roads().on_road(k.position.truncate()).is_some();
                    put(dst, &[f64::from(u8::from(on))], t)
                }
                TermKind::RoadClass => {
                    let roads = inp.world.roads();
                    let mut v = [0.0; 3];
                    if let Some(rp) = roads.on_road(k.position.truncate()) {
                        v[roads.roads()[rp.road as usize].class as usize] = 1.0;
                    }
                    put(dst, &v, t)
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
        (TermKind::Pitot, Sensor::Pitot(s)) => match s.latest() {
            Some(r) => put(dst, &[r.value.indicated_airspeed], t),
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

/// Where lane `theirs` lies from lane `mine`: 0 the same, 1 to the left, 2 to the right (lanes
/// of the same direction beside it), 3 anything else.
fn lane_relation(lanes: &LaneGraph, mine: Option<u32>, theirs: Option<u32>) -> usize {
    let (Some(a), Some(b)) = (mine, theirs) else { return 3 };
    if a == b {
        return 0;
    }
    let beside = |step: fn(&autonomousim_world::Lane) -> Option<u32>| {
        std::iter::successors(step(&lanes.lanes()[a as usize]), |&l| step(&lanes.lanes()[l as usize]))
            .take(8)
            .any(|l| l == b)
    };
    if beside(|l| l.left) {
        1
    } else if beside(|l| l.right) {
        2
    } else {
        3
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use autonomousim_sensors::{ImuConfig, LidarConfig};
    use autonomousim_world::testworlds;

    #[test]
    fn terms_resolve_and_write() {
        let sensors = vec![
            SensorSpec { name: "imu".into(), unit: 0, config: SensorConfig::Imu(ImuConfig::ideal()) },
            SensorSpec { name: "lidar".into(), unit: 0, config: SensorConfig::Lidar(LidarConfig::rl64()) },
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
        let obs = CompiledObs::new(&terms, &sensors, 4, 4, 0, false).unwrap();
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
            route: None,
            track: &RoadTrack::default(),
            vehicles: &[],
            pedestrians: &[],
            signals: &Signals::default(),
            time: 0.0,
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
            assert!(CompiledObs::new(&[bad], &sensors, 4, 4, 0, false).is_err());
        }
    }

    #[test]
    fn air_terms() {
        let terms = [ObsTerm::new(TermKind::AirData, 1.0), ObsTerm::new(TermKind::WindBody, 1.0)];
        let obs = CompiledObs::new(&terms, &[], 4, 4, 0, false).unwrap();
        assert_eq!(obs.dim(), 6);
        let world = testworlds::flat(50.0);
        // Heading north at 20 m/s, 2 m/s sinking, with a 5 m/s wind from the west.
        let att = DQuat::from_rotation_z(std::f64::consts::FRAC_PI_2);
        let kin = BodyKinematics {
            attitude: att,
            velocity: DVec3::new(0.0, 20.0, -2.0),
            wind: DVec3::new(5.0, 0.0, 0.0),
            ..Default::default()
        };
        let inp = ObsInput {
            kin: &kin,
            goal: Goal { position: DVec3::ZERO, yaw: 0.0 },
            agl: 10.0,
            motors: &[],
            motor_range: (0.0, 1.0),
            last_action: &[0.0; 4],
            wheeled: None,
            sensors: &[],
            world: &world,
            agents: &[],
            me: 0,
            grid: &AgentGrid::default(),
            route: None,
            track: &RoadTrack::default(),
            vehicles: &[],
            pedestrians: &[],
            signals: &Signals::default(),
            time: 0.0,
        };
        let mut out = vec![f32::NAN; 6];
        obs.write(&inp, &mut out);
        // Relative to the air: 20 m/s ahead, 5 m/s to the left (air from the left: β < 0).
        let v = DVec3::new(20.0, 5.0, -2.0);
        let want = [v.length(), 2f64.atan2(20.0), (-5.0 / v.length()).asin(), 0.0, -5.0, 0.0];
        for (o, w) in out.iter().zip(want) {
            assert!((f64::from(*o) - w).abs() < 1e-5, "{out:?} vs {want:?}");
        }
    }
}
