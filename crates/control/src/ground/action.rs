//! Normalised action modes of ground vehicles: policy outputs in `[−1, 1]` → setpoints.
//!
//! | Mode | Components | Meaning at ±1 |
//! |---|---|---|
//! | `raw` | drive, steering (steered vehicles) | pedal: full throttle (+1), full brake then full reverse throttle (−1); full lock |
//! | `raw` | left, right (side drives) | full motor command per side |
//! | `vk` | speed, curvature | `speed` forward (+1) or `reverse` backward (−1); ±`curvature` (1/m, + left) |
//! | `vw` | speed, yaw rate (side drives only) | as `vk`; ±`yaw_rate` (rad/s) |
//! | `per_wheel` | the vehicle's own channels, see below | full command per channel |
//!
//! `per_wheel` exposes only channels the vehicle has, in this order: `throttle` (combustion
//! drives, `[0, 1]`), `steering` (axles steered by the Ackermann linkage), `drive_<wheels>` (one per electric
//! motor, named by its wheels), `brake_<w>` (wheels with a service brake, `[0, 1]`) and
//! `steer_<w>` (wheels on independently steered axles, fraction of the axle's lock). One-sided
//! channels read negative actions as 0.
//!
//! Components outside `[−1, 1]` are clipped and non-finite ones read as 0.

use super::GroundSetpoint;
use crate::ControlError;
use autonomousim_vehicles::ground::{
    DriveInput, MAX_WHEELS, MotorSide, PowertrainDef, SteerMode, WheelCommands, WheeledDef,
};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

const RPM: f64 = std::f64::consts::PI / 30.0;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GroundActionMode {
    Raw,
    /// Speed and curvature.
    #[default]
    Vk,
    /// Speed and yaw rate.
    Vw,
    PerWheel,
}

impl GroundActionMode {
    pub const ALL: [GroundActionMode; 4] =
        [GroundActionMode::Raw, GroundActionMode::Vk, GroundActionMode::Vw, GroundActionMode::PerWheel];

    pub fn name(self) -> &'static str {
        match self {
            GroundActionMode::Raw => "raw",
            GroundActionMode::Vk => "vk",
            GroundActionMode::Vw => "vw",
            GroundActionMode::PerWheel => "per_wheel",
        }
    }
}

impl fmt::Display for GroundActionMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for GroundActionMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::ALL.into_iter().find(|m| m.name() == s).ok_or_else(|| format!("unknown ground action mode {s:?}"))
    }
}

/// Full-scale values of the normalised actions; omitted ones follow from the vehicle (see
/// [`GroundActionMap::new`]).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GroundActionLimits {
    /// Forward and reverse speed (m/s).
    pub speed: Option<f64>,
    pub reverse: Option<f64>,
    /// Path curvature (1/m).
    pub curvature: Option<f64>,
    /// Yaw rate (rad/s).
    pub yaw_rate: Option<f64>,
}

/// One `per_wheel` channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PerWheelChannel {
    Throttle,
    Steering,
    /// Motor driving the wheels in the bit mask.
    Drive(u32),
    Brake(usize),
    Steer(usize),
}

impl PerWheelChannel {
    pub fn name(self) -> String {
        match self {
            PerWheelChannel::Throttle => "throttle".into(),
            PerWheelChannel::Steering => "steering".into(),
            PerWheelChannel::Drive(mask) => {
                let wheels: Vec<String> =
                    (0..MAX_WHEELS).filter(|w| mask >> w & 1 == 1).map(|w| w.to_string()).collect();
                format!("drive_{}", wheels.join("_"))
            }
            PerWheelChannel::Brake(w) => format!("brake_{w}"),
            PerWheelChannel::Steer(w) => format!("steer_{w}"),
        }
    }
}

/// Maps normalised actions of one [`GroundActionMode`] to setpoints for one vehicle.
#[derive(Clone, Debug)]
pub struct GroundActionMap {
    mode: GroundActionMode,
    side_drive: bool,
    speed: f64,
    reverse: f64,
    curvature: f64,
    yaw_rate: f64,
    channels: Vec<PerWheelChannel>,
}

impl GroundActionMap {
    /// Defaults of omitted limits: `speed` the top speed (engine or motor no-load speed in the
    /// top gear) up to 20 m/s; `reverse` 0.3·`speed` for engines and `speed` for electric drives;
    /// `curvature` 95 % of the tightest bicycle-model curvature, or 2/track for side drives;
    /// `yaw_rate` 0.8·`speed`/track for side drives, `speed`·`curvature` otherwise.
    pub fn new(mode: GroundActionMode, limits: &GroundActionLimits, def: &WheeledDef) -> Result<Self, ControlError> {
        let n = def.num_wheels();
        let radius = def.tire(0).radius();
        let (top, combustion, side_drive) = match &def.powertrain {
            PowertrainDef::Combustion(c) => {
                let top_gear = c.gearbox.forward.iter().copied().fold(0.0, f64::max);
                (c.engine.max_rpm * RPM * top_gear * c.final_drive * radius, true, false)
            }
            PowertrainDef::Electric(e) => {
                let top = e
                    .motors
                    .iter()
                    .filter_map(|m| m.no_load_speed.map(|w0| w0 * m.ratio * def.tire(m.wheels[0] / 2).radius()))
                    .fold(f64::INFINITY, f64::min);
                let left = e.motors.iter().any(|m| m.side == MotorSide::Left);
                let right = e.motors.iter().any(|m| m.side == MotorSide::Right);
                let both = e.motors.iter().any(|m| m.side == MotorSide::Both);
                (top, false, left && right && !both)
            }
        };
        let track = 2.0 * def.axles.iter().map(|a| a.position.y).sum::<f64>() / def.axles.len() as f64;
        let speed = limits.speed.unwrap_or(top.min(20.0));
        let reverse = limits.reverse.unwrap_or(if combustion { 0.3 * speed } else { speed });
        let tightest = def.steering.and_then(|s| {
            let unsteered: Vec<f64> = def.axles.iter().filter(|a| a.steer == 0.0).map(|a| a.position.x).collect();
            let axle = def.axles.iter().max_by(|a, b| a.steer.abs().total_cmp(&b.steer.abs()))?;
            let reference = unsteered.iter().sum::<f64>() / unsteered.len().max(1) as f64;
            let wheelbase = (axle.position.x - reference).abs();
            (wheelbase > 0.0).then(|| (s.max_angle * axle.steer.abs()).tan() / wheelbase)
        });
        let curvature = limits.curvature.unwrap_or(match tightest {
            Some(k) => 0.95 * k,
            None => 2.0 / track,
        });
        let yaw_rate = limits.yaw_rate.unwrap_or(if side_drive { 0.8 * speed / track } else { speed * curvature });
        let pos = |x: f64| x > 0.0 && x.is_finite();
        if !(pos(speed) && pos(reverse) && pos(curvature) && pos(yaw_rate)) {
            return Err(ControlError::InvalidConfig(format!(
                "{}: action limits must be positive (speed {speed}, reverse {reverse}, curvature {curvature}, yaw rate {yaw_rate})",
                def.name
            )));
        }
        if mode == GroundActionMode::Vw && !side_drive {
            return Err(ControlError::InvalidConfig(format!(
                "{}: vw needs a skid-steer or diff-drive vehicle",
                def.name
            )));
        }
        let mut channels = Vec::new();
        if mode == GroundActionMode::PerWheel {
            if combustion {
                channels.push(PerWheelChannel::Throttle);
            }
            if def.axles.iter().any(|a| a.steer != 0.0 && a.steer_mode == SteerMode::Ackermann) {
                channels.push(PerWheelChannel::Steering);
            }
            if let PowertrainDef::Electric(e) = &def.powertrain {
                channels.extend(
                    e.motors.iter().map(|m| PerWheelChannel::Drive(m.wheels.iter().fold(0, |b, &w| b | 1 << w))),
                );
            }
            channels.extend((0..n).filter(|&w| def.axles[w / 2].brake.max_torque > 0.0).map(PerWheelChannel::Brake));
            channels.extend(
                (0..n)
                    .filter(|&w| matches!(def.axles[w / 2].steer_mode, SteerMode::Independent { .. }))
                    .map(PerWheelChannel::Steer),
            );
        }
        Ok(Self { mode, side_drive, speed, reverse, curvature, yaw_rate, channels })
    }

    pub fn mode(&self) -> GroundActionMode {
        self.mode
    }

    /// Resolved full-scale values: speed, reverse speed (m/s), curvature (1/m), yaw rate (rad/s).
    pub fn speed(&self) -> f64 {
        self.speed
    }

    pub fn reverse(&self) -> f64 {
        self.reverse
    }

    pub fn curvature(&self) -> f64 {
        self.curvature
    }

    pub fn yaw_rate(&self) -> f64 {
        self.yaw_rate
    }

    /// The `per_wheel` channels (empty in other modes).
    pub fn channels(&self) -> &[PerWheelChannel] {
        &self.channels
    }

    pub fn dim(&self) -> usize {
        match self.mode {
            GroundActionMode::PerWheel => self.channels.len(),
            _ => 2,
        }
    }

    /// Names of the action components.
    pub fn names(&self) -> Vec<String> {
        let two = |a: &str, b: &str| vec![a.to_string(), b.to_string()];
        match self.mode {
            GroundActionMode::Raw if self.side_drive => two("left", "right"),
            GroundActionMode::Raw => two("drive", "steering"),
            GroundActionMode::Vk => two("speed", "curvature"),
            GroundActionMode::Vw => two("speed", "yaw_rate"),
            GroundActionMode::PerWheel => self.channels.iter().map(|c| c.name()).collect(),
        }
    }

    /// Setpoint for `action` (length [`dim`](Self::dim)).
    pub fn setpoint(&self, action: &[f64]) -> GroundSetpoint {
        assert_eq!(action.len(), self.dim(), "{} action length", self.mode);
        let a = |i: usize| if action[i].is_finite() { action[i].clamp(-1.0, 1.0) } else { 0.0 };
        let speed = |x: f64| if x >= 0.0 { x * self.speed } else { x * self.reverse };
        match self.mode {
            GroundActionMode::Raw if self.side_drive => GroundSetpoint::Sides { left: a(0), right: a(1) },
            GroundActionMode::Raw => GroundSetpoint::Pedal { drive: a(0), steering: a(1), handbrake: false },
            GroundActionMode::Vk => {
                GroundSetpoint::SpeedCurvature { speed: speed(a(0)), curvature: a(1) * self.curvature }
            }
            GroundActionMode::Vw => GroundSetpoint::SpeedYawRate { speed: speed(a(0)), yaw_rate: a(1) * self.yaw_rate },
            GroundActionMode::PerWheel => {
                let mut input = DriveInput::default();
                let mut wheels = WheelCommands::default();
                for (i, c) in self.channels.iter().enumerate() {
                    let x = a(i);
                    match *c {
                        PerWheelChannel::Throttle => input.throttle = x.max(0.0),
                        PerWheelChannel::Steering => input.steering = x,
                        PerWheelChannel::Drive(mask) => {
                            for w in (0..MAX_WHEELS).filter(|w| mask >> w & 1 == 1) {
                                wheels.drive[w] = x;
                            }
                        }
                        PerWheelChannel::Brake(w) => wheels.brake[w] = x.max(0.0),
                        PerWheelChannel::Steer(w) => wheels.steer[w] = x,
                    }
                }
                input.wheels = Some(wheels);
                GroundSetpoint::Direct(input)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use autonomousim_vehicles::presets;

    fn map(mode: GroundActionMode, name: &str) -> Result<GroundActionMap, ControlError> {
        GroundActionMap::new(mode, &GroundActionLimits::default(), &presets::wheeled(name).unwrap())
    }

    #[test]
    fn modes_names_and_limits() {
        for m in GroundActionMode::ALL {
            assert_eq!(m.name().parse::<GroundActionMode>().unwrap(), m);
        }
        let car = map(GroundActionMode::Vk, "sedan_like").unwrap();
        assert_eq!(car.speed(), 20.0);
        assert!((car.reverse() - 6.0).abs() < 1e-12);
        assert!(car.curvature() > 0.1 && car.curvature() < 0.3, "{}", car.curvature());
        let robot = map(GroundActionMode::Vw, "rover_diff").unwrap();
        assert!((robot.speed() - 367.0 * 0.05 * 0.1).abs() < 0.2, "{}", robot.speed());
        assert_eq!(robot.names(), ["speed", "yaw_rate"]);
        assert_eq!(map(GroundActionMode::Raw, "rover_skid").unwrap().names(), ["left", "right"]);
        assert_eq!(map(GroundActionMode::Raw, "sedan_like").unwrap().names(), ["drive", "steering"]);
        assert!(map(GroundActionMode::Vw, "sedan_like").is_err());
    }

    #[test]
    fn setpoints_scale_and_clip() {
        let car = map(GroundActionMode::Vk, "sedan_like").unwrap();
        let GroundSetpoint::SpeedCurvature { speed, curvature } = car.setpoint(&[-0.5, f64::NAN]) else { panic!() };
        assert!((speed + 3.0).abs() < 1e-12 && curvature == 0.0);
        let GroundSetpoint::SpeedCurvature { speed, curvature } = car.setpoint(&[7.0, -1.0]) else { panic!() };
        assert!(speed == 20.0 && curvature == -car.curvature());
        let GroundSetpoint::Sides { left, right } =
            map(GroundActionMode::Raw, "rover_skid").unwrap().setpoint(&[0.5, -2.0])
        else {
            panic!()
        };
        assert!(left == 0.5 && right == -1.0);
    }

    #[test]
    fn per_wheel_has_only_the_vehicles_channels() {
        let names = |n: &str| map(GroundActionMode::PerWheel, n).unwrap().names();
        assert_eq!(names("sedan_like"), ["throttle", "steering", "brake_0", "brake_1", "brake_2", "brake_3"]);
        assert_eq!(names("rover_diff"), ["drive_0", "drive_1", "brake_0", "brake_1"]);
        let skid = names("rover_skid");
        assert_eq!(&skid[..2], ["drive_0_2", "drive_1_3"]);
        let m = map(GroundActionMode::PerWheel, "sedan_like").unwrap();
        let GroundSetpoint::Direct(input) = m.setpoint(&[-1.0, 0.5, 1.0, -1.0, 0.0, 0.25]) else { panic!() };
        let w = input.wheels.unwrap();
        assert_eq!((input.throttle, input.steering), (0.0, 0.5));
        assert_eq!(&w.brake[..4], [1.0, 0.0, 0.0, 0.25]);
    }
}
