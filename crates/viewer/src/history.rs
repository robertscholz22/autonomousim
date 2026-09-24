//! Flight data for the HUD plots: height, distance to the goal, and what the cascade was asked
//! for against what the vehicle did, at the level where the setpoint enters the cascade. For
//! ground vehicles: yaw rate and speed against their references, sideslip, and every tyre's
//! slip and force.
//!
//! Live, the followed agent is sampled every frame over the last [`WINDOW`] seconds; a replay
//! is turned into curves over the whole episode (setpoints rebuilt from the recorded actions).

use crate::replay::Replay;
use crate::sim::Sim;
use autonomousim_control::ground::GroundSetpoint;
use autonomousim_control::multirotor::{Frame, Setpoint, StateEstimate};
use autonomousim_core::math::quat::yaw;
use autonomousim_sim::WorldInstance;
use autonomousim_world::StaticWorld;
use bevy::prelude::*;
use glam::{DQuat, DVec3};
use std::collections::VecDeque;
use std::sync::Arc;

/// Seconds of live history kept.
pub const WINDOW: f64 = 30.0;

/// Which quantity the command plot shows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Tracking {
    #[default]
    Motors,
    /// Body rates (°/s).
    Rates,
    /// Tilt in the heading frame (°; x: roll right-down is negative, y: nose down).
    Tilt,
    /// Velocity (m/s) in the setpoint's frame.
    Velocity,
    /// World position (m).
    Position,
    /// Ground vehicles: yaw rate (°/s) and forward speed (m/s).
    Ground,
}

impl Tracking {
    pub fn label(self) -> &'static str {
        match self {
            Tracking::Motors => "rotor speed (rad/s)",
            Tracking::Rates => "body rates (°/s)",
            Tracking::Tilt => "tilt (°)",
            Tracking::Velocity => "velocity (m/s)",
            Tracking::Position => "position (m)",
            Tracking::Ground => "yaw rate (°/s) and speed (m/s)",
        }
    }

    /// Names of the three components.
    pub fn axes(self) -> [&'static str; 3] {
        match self {
            Tracking::Motors => ["1", "2", "3"],
            Tracking::Rates => ["roll", "pitch", "yaw"],
            Tracking::Tilt => ["x", "y", "-"],
            Tracking::Velocity | Tracking::Position => ["x", "y", "z"],
            Tracking::Ground => ["yaw rate", "speed", "-"],
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Sample {
    pub time: f64,
    /// Height above the ground or water (m).
    pub agl: f64,
    /// Distance to the current goal (m).
    pub goal_distance: f64,
    /// Setpoint and the matching measured quantity (NaN where there is none).
    pub command: [f64; 3],
    pub actual: [f64; 3],
    /// Ground vehicles: sideslip angle (°) and per wheel tyre slip and force.
    pub sideslip: f64,
    pub wheels: Vec<WheelSample>,
}

/// A tyre's slip and force in its contact frame.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WheelSample {
    /// Longitudinal slip κ and slip angle (°).
    pub kappa: f64,
    pub alpha: f64,
    /// Longitudinal and lateral force over the vertical load.
    pub fx: f64,
    pub fy: f64,
}

impl WheelSample {
    /// From `[κ, tan α, Fx, Fy, Fz]` per wheel. Wheels carrying less than a fifth of the mean
    /// load (nearly airborne) are left out (NaN): their force ratios mean little.
    fn all(wheels: impl Iterator<Item = [f64; 5]>) -> Vec<Self> {
        let w: Vec<[f64; 5]> = wheels.collect();
        let mean = w.iter().map(|x| x[4]).sum::<f64>() / w.len().max(1) as f64;
        w.iter()
            .map(|&[kappa, tan_alpha, fx, fy, fz]| {
                if fz > 0.2 * mean && fz > 0.0 {
                    Self { kappa, alpha: tan_alpha.atan().to_degrees(), fx: fx / fz, fy: fy / fz }
                } else {
                    Self { kappa: f64::NAN, alpha: f64::NAN, fx: f64::NAN, fy: f64::NAN }
                }
            })
            .collect()
    }
}

/// Yaw rate (°/s) and forward speed (m/s) of a ground vehicle and their references: the
/// commanded speed and yaw rate (the curvature times the speed) of speed modes, else the
/// kinematic yaw rate `v·tan δ/L` of the bicycle steering angle `δ` (without wheelbase: none).
pub fn ground_tracking(
    setpoint: Option<&GroundSetpoint>,
    speed: f64,
    yaw_rate: f64,
    steering: f64,
    wheelbase: Option<f64>,
) -> ([f64; 3], [f64; 3]) {
    let kinematic = wheelbase.map_or(f64::NAN, |l| speed * steering.tan() / l);
    let (r_ref, v_ref) = match setpoint {
        Some(&GroundSetpoint::SpeedCurvature { speed: v, curvature }) => (speed * curvature, v),
        Some(&GroundSetpoint::SpeedYawRate { speed: v, yaw_rate }) => (yaw_rate, v),
        _ => (kinematic, f64::NAN),
    };
    ([r_ref.to_degrees(), v_ref, f64::NAN], [yaw_rate.to_degrees(), speed, f64::NAN])
}

/// Sideslip angle (°) from the velocity in the vehicle frame; NaN below 1 m/s.
fn sideslip(v_body: DVec3) -> f64 {
    if v_body.truncate().length() < 1.0 { f64::NAN } else { v_body.y.atan2(v_body.x.abs()).to_degrees() }
}

/// Setpoint components and the measured quantity they ask for.
pub fn tracking(setpoint: &Setpoint, est: &StateEstimate, motors: &[f64]) -> (Tracking, [f64; 3], [f64; 3]) {
    let deg = |v: DVec3| (v * 180.0 / std::f64::consts::PI).to_array();
    match *setpoint {
        Setpoint::Motors(w) => {
            let actual = [0, 1, 2].map(|i| motors.get(i).copied().unwrap_or(f64::NAN));
            (Tracking::Motors, [w[0], w[1], w[2]], actual)
        }
        Setpoint::Ctbr { rates, .. } => (Tracking::Rates, deg(rates), deg(est.rates)),
        Setpoint::Attitude { tilt, .. } => {
            let z = DQuat::from_rotation_z(-yaw(est.attitude)) * (est.attitude * DVec3::Z);
            let angle = z.z.clamp(-1.0, 1.0).acos();
            let scale = if angle > 1e-9 { angle / angle.sin() } else { 1.0 };
            let actual = DVec3::new(-z.y * scale, z.x * scale, f64::NAN);
            (Tracking::Tilt, deg(tilt.extend(f64::NAN)), deg(actual))
        }
        Setpoint::Velocity { velocity, frame, .. } => {
            let actual = match frame {
                Frame::World => est.velocity,
                Frame::Heading => DQuat::from_rotation_z(-yaw(est.attitude)) * est.velocity,
            };
            (Tracking::Velocity, velocity.to_array(), actual.to_array())
        }
        Setpoint::Position { position, .. } => (Tracking::Position, position.to_array(), est.position.to_array()),
    }
}

/// Plot data of the followed agent.
#[derive(Resource, Default)]
pub struct History {
    pub tracking: Tracking,
    pub samples: VecDeque<Sample>,
    /// What the samples belong to: agent, map, and for replays the episode.
    agent: usize,
    map: Option<Arc<StaticWorld>>,
    episode: Option<usize>,
}

impl History {
    fn matches(&self, agent: usize, map: &Arc<StaticWorld>, episode: Option<usize>) -> bool {
        self.agent == agent && self.episode == episode && self.map.as_ref().is_some_and(|m| Arc::ptr_eq(m, map))
    }

    fn restart(&mut self, agent: usize, map: &Arc<StaticWorld>, episode: Option<usize>) {
        self.samples.clear();
        self.agent = agent;
        self.map = Some(map.clone());
        self.episode = episode;
    }
}

/// A live sample of agent `i`.
fn live_sample(world: &WorldInstance, i: usize) -> (Tracking, Sample) {
    let agent = world.agent(i);
    let v = &agent.vehicle;
    let (mut sideslip_deg, mut wheels) = (f64::NAN, Vec::new());
    let (tracking, command, actual) = match (v.as_multirotor(), agent.command().as_multirotor()) {
        (Some(m), Some(sp)) => tracking(sp, &StateEstimate::of(m), m.motor_speeds()),
        _ => match v.as_wheeled() {
            Some(w) => {
                let vb = w.lin_vel_body();
                sideslip_deg = sideslip(vb);
                wheels = WheelSample::all(
                    w.wheels().map(|s| [s.tire.kappa, s.tire.tan_alpha, s.tire.fx, s.tire.fy, s.tire.fz]),
                );
                let wheelbase = agent.controller.as_ground().and_then(|c| c.wheelbase());
                let (c, a) = ground_tracking(
                    agent.command().as_ground(),
                    vb.x,
                    w.ang_vel_body().z,
                    w.steering_angle(),
                    wheelbase,
                );
                (Tracking::Ground, c, a)
            }
            None => (Tracking::default(), [f64::NAN; 3], [f64::NAN; 3]),
        },
    };
    let sample = Sample {
        time: world.time(),
        agl: agent.agl_now(world.map()),
        goal_distance: (v.position() - agent.goal().position).length(),
        command,
        actual,
        sideslip: sideslip_deg,
        wheels,
    };
    (tracking, sample)
}

/// Curves of `agent` over the playback's episode: every recorded state, with the setpoint
/// of the action in force at that time.
fn episode_samples(replay: &Replay, world: &WorldInstance, agent: usize) -> (Tracking, Vec<Sample>) {
    let ep = replay.current();
    let map = &world.scenario().maps[ep.map];
    let group = &world.scenario().groups[world.agent(agent).group];
    if let Some(ground) = group.action_map.as_ground() {
        return (Tracking::Ground, ground_episode_samples(replay, world, agent, map, ground));
    }
    let Some(action_map) = group.action_map.as_multirotor() else { return (Tracking::default(), Vec::new()) };
    let (Some(states), actions) = (ep.states.get(agent), ep.actions.get(agent).map_or(&[][..], |a| &a[..])) else {
        return (Tracking::default(), Vec::new());
    };
    let mut kind = Tracking::default();
    let samples = states
        .iter()
        .map(|s| {
            let est = StateEstimate {
                position: s.position,
                velocity: s.velocity,
                attitude: s.orientation,
                rates: s.rates,
                ground_contact: false,
            };
            let k = actions.partition_point(|a| a.time <= s.time);
            let (command, actual) = match k.checked_sub(1).map(|k| &actions[k]) {
                Some(a) if a.action.len() == action_map.dim() => {
                    let (t, c, m) = tracking(&action_map.setpoint(&a.action, &est), &est, &s.motors);
                    kind = t;
                    (c, m)
                }
                _ => ([f64::NAN; 3], [f64::NAN; 3]),
            };
            Sample {
                time: s.time,
                agl: s.position.z - map.surface_height(s.position.x, s.position.y),
                goal_distance: (s.position - s.goal).length(),
                command,
                actual,
                sideslip: f64::NAN,
                wheels: Vec::new(),
            }
        })
        .collect();
    (kind, samples)
}

/// Ground vehicle curves over the playback's episode, with the setpoints of the recorded
/// actions (none for a recorded manual drive: the kinematic yaw rate is the reference).
fn ground_episode_samples(
    replay: &Replay,
    world: &WorldInstance,
    agent: usize,
    map: &StaticWorld,
    action_map: &autonomousim_control::ground::GroundActionMap,
) -> Vec<Sample> {
    let ep = replay.current();
    let Some(states) = ep.states.get(agent) else { return Vec::new() };
    let actions = ep.actions.get(agent).map_or(&[][..], |a| &a[..]);
    let wheelbase = world.agent(agent).controller.as_ground().and_then(|c| c.wheelbase());
    states
        .iter()
        .map(|s| {
            let vb = s.orientation.inverse() * s.velocity;
            let k = actions.partition_point(|a| a.time <= s.time);
            let setpoint = k
                .checked_sub(1)
                .map(|k| &actions[k])
                .filter(|a| a.action.len() == action_map.dim())
                .map(|a| action_map.setpoint(&a.action));
            let (command, actual) = ground_tracking(setpoint.as_ref(), vb.x, s.rates.z, s.steering, wheelbase);
            Sample {
                time: s.time,
                agl: s.position.z - map.surface_height(s.position.x, s.position.y),
                goal_distance: (s.position - s.goal).length(),
                command,
                actual,
                sideslip: sideslip(vb),
                wheels: WheelSample::all(s.wheels.iter().map(|w| [w.kappa, w.tan_alpha, w.fx, w.fy, w.load])),
            }
        })
        .collect()
}

pub fn record(sim: Res<Sim>, mut history: ResMut<History>) {
    let map = sim.world.map();
    let i = sim.pilot;
    if let Some(r) = &sim.replay {
        if !history.matches(i, map, Some(r.episode)) {
            history.restart(i, map, Some(r.episode));
            let (tracking, samples) = episode_samples(r, &sim.world, i);
            history.tracking = tracking;
            history.samples = samples.into();
        }
        return;
    }
    if sim.paused {
        return;
    }
    let (tracking, sample) = live_sample(&sim.world, i);
    let time = sample.time;
    let last = history.samples.back().map(|s| s.time);
    // A new episode, map, agent or kind of setpoint starts the plots over.
    if !history.matches(i, map, None) || tracking != history.tracking || last.is_some_and(|t| sample.time < t) {
        history.restart(i, map, None);
        history.tracking = tracking;
    }
    if last != Some(time) {
        history.samples.push_back(sample);
    }
    while history.samples.front().is_some_and(|s| s.time < time - WINDOW) {
        history.samples.pop_front();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec2;

    fn level(attitude: DQuat) -> StateEstimate {
        StateEstimate {
            position: DVec3::ZERO,
            velocity: DVec3::new(0.0, 2.0, 0.0),
            attitude,
            rates: DVec3::new(0.1, 0.0, 0.0),
            ground_contact: false,
        }
    }

    #[test]
    fn measured_quantities_match_their_setpoints() {
        use autonomousim_control::multirotor::YawCommand;
        // Heading north, nose 10° down: tilt about the heading frame's y axis.
        let q = DQuat::from_rotation_z(std::f64::consts::FRAC_PI_2) * DQuat::from_rotation_y(10f64.to_radians());
        let sp = Setpoint::Attitude { tilt: DVec2::new(0.0, 0.2), yaw: YawCommand::Rate(0.0), thrust: 10.0 };
        let (kind, cmd, act) = tracking(&sp, &level(q), &[]);
        assert_eq!(kind, Tracking::Tilt);
        assert!((cmd[1] - 0.2f64.to_degrees()).abs() < 1e-9);
        assert!(act[0].abs() < 1e-9 && (act[1] - 10.0).abs() < 1e-9, "{act:?}");
        // Flying north is forward in the heading frame.
        let sp = Setpoint::Velocity { velocity: DVec3::X, frame: Frame::Heading, yaw: YawCommand::Rate(0.0) };
        let (_, _, act) = tracking(&sp, &level(q), &[]);
        assert!((act[0] - 2.0).abs() < 1e-9 && act[1].abs() < 1e-9, "{act:?}");
        let sp = Setpoint::Ctbr { thrust: 10.0, rates: DVec3::ZERO };
        let (kind, _, act) = tracking(&sp, &level(q), &[]);
        assert_eq!(kind, Tracking::Rates);
        assert!((act[0] - 0.1f64.to_degrees()).abs() < 1e-9);
    }

    #[test]
    fn ground_references_and_tyre_samples() {
        // Pedal driving: the reference is the kinematic yaw rate v·tan δ / L.
        let sp = GroundSetpoint::Pedal { drive: 0.5, steering: 0.2, handbrake: false };
        let (cmd, act) = ground_tracking(Some(&sp), 10.0, 0.5, 0.1, Some(2.5));
        assert!((cmd[0] - (10.0 * 0.1f64.tan() / 2.5).to_degrees()).abs() < 1e-9);
        assert!(cmd[1].is_nan() && (act[0] - 0.5f64.to_degrees()).abs() < 1e-9 && act[1] == 10.0);
        // Speed modes ask for their speed and yaw rate (curvature times the actual speed).
        let sp = GroundSetpoint::SpeedCurvature { speed: 8.0, curvature: 0.1 };
        let (cmd, _) = ground_tracking(Some(&sp), 6.0, 0.5, 0.1, Some(2.5));
        assert!((cmd[0] - 0.6f64.to_degrees()).abs() < 1e-9 && cmd[1] == 8.0);
        let sp = GroundSetpoint::SpeedYawRate { speed: 1.0, yaw_rate: -0.3 };
        assert_eq!(ground_tracking(Some(&sp), 1.0, 0.0, 0.0, None).0[..2], [-0.3f64.to_degrees(), 1.0]);
        assert!(ground_tracking(None, 1.0, 0.0, 0.1, None).0[0].is_nan());
        // Sideslip: undefined when slow, positive sliding left, the same driving backwards.
        assert!(sideslip(DVec3::new(0.5, 0.2, 0.0)).is_nan());
        assert!((sideslip(DVec3::new(5.0, 5.0, 0.0)) - 45.0).abs() < 1e-9);
        assert!((sideslip(DVec3::new(-5.0, 5.0, 0.0)) - 45.0).abs() < 1e-9);
        // Tyres: force per load and slip angle in degrees; a nearly unloaded wheel is left out.
        let w = WheelSample::all([[0.1, 1.0, 400.0, -800.0, 4000.0], [0.0, 0.0, 0.0, 0.0, 300.0]].into_iter());
        assert_eq!((w[0].kappa, w[0].fx, w[0].fy), (0.1, 0.1, -0.2));
        assert!((w[0].alpha - 45.0).abs() < 1e-9);
        assert!(w[1].kappa.is_nan() && w[1].fy.is_nan());
    }
}
