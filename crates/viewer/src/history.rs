//! Flight data for the HUD plots: height, distance to the goal, and what the cascade was asked
//! for against what the vehicle did, at the level where the setpoint enters the cascade.
//!
//! Live, the followed agent is sampled every frame over the last [`WINDOW`] seconds; a replay
//! is turned into curves over the whole episode (setpoints rebuilt from the recorded actions).

use crate::replay::Replay;
use crate::sim::Sim;
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
}

impl Tracking {
    pub fn label(self) -> &'static str {
        match self {
            Tracking::Motors => "rotor speed (rad/s)",
            Tracking::Rates => "body rates (°/s)",
            Tracking::Tilt => "tilt (°)",
            Tracking::Velocity => "velocity (m/s)",
            Tracking::Position => "position (m)",
        }
    }

    /// Names of the three components.
    pub fn axes(self) -> [&'static str; 3] {
        match self {
            Tracking::Motors => ["1", "2", "3"],
            Tracking::Rates => ["roll", "pitch", "yaw"],
            Tracking::Tilt => ["x", "y", "-"],
            Tracking::Velocity | Tracking::Position => ["x", "y", "z"],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sample {
    pub time: f64,
    /// Height above the ground or water (m).
    pub agl: f64,
    /// Distance to the current goal (m).
    pub goal_distance: f64,
    /// Setpoint and the matching measured quantity (NaN where there is none).
    pub command: [f64; 3],
    pub actual: [f64; 3],
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
    let est = StateEstimate::of(v);
    let (tracking, command, actual) = tracking(agent.setpoint(), &est, v.motor_speeds());
    let sample = Sample {
        time: world.time(),
        agl: agent.agl_now(world.map()),
        goal_distance: (v.position() - agent.goal().position).length(),
        command,
        actual,
    };
    (tracking, sample)
}

/// Curves of `agent` over the playback's episode: every recorded state, with the setpoint
/// of the action in force at that time.
fn episode_samples(replay: &Replay, world: &WorldInstance, agent: usize) -> (Tracking, Vec<Sample>) {
    let ep = replay.current();
    let map = &world.scenario().maps[ep.map];
    let group = &world.scenario().groups[world.agent(agent).group];
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
                Some(a) if a.action.len() == group.action_map.dim() => {
                    let (t, c, m) = tracking(&group.action_map.setpoint(&a.action, &est), &est, &s.motors);
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
            }
        })
        .collect();
    (kind, samples)
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
    let last = history.samples.back().map(|s| s.time);
    // A new episode, map, agent or kind of setpoint starts the plots over.
    if !history.matches(i, map, None) || tracking != history.tracking || last.is_some_and(|t| sample.time < t) {
        history.restart(i, map, None);
        history.tracking = tracking;
    }
    if last != Some(sample.time) {
        history.samples.push_back(sample);
    }
    while history.samples.front().is_some_and(|s| s.time < sample.time - WINDOW) {
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
}
