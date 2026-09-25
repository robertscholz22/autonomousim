//! Playback of an MCAP recording: the recorded states drive the agents of a world built from
//! the recorded scenario (its maps checked against their hashes), so the rest of the viewer
//! draws a replay exactly like a live simulation.

use autonomousim_core::math::Pose;
use autonomousim_sim::record::{RecordedEpisode, RecordedState, RecordedWheel, Recording};
use autonomousim_sim::{Events, WorldInstance};
use autonomousim_vehicles::ground::tire::TireForces;
use autonomousim_vehicles::ground::{PowertrainStatus, WheelState, WheeledInit};
use glam::DVec3;

pub struct Replay {
    pub recording: Recording,
    /// Episode being played.
    pub episode: usize,
    /// Playback time within the episode (s).
    pub time: f64,
    /// Start over after the last episode.
    pub looping: bool,
}

/// A recorded state interpolated between two samples.
#[derive(Clone, Debug)]
pub struct Sample {
    pub pose: Pose,
    pub velocity: DVec3,
    pub rates: DVec3,
    pub motors: Vec<f64>,
    /// Ground vehicles: bicycle steering angle (rad) and wheels.
    pub steering: f64,
    pub wheels: Vec<RecordedWheel>,
    pub goal: DVec3,
    /// The sample at or before the playback time (events, flags).
    pub last: RecordedState,
}

impl Replay {
    pub fn new(recording: Recording, episode: usize) -> Self {
        let episode = episode.min(recording.episodes.len().saturating_sub(1));
        Self { recording, episode, time: 0.0, looping: true }
    }

    pub fn current(&self) -> &RecordedEpisode {
        &self.recording.episodes[self.episode]
    }

    pub fn duration(&self) -> f64 {
        self.current().duration()
    }

    /// Advance the playback by `dt` of recorded time, moving on to the next episode at the end
    /// of this one (and to the first after the last, when looping).
    pub fn advance(&mut self, dt: f64) {
        self.time += dt;
        if self.time > self.duration() {
            if self.episode + 1 < self.recording.episodes.len() {
                self.set_episode(self.episode + 1);
            } else if self.looping {
                self.set_episode(0);
            } else {
                self.time = self.duration();
            }
        }
    }

    pub fn seek(&mut self, time: f64) {
        self.time = time.clamp(0.0, self.duration());
    }

    pub fn set_episode(&mut self, episode: usize) {
        self.episode = episode.min(self.recording.episodes.len() - 1);
        self.time = 0.0;
    }

    /// Move by `n` state samples (at the recording's state rate), e.g. to step while paused.
    pub fn step_samples(&mut self, n: i32) {
        let dt = 1.0 / f64::from(self.recording.state_hz.max(1));
        self.seek((self.time / dt).round() * dt + f64::from(n) * dt);
    }

    /// Agent `agent` at the playback time; None if it has no states in this episode.
    pub fn sample(&self, agent: usize) -> Option<Sample> {
        let states = self.current().states.get(agent)?;
        let first = states.first()?;
        let k = states.partition_point(|s| s.time <= self.time);
        let (a, b) = match k {
            0 => (first, first),
            k if k == states.len() => (&states[k - 1], &states[k - 1]),
            k => (&states[k - 1], &states[k]),
        };
        let alpha = if b.time > a.time { ((self.time - a.time) / (b.time - a.time)).clamp(0.0, 1.0) } else { 0.0 };
        Some(Sample {
            pose: Pose { pos: a.position.lerp(b.position, alpha), rot: a.orientation.slerp(b.orientation, alpha) },
            velocity: a.velocity.lerp(b.velocity, alpha),
            rates: a.rates.lerp(b.rates, alpha),
            motors: a.motors.iter().zip(&b.motors).map(|(x, y)| x + (y - x) * alpha).collect(),
            steering: a.steering + (b.steering - a.steering) * alpha,
            wheels: a.wheels.iter().zip(&b.wheels).map(|(x, y)| lerp_wheel(x, y, alpha)).collect(),
            goal: a.goal,
            last: a.clone(),
        })
    }

    /// Events of agent `agent` from the start of the episode to the playback time.
    pub fn latched(&self, agent: usize) -> Events {
        let states = self.current().states.get(agent).map_or(&[][..], |s| &s[..]);
        Events(states.iter().take_while(|s| s.time <= self.time).fold(0, |acc, s| acc | s.events))
    }

    /// Recorded positions of `agent` from `window` seconds before the playback time up to it.
    pub fn trail(&self, agent: usize, window: f64) -> Vec<DVec3> {
        let Some(states) = self.current().states.get(agent) else { return Vec::new() };
        let mut points: Vec<DVec3> = states
            .iter()
            .skip_while(|s| s.time < self.time - window)
            .take_while(|s| s.time <= self.time)
            .map(|s| s.position)
            .collect();
        if let Some(s) = self.sample(agent) {
            points.push(s.pose.pos);
        }
        points
    }

    /// Put the agents of `world` where the recording has them: the episode's map, the
    /// interpolated state, rotor speeds, events and goals.
    pub fn apply(&self, world: &mut WorldInstance) {
        let ep = self.current();
        if world.map_index() != ep.map {
            world.set_map(ep.map);
        }
        for i in 0..world.agents().len() {
            let Some(s) = self.sample(i) else { continue };
            let agent = world.agent_mut(i);
            if let Some(v) = agent.vehicle.as_wheeled_mut() {
                let init = WheeledInit { pose: s.pose, lin_vel_world: s.velocity, ang_vel_body: s.rates };
                let wheels: Vec<WheelState> = s.wheels.iter().map(wheel_state).collect();
                let powertrain =
                    PowertrainStatus { gear: s.last.gear, engine_speed: s.last.engine_speed, engine_torque: f64::NAN };
                v.show(&init, s.steering, &wheels, powertrain);
            } else {
                agent.vehicle.place(s.pose, s.velocity, s.rates);
            }
            if let Some(v) = agent.vehicle.as_multirotor_mut()
                && s.motors.len() == v.motor_speeds().len()
            {
                v.set_motor_speeds(&s.motors);
            }
            agent.events = Events(s.last.events);
            agent.disabled = s.last.disabled;
            if let Some(goals) = ep.goals.get(i).filter(|g| !g.is_empty()) {
                if agent.goals != *goals {
                    agent.goals = goals.clone();
                }
                agent.goal_index = goals.iter().position(|g| g.position == s.goal).unwrap_or(agent.goal_index);
            }
        }
    }
}

fn lerp_wheel(a: &RecordedWheel, b: &RecordedWheel, alpha: f64) -> RecordedWheel {
    let l = |x: f64, y: f64| x + (y - x) * alpha;
    RecordedWheel {
        spin: l(a.spin, b.spin),
        spin_angle: l(a.spin_angle, b.spin_angle),
        steer: l(a.steer, b.steer),
        travel: l(a.travel, b.travel),
        drive_torque: l(a.drive_torque, b.drive_torque),
        brake_torque: l(a.brake_torque, b.brake_torque),
        load: l(a.load, b.load),
        kappa: l(a.kappa, b.kappa),
        tan_alpha: l(a.tan_alpha, b.tan_alpha),
        fx: l(a.fx, b.fx),
        fy: l(a.fy, b.fy),
    }
}

/// A recorded wheel as the vehicle's wheel output.
pub fn wheel_state(w: &RecordedWheel) -> WheelState {
    WheelState {
        travel: w.travel,
        steer: w.steer,
        spin_angle: w.spin_angle,
        spin: w.spin,
        drive_torque: w.drive_torque,
        brake_torque: w.brake_torque,
        tire: TireForces {
            fz: w.load,
            fx: w.fx,
            fy: w.fy,
            kappa: w.kappa,
            tan_alpha: w.tan_alpha,
            ..Default::default()
        },
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use autonomousim_control::ground::GroundActionMode;
    use autonomousim_control::multirotor::ActionMode;
    use autonomousim_core::rng::Seed;
    use autonomousim_sim::record::{Recorder, RecorderConfig};
    use autonomousim_sim::scenario::{MapSource, Testworld};
    use autonomousim_sim::{GroupSpec, Scenario};
    use std::sync::Arc;

    /// Two episodes of two drones falling with their rotors off, recorded to MCAP and read back.
    fn recording() -> (Recording, WorldInstance) {
        let sc = Scenario {
            map: MapSource::Testworld(Testworld::Flat { size: 100.0 }),
            groups: vec![GroupSpec { count: 2, action_mode: Some(ActionMode::Ctbr.into()), ..Default::default() }],
            ..Default::default()
        };
        let sc = Arc::new(sc.compile().unwrap());
        let path = std::env::temp_dir().join(format!("autonomousim-replay-{}.mcap", std::process::id()));
        let mut rec = Recorder::create(&path, RecorderConfig::default()).unwrap();
        let mut w = WorldInstance::new(sc, Seed::from_u64(3));
        for steps in [50, 25] {
            rec.on_reset(&w);
            for _ in 0..steps {
                w.set_actions(0, &[0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0]);
                rec.on_actions(&w);
                w.step_with(&mut |w| rec.on_tick(w));
            }
            w.reset(None);
        }
        rec.finish().unwrap();
        let recording = Recording::read(&path).unwrap();
        std::fs::remove_file(&path).ok();
        let world = WorldInstance::new(Arc::new(recording.compile().unwrap()), Seed::from_u64(0));
        (recording, world)
    }

    #[test]
    fn playback_interpolates_and_moves_through_episodes() {
        let (recording, mut world) = recording();
        let mut r = Replay::new(recording, 0);
        assert_eq!((r.recording.episodes.len(), r.duration()), (2, 1.0));
        let states = r.current().states[0].clone();
        // Exactly on a sample, then halfway between two.
        r.seek(0.2);
        assert_eq!(r.sample(0).unwrap().pose.pos, states[10].position);
        r.seek(0.21);
        let mid = r.sample(0).unwrap().pose.pos;
        assert!((mid - (states[10].position + states[11].position) / 2.0).length() < 1e-12);
        // The agents are placed where the recording has them.
        r.apply(&mut world);
        assert_eq!(world.agent(0).vehicle.position(), mid);
        assert_eq!(world.agent(1).goals, r.current().goals[1]);
        // The drones fall with their rotors off and crash before the end.
        r.seek(1.0);
        assert!(r.latched(0).intersects(Events::TERMINAL));
        assert!(!Replay::new(r.recording.clone(), 0).latched(0).intersects(Events::TERMINAL));
        assert_eq!(r.trail(0, 0.5).len(), 26 + 1);
        // Stepping by samples, then on to the next episode and back to the first.
        r.seek(0.5);
        r.step_samples(-2);
        assert!((r.time - 0.46).abs() < 1e-9);
        r.advance(0.6);
        assert_eq!((r.episode, r.time), (1, 0.0));
        r.advance(0.6);
        assert_eq!(r.episode, 0);
        r.looping = false;
        r.set_episode(1);
        r.advance(2.0);
        assert_eq!((r.episode, r.time), (1, 0.5));
    }

    #[test]
    fn replayed_ground_vehicles_show_their_wheels() {
        let sc = Scenario {
            map: MapSource::Testworld(Testworld::Flat { size: 400.0 }),
            groups: vec![GroupSpec {
                vehicle: autonomousim_sim::scenario::VehicleRef::Name("offroad_4x4".into()),
                action_mode: Some(GroundActionMode::Raw.into()),
                ..Default::default()
            }],
            ..Default::default()
        };
        let sc = Arc::new(sc.compile().unwrap());
        let path = std::env::temp_dir().join(format!("autonomousim-replay-ground-{}.mcap", std::process::id()));
        let mut rec = Recorder::create(&path, RecorderConfig::default()).unwrap();
        let mut w = WorldInstance::new(sc, Seed::from_u64(3));
        rec.on_reset(&w);
        for k in 0..100 {
            w.set_actions(0, &[0.6, if k < 50 { 0.0 } else { 0.5 }]);
            rec.on_actions(&w);
            w.step_with(&mut |w| rec.on_tick(w));
        }
        rec.finish().unwrap();
        let recording = Recording::read(&path).unwrap();
        std::fs::remove_file(&path).ok();
        let mut world = WorldInstance::new(Arc::new(recording.compile().unwrap()), Seed::from_u64(0));
        let mut r = Replay::new(recording, 0);
        r.seek(r.duration());
        r.apply(&mut world);
        let (live, shown) = (w.agent(0).vehicle.as_wheeled().unwrap(), world.agent(0).vehicle.as_wheeled().unwrap());
        assert!(live.lin_vel_body().x > 3.0);
        // The wheels turn, steer and travel as they did; the drawn wheel poses agree (the live
        // ones lag a tick).
        for (i, (a, b)) in live.wheels().zip(shown.wheels()).enumerate() {
            assert!((a.spin_angle - b.spin_angle).abs() < 1e-5, "wheel {i}");
            assert!((a.steer - b.steer).abs() < 1e-5 && (a.travel - b.travel).abs() < 1e-5, "wheel {i}");
            assert!((a.tire.fy - b.tire.fy).abs() < 1.0 && (a.tire.kappa - b.tire.kappa).abs() < 1e-5, "wheel {i}");
            let (pa, pb) = (live.wheel_pose(i), shown.wheel_pose(i));
            assert!((pa.pos - pb.pos).length() < 0.02, "wheel {i}: {} {}", pa.pos, pb.pos);
            assert!(pa.rot.angle_between(pb.rot) < 0.05, "wheel {i}");
        }
        assert!(shown.wheels().next().unwrap().steer > 0.05);
        assert_eq!(shown.powertrain().gear, live.powertrain().gear);
    }

    /// Plays every episode of a recording as the viewer's `replay` mode does, for recordings
    /// the tests cannot make (`examples/eval_record.py`): `AUTONOMOUSIM_RECORDING=$PWD/recordings/<run>.mcap
    /// cargo test -p autonomousim-viewer --release -- --ignored --nocapture recorded_file`.
    #[test]
    #[ignore = "needs a recording (AUTONOMOUSIM_RECORDING)"]
    fn recorded_file_plays() {
        let path = std::env::var("AUTONOMOUSIM_RECORDING").expect("AUTONOMOUSIM_RECORDING");
        let recording = Recording::read(path).unwrap();
        // Rebuilds the maps and checks their hashes.
        let mut world = WorldInstance::new(Arc::new(recording.compile().unwrap()), Seed::from_u64(0));
        let episodes = recording.episodes.len();
        let mut r = Replay::new(recording, 0);
        for e in 0..episodes {
            r.set_episode(e);
            let ep = r.current();
            let last: Vec<_> = ep.states.iter().map(|s| s.last().unwrap().clone()).collect();
            let mut t = 0.0;
            while t < r.duration() {
                r.seek(t);
                r.apply(&mut world);
                t += 0.05;
            }
            r.seek(r.duration());
            r.apply(&mut world);
            for (i, s) in last.iter().enumerate() {
                assert!((world.agent(i).vehicle.position() - s.position).length() < 1e-9, "agent {i}");
                assert_eq!(world.agent(i).disabled, s.disabled, "agent {i}");
            }
            let stopped = world.agents().iter().filter(|a| a.disabled).count();
            println!(
                "episode {e}: {:.1} s, {} agents, {stopped} stopped at the end",
                r.duration(),
                world.agents().len()
            );
        }
    }
}
