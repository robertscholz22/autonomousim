//! What the viewer shows: either a live [`WorldInstance`] stepped at its physics rate from the
//! frame clock (fixed-step accumulator) with keyboard flight or driving of one agent, or the
//! playback of a recording ([`Replay`]) that places the agents of the same kind of world.
//! Live, a trained policy ([`Autopilot`]) can fly the agents instead, and the session can be
//! recorded.

use crate::autopilot::{self, Autopilot};
use crate::camera::{CameraMode, CameraRig};
use crate::replay::Replay;
use autonomousim_control::Command;
use autonomousim_control::ground::GroundSetpoint;
use autonomousim_control::multirotor::{Frame, Setpoint, YawCommand};
use autonomousim_core::math::Pose;
use autonomousim_sim::record::Recorder;
use autonomousim_sim::{Events, WorldInstance};
use autonomousim_vehicles::Family;
use bevy::prelude::*;
use bevy_egui::EguiContexts;
use glam::{DVec2, DVec3};
use std::sync::Mutex;

/// Simulated time advanced per frame at most (s); slower frames run the simulation slower.
const MAX_FRAME_STEP: f64 = 0.1;

/// Keyboard steering of ground vehicles: the rate at which the steering follows the keys
/// (full lock per second), and the speed (m/s) at which the reachable lock is halved (it falls
/// as `1/(1 + (v/v₀)²)`, so that a tap of the key does not spin the car at speed).
const STEER_RATE: f64 = 2.5;
const STEER_FADE_SPEED: f64 = 12.0;

/// How the keys fly the pilot.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PilotMode {
    /// Velocity in the heading frame and yaw rate (the cascade holds position and attitude).
    #[default]
    Velocity,
    /// Tilt and yaw rate; collective thrust around hover, raised to make up for the tilt.
    Attitude,
    /// Body rates (acro); thrust as in attitude mode.
    Rates,
}

impl PilotMode {
    pub fn next(self) -> Self {
        match self {
            PilotMode::Velocity => PilotMode::Attitude,
            PilotMode::Attitude => PilotMode::Rates,
            PilotMode::Rates => PilotMode::Velocity,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            PilotMode::Velocity => "velocity",
            PilotMode::Attitude => "attitude",
            PilotMode::Rates => "rates",
        }
    }
}

#[derive(Resource)]
pub struct Sim {
    pub world: WorldInstance,
    /// Agent flown from the keyboard and followed by the camera and the HUD.
    pub pilot: usize,
    pub paused: bool,
    /// Simulated (or recorded) seconds per real second.
    pub time_scale: f64,
    pub pilot_mode: PilotMode,
    /// Pilot input: forward, left, up and yaw, each in [−1, 1]. Ground vehicles use forward
    /// (pedal) and left (steering).
    pub stick: [f64; 4],
    /// Ground vehicles: parking brake held, and the steering command after the rate limit and
    /// the fade with speed.
    pub handbrake: bool,
    pub steer: f64,
    /// Drives the followed ground vehicle instead of the keys (`--demo`).
    pub drive_command: Option<GroundSetpoint>,
    /// Horizontal and vertical speed at full stick (m/s), yaw rate (rad/s).
    pub max_speed: f64,
    pub max_climb: f64,
    pub max_yaw_rate: f64,
    /// Tilt at full stick in attitude mode (rad) and roll/pitch rate in rates mode (rad/s).
    pub max_tilt: f64,
    pub max_rate: f64,
    /// Simulated time not yet stepped (s).
    accumulator: f64,
    /// Pose of every agent one tick before the current state, for interpolation.
    previous: Vec<Pose>,
    /// Events of every agent since the reset.
    pub latched: Vec<Events>,
    /// Real-time factor over the last frames.
    pub real_time_factor: f64,
    /// Episodes started.
    pub episodes: u64,
    /// Playing back a recording instead of simulating.
    pub replay: Option<Replay>,
    /// A trained policy flying the agents of its group (live only).
    pub autopilot: Option<Autopilot>,
    /// Recording of the live session (`--record`; in a mutex since the recorder is not `Sync`).
    recorder: Option<Mutex<Recorder>>,
}

impl Sim {
    pub fn new(world: WorldInstance) -> Self {
        let n = world.agents().len();
        let mut s = Self {
            world,
            pilot: 0,
            paused: false,
            time_scale: 1.0,
            pilot_mode: PilotMode::Velocity,
            stick: [0.0; 4],
            handbrake: false,
            steer: 0.0,
            drive_command: None,
            max_speed: 8.0,
            max_climb: 3.0,
            max_yaw_rate: 1.5,
            max_tilt: 25f64.to_radians(),
            max_rate: 3.0,
            accumulator: 0.0,
            previous: Vec::new(),
            latched: vec![Events::NONE; n],
            real_time_factor: 1.0,
            episodes: 1,
            replay: None,
            autopilot: None,
            recorder: None,
        };
        s.snapshot_poses();
        s
    }

    /// Play `replay` back in `world` (built from the recorded scenario).
    pub fn replay(world: WorldInstance, replay: Replay) -> Self {
        let mut s = Self::new(world);
        replay.apply(&mut s.world);
        s.replay = Some(replay);
        s
    }

    /// Record the live session from now on: the current state starts an episode, and every
    /// reset starts the next.
    pub fn record(&mut self, mut recorder: Recorder) {
        recorder.on_reset(&self.world);
        self.recorder = Some(Mutex::new(recorder));
    }

    pub fn is_recording(&self) -> bool {
        self.recorder.is_some()
    }

    /// Write the rest of the recording and close it.
    pub fn finish_recording(&mut self) -> anyhow::Result<()> {
        match self.recorder.take() {
            Some(r) => Ok(r.into_inner().map_err(|_| anyhow::anyhow!("recorder poisoned"))?.finish()?),
            None => Ok(()),
        }
    }

    /// Go on in `world` (e.g. on a regenerated map) with a new episode. A recording ends here,
    /// since it holds one scenario.
    pub fn set_world(&mut self, world: WorldInstance) {
        if let Err(e) = self.finish_recording() {
            warn!("finishing the recording: {e:#}");
        }
        self.world = world;
        self.pilot = self.pilot.min(self.world.agents().len() - 1);
        self.latched = vec![Events::NONE; self.world.agents().len()];
        self.accumulator = 0.0;
        self.episodes += 1;
        if let Some(a) = &mut self.autopilot {
            a.ended = None;
        }
        self.snapshot_poses();
    }

    fn snapshot_poses(&mut self) {
        self.previous.clear();
        self.previous.extend(self.world.agents().iter().map(|a| a.vehicle.pose()));
    }

    /// A new episode, or back to the start of the recorded one.
    pub fn reset(&mut self) {
        if let Some(r) = &mut self.replay {
            r.seek(0.0);
            return;
        }
        self.world.reset(None);
        if let Some(Ok(r)) = self.recorder.as_mut().map(Mutex::get_mut) {
            r.on_reset(&self.world);
        }
        self.accumulator = 0.0;
        self.latched.iter_mut().for_each(|e| *e = Events::NONE);
        self.episodes += 1;
        if let Some(a) = &mut self.autopilot {
            a.ended = None;
        }
        self.snapshot_poses();
    }

    /// Time since the episode started (s).
    pub fn time(&self) -> f64 {
        self.replay.as_ref().map_or_else(|| self.world.time(), |r| r.time)
    }

    /// Episode number (from 1) and the number of episodes if known.
    pub fn episode(&self) -> (u64, Option<usize>) {
        match &self.replay {
            Some(r) => (r.episode as u64 + 1, Some(r.recording.episodes.len())),
            None => (self.episodes, None),
        }
    }

    /// The agent flown from the keyboard: the followed one, unless the autopilot flies it.
    pub fn manual_agent(&self) -> Option<usize> {
        match &self.autopilot {
            Some(a) if a.flies_pilot => None,
            _ => Some(self.pilot),
        }
    }

    /// The pilot's setpoint for the current stick input and pilot mode.
    pub fn setpoint(&self) -> Setpoint {
        let [forward, left, up, yaw] = self.stick;
        let yaw_rate = yaw * self.max_yaw_rate;
        if self.pilot_mode == PilotMode::Velocity {
            let velocity = DVec3::new(forward * self.max_speed, left * self.max_speed, up * self.max_climb);
            return Setpoint::Velocity { velocity, frame: Frame::Heading, yaw: YawCommand::Rate(yaw_rate) };
        }
        // Collective thrust around hover, raised by 1/cos(tilt) (up to 2×) to hold height.
        let v = &self.world.agent(self.pilot).vehicle;
        let up_z = (v.orientation() * DVec3::Z).z;
        let hover = v.mass() * self.world.env().gravity.length();
        let thrust = hover * (1.0 + 0.5 * up) / up_z.max(0.5);
        if self.pilot_mode == PilotMode::Attitude {
            // Tilt as a rotation vector in the heading frame: nose down (about +y) flies
            // forward, left side up (about −x) flies left.
            let tilt = DVec2::new(-left, forward) * self.max_tilt;
            Setpoint::Attitude { tilt, yaw: YawCommand::Rate(yaw_rate), thrust }
        } else {
            let rates = DVec3::new(-left * self.max_rate, forward * self.max_rate, yaw_rate);
            Setpoint::Ctbr { thrust, rates }
        }
    }

    /// The pilot's command for the followed vehicle: [`setpoint`](Self::setpoint) for a
    /// multirotor. For a ground vehicle, forward/back is the pedal (brake, then reverse) and
    /// the steering follows left/right; side drives turn by driving their sides apart.
    pub fn pilot_command(&self) -> Command {
        let agent = self.world.agent(self.pilot);
        match agent.vehicle.family() {
            Family::Multirotor => self.setpoint().into(),
            Family::Wheeled => {
                if let Some(sp) = self.drive_command {
                    return sp.into();
                }
                let forward = self.stick[0];
                if agent.controller.as_ground().is_some_and(|c| c.is_side_drive()) {
                    let (left, right) =
                        ((forward - self.steer).clamp(-1.0, 1.0), (forward + self.steer).clamp(-1.0, 1.0));
                    GroundSetpoint::Sides { left, right }.into()
                } else {
                    GroundSetpoint::Pedal { drive: forward, steering: self.steer, handbrake: self.handbrake }.into()
                }
            }
        }
    }

    /// Move the steering command toward the keys by `dt` of simulated time.
    fn update_steering(&mut self, dt: f64) {
        let agent = self.world.agent(self.pilot);
        let steered = agent.controller.as_ground().is_some_and(|c| c.has_steering());
        let fade =
            if steered { 1.0 / (1.0 + (agent.vehicle.lin_vel_body().x / STEER_FADE_SPEED).powi(2)) } else { 1.0 };
        let step = STEER_RATE * dt;
        self.steer += (self.stick[1] * fade - self.steer).clamp(-step, step);
    }

    /// Advance by `real_dt` seconds of wall-clock time.
    pub fn advance(&mut self, real_dt: f64) {
        if let Some(r) = &mut self.replay {
            if !self.paused {
                r.advance(real_dt * self.time_scale);
            }
            r.apply(&mut self.world);
            for (i, l) in self.latched.iter_mut().enumerate() {
                *l = r.latched(i);
            }
            self.real_time_factor = if self.paused { 0.0 } else { self.time_scale };
            return;
        }
        if self.paused {
            self.real_time_factor = 0.0;
            return;
        }
        let dt = self.world.clock().dt();
        let wanted = real_dt * self.time_scale;
        self.accumulator += wanted.min(MAX_FRAME_STEP);
        let manual = self.manual_agent();
        if let Some(pilot) = manual {
            self.update_steering(wanted.min(MAX_FRAME_STEP));
            let command = self.pilot_command();
            self.world.set_command(pilot, command);
        }
        let mut stepped = 0.0;
        while self.accumulator >= dt {
            if self.accumulator < 2.0 * dt {
                self.snapshot_poses();
            }
            if let Some(a) = &mut self.autopilot {
                a.before_tick(&mut self.world, manual);
            }
            for a in 0..self.world.agents().len() {
                self.world.agent_mut(a).events = Events::NONE;
            }
            self.world.tick();
            if let Some(Ok(r)) = self.recorder.as_mut().map(Mutex::get_mut) {
                r.on_tick(&self.world);
            }
            for (l, a) in self.latched.iter_mut().zip(self.world.agents()) {
                *l |= a.events;
            }
            self.accumulator -= dt;
            stepped += dt;
        }
        if real_dt > 0.0 {
            self.real_time_factor = 0.9 * self.real_time_factor + 0.1 * stepped / real_dt;
        }
        self.autoreset(manual);
        // A diverged state cannot be drawn; start over.
        if self.world.agents().iter().any(|a| !a.vehicle.position().is_finite()) {
            warn!("non-finite vehicle state, resetting");
            self.reset();
        }
    }

    /// While the autopilot flies every agent of its group: start the next episode shortly after
    /// all of them ended theirs (or the task's time ran out).
    fn autoreset(&mut self, manual: Option<usize>) {
        let Some(a) = &mut self.autopilot else { return };
        if manual.is_some() {
            a.ended = None;
            return;
        }
        let now = self.world.time();
        match a.ended {
            None if a.episode_over(&self.world, &self.latched) => a.ended = Some(now),
            Some(t) if now >= t + autopilot::RESET_DELAY => {
                a.count_episode(&self.world, &self.latched, manual);
                self.reset();
            }
            _ => {}
        }
    }

    /// Pose of agent `i` to draw: interpolated between the last two ticks, or between the
    /// recorded samples around the playback time.
    pub fn render_pose(&self, i: usize) -> Pose {
        if let Some(s) = self.replay.as_ref().and_then(|r| r.sample(i)) {
            return s.pose;
        }
        let now = self.world.agent(i).vehicle.pose();
        let Some(prev) = self.previous.get(i) else { return now };
        let alpha = (self.accumulator / self.world.clock().dt()).clamp(0.0, 1.0);
        Pose { pos: prev.pos.lerp(now.pos, alpha), rot: prev.rot.slerp(now.rot, alpha) }
    }
}

/// Keys common to both modes: P pause, `[`/`]` time scale, R reset (live) or restart
/// (replay), Tab next agent. Live only: W/S forward/back, A/D left/right, Space/Shift up/down,
/// Q/E yaw, M pilot mode, `-`/`=` speed, T take over from the autopilot. Ground vehicles:
/// W/S pedal (brake, then reverse), A/D steering, Space handbrake. The free camera takes the
/// flight keys.
///
/// A gamepad flies as a mode-2 transmitter: left stick climb and yaw, right stick forward and
/// sideways; Start pauses, Select changes the pilot mode, East (B) resets. It drives with the
/// right trigger (throttle), the left trigger (brake, then reverse), the left stick
/// (steering) and South (A, handbrake).
pub fn pilot_input(
    keys: Res<ButtonInput<KeyCode>>,
    gamepads: Query<&Gamepad>,
    camera: Query<&CameraRig>,
    mut egui: EguiContexts,
    mut sim: ResMut<Sim>,
) {
    if egui.ctx_mut().is_ok_and(|c| c.egui_wants_keyboard_input()) {
        sim.stick = [0.0; 4];
        sim.handbrake = false;
        return;
    }
    let ground = sim.world.agent(sim.pilot).vehicle.family() == Family::Wheeled;
    let axis = |pos: KeyCode, neg: KeyCode| f64::from(keys.pressed(pos) as u8) - f64::from(keys.pressed(neg) as u8);
    let mut stick = [axis(KeyCode::KeyW, KeyCode::KeyS), axis(KeyCode::KeyA, KeyCode::KeyD), 0.0, 0.0];
    let mut handbrake = false;
    if ground {
        handbrake = keys.pressed(KeyCode::Space);
    } else {
        stick[2] = axis(KeyCode::Space, KeyCode::ShiftLeft);
        stick[3] = axis(KeyCode::KeyQ, KeyCode::KeyE);
    }
    let (mut pause, mut mode, mut reset) = (false, false, false);
    for pad in &gamepads {
        let (l, r) = (pad.left_stick().as_dvec2(), pad.right_stick().as_dvec2());
        let pads = if ground {
            let trigger = |b: GamepadButton| f64::from(pad.get(b).unwrap_or(0.0));
            handbrake |= pad.pressed(GamepadButton::South);
            [trigger(GamepadButton::RightTrigger2) - trigger(GamepadButton::LeftTrigger2), -l.x, 0.0, 0.0]
        } else {
            [r.y, -r.x, l.y, -l.x]
        };
        for (s, p) in stick.iter_mut().zip(pads) {
            *s = (*s + p).clamp(-1.0, 1.0);
        }
        pause |= pad.just_pressed(GamepadButton::Start);
        mode |= pad.just_pressed(GamepadButton::Select);
        reset |= pad.just_pressed(GamepadButton::East);
    }
    let horizontal = DVec2::new(stick[0], stick[1]);
    if !ground && horizontal.length() > 1.0 {
        let h = horizontal.normalize();
        (stick[0], stick[1]) = (h.x, h.y);
    }
    let free = camera.single().is_ok_and(|c| c.mode == CameraMode::Free);
    let idle = free || sim.replay.is_some();
    sim.stick = if idle { [0.0; 4] } else { stick };
    sim.handbrake = handbrake && !idle;

    if keys.just_pressed(KeyCode::KeyR) || reset {
        sim.reset();
    }
    if keys.just_pressed(KeyCode::KeyP) || pause {
        sim.paused = !sim.paused;
    }
    if keys.just_pressed(KeyCode::BracketRight) {
        sim.time_scale = (sim.time_scale * 2.0).min(8.0);
    }
    if keys.just_pressed(KeyCode::BracketLeft) {
        sim.time_scale = (sim.time_scale / 2.0).max(0.125);
    }
    if keys.just_pressed(KeyCode::Tab) {
        sim.pilot = (sim.pilot + 1) % sim.world.agents().len();
    }
    if let Some(a) = &mut sim.autopilot
        && keys.just_pressed(KeyCode::KeyT)
    {
        a.flies_pilot = !a.flies_pilot;
    }
    if sim.replay.is_none() && !ground {
        if keys.just_pressed(KeyCode::KeyM) || mode {
            sim.pilot_mode = sim.pilot_mode.next();
        }
        if keys.just_pressed(KeyCode::Equal) {
            sim.max_speed = (sim.max_speed * 1.5).min(40.0);
        }
        if keys.just_pressed(KeyCode::Minus) {
            sim.max_speed = (sim.max_speed / 1.5).max(1.0);
        }
    }
}

/// Replay keys: ←/→ one second back/forward (with Shift: one sample), N/B (or PageDown/PageUp)
/// next/previous episode, Home start of the episode.
pub fn replay_input(keys: Res<ButtonInput<KeyCode>>, mut egui: EguiContexts, mut sim: ResMut<Sim>) {
    if egui.ctx_mut().is_ok_and(|c| c.egui_wants_keyboard_input()) {
        return;
    }
    let shift = keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight);
    let Some(r) = sim.replay.as_mut() else { return };
    for (key, dir) in [(KeyCode::ArrowLeft, -1), (KeyCode::ArrowRight, 1)] {
        if keys.just_pressed(key) {
            if shift {
                r.step_samples(dir);
            } else {
                r.seek(r.time + f64::from(dir));
            }
        }
    }
    let episodes = r.recording.episodes.len();
    if keys.any_just_pressed([KeyCode::KeyN, KeyCode::PageDown]) {
        r.set_episode((r.episode + 1) % episodes);
    }
    if keys.any_just_pressed([KeyCode::KeyB, KeyCode::PageUp]) {
        r.set_episode((r.episode + episodes - 1) % episodes);
    }
    if keys.just_pressed(KeyCode::Home) {
        r.seek(0.0);
    }
}

pub fn step(time: Res<Time>, mut sim: ResMut<Sim>) {
    sim.advance(f64::from(time.delta_secs()));
}

#[cfg(test)]
mod tests {
    use super::*;
    use autonomousim_control::ground::GroundActionMode;
    use autonomousim_control::multirotor::ActionMode;
    use autonomousim_core::math::quat::yaw;
    use autonomousim_core::rng::Seed;
    use autonomousim_sim::scenario::{MapSource, Testworld};
    use autonomousim_sim::{GroupSpec, Scenario};
    use std::sync::Arc;

    fn sim() -> Sim {
        let sc = Scenario {
            map: MapSource::Testworld(Testworld::Flat { size: 200.0 }),
            groups: vec![GroupSpec {
                vehicle: autonomousim_sim::scenario::VehicleRef::Name("iris_like".into()),
                action_mode: Some(ActionMode::Velocity.into()),
                ..Default::default()
            }],
            ..Default::default()
        };
        Sim::new(WorldInstance::new(Arc::new(sc.compile().unwrap()), Seed::from_u64(1)))
    }

    /// A ground vehicle driven by the pedal, as the viewer's driver group.
    fn ground_sim(vehicle: &str) -> Sim {
        let sc = Scenario {
            map: MapSource::Testworld(Testworld::Flat { size: 2000.0 }),
            groups: vec![GroupSpec {
                vehicle: autonomousim_sim::scenario::VehicleRef::Name(vehicle.into()),
                action_mode: Some(GroundActionMode::Raw.into()),
                disable_on_terminal: false,
                ..Default::default()
            }],
            ..Default::default()
        };
        Sim::new(WorldInstance::new(Arc::new(sc.compile().unwrap()), Seed::from_u64(1)))
    }

    /// Run `seconds` of 60 Hz frames with the given stick; returns the displacement along the
    /// initial heading and the heading change.
    fn drive(s: &mut Sim, stick: [f64; 4], seconds: f64) -> (f64, f64) {
        let start = s.world.agent(0).vehicle.position();
        let heading = yaw(s.world.agent(0).vehicle.orientation());
        s.stick = stick;
        for _ in 0..(seconds * 60.0).round() as usize {
            s.advance(1.0 / 60.0);
        }
        let v = &s.world.agent(0).vehicle;
        let d = (v.position() - start).truncate();
        let turn = (yaw(v.orientation()) - heading + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU)
            - std::f64::consts::PI;
        (d.dot(glam::DVec2::from_angle(heading)), turn)
    }

    #[test]
    fn keys_drive_and_steer_a_car() {
        let mut s = ground_sim("offroad_4x4");
        // Settle on the suspension, then the pedal drives forward.
        drive(&mut s, [0.0; 4], 0.5);
        assert!(!s.world.agent(0).controller.as_ground().unwrap().last_input().parking);
        let (forward, turn) = drive(&mut s, [1.0, 0.0, 0.0, 0.0], 3.0);
        assert!(forward > 5.0 && turn.abs() < 0.05, "forward {forward}, turn {turn}");
        // The steering follows the keys at the rate limit, less at speed.
        let speed = s.world.agent(0).vehicle.lin_vel_body().x;
        drive(&mut s, [0.5, 1.0, 0.0, 0.0], 0.1);
        assert!((s.steer - STEER_RATE * 0.1).abs() < 0.02, "steer {}", s.steer);
        let (_, turn) = drive(&mut s, [0.5, 1.0, 0.0, 0.0], 2.0);
        let fade = 1.0 / (1.0 + (speed / STEER_FADE_SPEED).powi(2));
        assert!(turn > 0.3 && s.steer < 1.0 && s.steer > 0.5 * fade, "turn {turn}, steer {}", s.steer);
        // Released, the steering centres. The handbrake locks the rear wheels and slows the car
        // (and swings its tail out); the brake pedal stops it, and the handbrake then holds it.
        drive(&mut s, [0.0; 4], 1.0);
        assert_eq!(s.steer, 0.0);
        let speed = s.world.agent(0).vehicle.lin_vel_body().x;
        s.handbrake = true;
        drive(&mut s, [0.0; 4], 1.0);
        assert!(s.world.agent(0).controller.as_ground().unwrap().last_input().parking);
        assert!(s.world.agent(0).vehicle.lin_vel_body().x < speed - 1.0);
        s.handbrake = false;
        for _ in 0..600 {
            if s.world.agent(0).vehicle.lin_vel_body().length() < 0.3 {
                break;
            }
            drive(&mut s, [-1.0, 0.0, 0.0, 0.0], 1.0 / 60.0);
        }
        s.handbrake = true;
        drive(&mut s, [0.0; 4], 1.0);
        let (moved, _) = drive(&mut s, [0.0; 4], 2.0);
        assert!(moved.abs() < 0.01, "moved {moved}");
        assert!(!s.latched[0].intersects(Events::TERMINAL), "{:?}", s.latched[0]);
    }

    #[test]
    fn side_drives_turn_on_the_spot() {
        let mut s = ground_sim("rover_skid");
        drive(&mut s, [0.0; 4], 0.5);
        let (forward, turn) = drive(&mut s, [0.0, 1.0, 0.0, 0.0], 0.5);
        assert!(turn > 0.3 && forward.abs() < 0.5, "forward {forward}, turn {turn}");
        let (forward, _) = drive(&mut s, [1.0, 0.0, 0.0, 0.0], 2.0);
        assert!(forward.abs() > 1.0, "forward {forward}");
    }

    #[test]
    fn drive_command_overrides_the_keys() {
        let mut s = ground_sim("offroad_4x4");
        s.stick = [1.0, 1.0, 0.0, 0.0];
        s.drive_command = Some(GroundSetpoint::Pedal { drive: -0.5, steering: 0.0, handbrake: true });
        assert!(matches!(
            s.pilot_command(),
            Command::Ground(GroundSetpoint::Pedal { drive, handbrake: true, .. }) if drive == -0.5
        ));
    }

    #[test]
    fn frames_advance_the_simulation_in_real_time() {
        let mut s = sim();
        let dt = s.world.clock().dt();
        for _ in 0..60 {
            s.advance(1.0 / 60.0);
        }
        let ticks = s.world.clock().tick;
        assert!((499..=500).contains(&ticks), "{ticks}");
        // Paused: nothing moves.
        s.paused = true;
        s.advance(0.5);
        assert_eq!(s.world.clock().tick, ticks);
        // Twice as fast, and a long frame is capped at MAX_FRAME_STEP of simulated time.
        s.paused = false;
        s.time_scale = 2.0;
        s.advance(0.04);
        assert!((s.world.clock().tick - ticks) as f64 >= 0.08 / dt - 1.0);
        let t = s.world.time();
        s.advance(5.0);
        assert!((s.world.time() - t - MAX_FRAME_STEP).abs() <= dt + 1e-9);
        // The drawn pose lies between the last two ticks.
        let p = s.render_pose(0);
        assert!((p.pos - s.world.agent(0).vehicle.position()).length() < 0.1);
    }

    #[test]
    fn keyboard_commands_fly_the_pilot() {
        let mut s = sim();
        let start = s.world.agent(0).vehicle.position();
        let heading = yaw(s.world.agent(0).vehicle.orientation());
        s.max_speed = 3.0;
        s.stick = [1.0, 0.0, 1.0 / 3.0, 0.0];
        for _ in 0..240 {
            s.advance(1.0 / 60.0);
        }
        let d = s.world.agent(0).vehicle.position() - start;
        let forward = d.truncate().dot(glam::DVec2::from_angle(heading));
        assert!(forward > 5.0 && d.z > 2.0, "moved {d}");
        assert!(!s.latched[0].intersects(Events::TERMINAL));
        s.reset();
        assert_eq!((s.world.clock().tick, s.episodes), (0, 2));
    }

    #[test]
    fn attitude_and_rate_modes_fly_forward() {
        for mode in [PilotMode::Attitude, PilotMode::Rates] {
            let mut s = sim();
            s.pilot_mode = mode;
            let start = s.world.agent(0).vehicle.position();
            let heading = yaw(s.world.agent(0).vehicle.orientation());
            // Tilt forward (in rates mode: pitch for a quarter second, then hold level rates).
            for k in 0..120 {
                let pitch = if mode == PilotMode::Attitude || k < 5 { 1.0 } else { 0.0 };
                s.stick = [pitch, 0.0, 0.0, 0.0];
                s.advance(1.0 / 60.0);
            }
            let d = s.world.agent(0).vehicle.position() - start;
            let forward = d.truncate().dot(glam::DVec2::from_angle(heading));
            assert!(forward > 1.0 && d.z.abs() < 2.0, "{mode:?}: moved {d}");
            assert!(!s.latched[0].intersects(Events::TERMINAL), "{mode:?}");
        }
    }
}
