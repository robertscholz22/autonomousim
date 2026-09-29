//! What the viewer shows: either a live [`WorldInstance`] stepped at its physics rate from the
//! frame clock (fixed-step accumulator) with keyboard flight or driving of one agent, or the
//! playback of a recording ([`Replay`]) that places the agents of the same kind of world.
//! Live, a trained policy ([`Autopilot`]) can fly the agents instead, and the session can be
//! recorded.

use crate::autopilot::{self, Autopilot};
use crate::camera::{CameraMode, CameraRig};
use crate::replay::Replay;
use autonomousim_control::Command;
use autonomousim_control::fixedwing::{FixedWingActionMap, FixedWingActionMode, FixedWingSetpoint, Lateral, Vertical};
use autonomousim_control::ground::{GroundActionMap, GroundActionMode, GroundSetpoint};
use autonomousim_control::multirotor::{Frame, Setpoint, YawCommand};
use autonomousim_control::rotorcraft::{HelicopterActionMap, HelicopterActionMode, HelicopterSetpoint};
use autonomousim_control::tiltrotor::{TiltrotorActionMap, TiltrotorActionMode, TiltrotorSetpoint};
use autonomousim_core::math::Pose;
use autonomousim_sensors::{CameraImage, Sensor};
use autonomousim_sim::camera::{self, Cameras};
use autonomousim_sim::record::Recorder;
use autonomousim_sim::{Events, WorldInstance};
use autonomousim_vehicles::Family;
use autonomousim_vehicles::ground::PowertrainDef;
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

/// Keyboard riding of single-track vehicles in `vk`: the rates (of the full-scale speed per
/// second) at which W raises the speed setpoint and S (or Space, faster) lowers it.
const RIDE_ACCEL: f64 = 0.15;
const RIDE_DECEL: f64 = 0.4;
const RIDE_STOP: f64 = 1.0;
/// The rate of a rider's turn command (of the full scale per second): a turn builds up over
/// a second, as the rider leans into it.
const RIDE_STEER_RATE: f64 = 1.0;
/// The lean of a full-stick turn (rad) on a motorcycle and on a bicycle: at the `vk` full
/// scale of 0.7 rad the rider regulator overshoots a sudden turn and the tyres lose grip; the
/// bicycle's regulator falls from about 0.35 rad of steady lean.
pub const RIDE_LEAN: [f64; 2] = [0.35, 0.2];
/// Below this speed (m/s) a rider's turn command fades in proportion: the bike hardly
/// balances itself there.
const RIDE_TURN_SPEED: f64 = 4.0;
/// Keyboard flight of aircraft: how fast the keys move the airspeed setpoint (m/s per second)
/// and the throttle in `rates` (full range per second × this).
const AIRSPEED_RATE: f64 = 3.0;
const THROTTLE_RATE: f64 = 0.5;

/// Keyboard flight of helicopters in `attitude` and `rates`: the collective Space/Shift add
/// to the trim's (of the full range).
const COLLECTIVE_KEY: f64 = 0.25;

/// How the keys fly the pilot. Aircraft: `Velocity` is `guidance` (A/D course rate,
/// Space/Shift climb rate, W/S airspeed setpoint), `Attitude` holds a bank (A/D) and a pitch
/// (W/S, forward is nose down, about the level-flight pitch) at an airspeed setpoint
/// (Space/Shift), and `Rates` flies body rates (Q/E yaw) with the throttle on Space/Shift.
/// Helicopters: see [`Sim::heli_setpoint`].
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

    /// The name for a vehicle of `family` (aircraft fly `guidance` for `velocity`).
    pub fn name_for(self, family: Family) -> &'static str {
        match (self, family) {
            (PilotMode::Velocity, Family::FixedWing) => "guidance",
            _ => self.name(),
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
    /// Single-track vehicles: the speed setpoint as a fraction of the `vk` full scale.
    pub ride_speed: f64,
    /// Aircraft: the airspeed setpoint (m/s) and the throttle in `rates`; `None` until the
    /// keys fly it (then from the current airspeed and throttle).
    pub airspeed: Option<f64>,
    pub throttle: Option<f64>,
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
    /// Renders the camera sensors (scenarios with cameras; in a mutex since the GPU objects
    /// are not `Sync`).
    cameras: Option<Mutex<Cameras>>,
    /// Replay: the last camera image rendered from the recorded state, with its episode,
    /// playback time, agent and sensor.
    replay_frame: Option<((usize, u64, usize, usize), CameraImage)>,
}

impl Sim {
    pub fn new(world: WorldInstance) -> Self {
        let n = world.agents().len();
        // Follow the first agent that is not scripted (scripted traffic may come first).
        let groups = &world.scenario().groups;
        let pilot = groups.iter().find(|g| !g.scripted()).map_or(0, |g| g.first_agent);
        // Aircraft fly by attitude from the start.
        let aircraft = world.agent(pilot).vehicle.family() == Family::FixedWing;
        let mut s = Self {
            world,
            pilot,
            paused: false,
            time_scale: 1.0,
            pilot_mode: if aircraft { PilotMode::Attitude } else { PilotMode::Velocity },
            stick: [0.0; 4],
            handbrake: false,
            steer: 0.0,
            ride_speed: 0.0,
            airspeed: None,
            throttle: None,
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
            cameras: None,
            replay_frame: None,
        };
        s.start_cameras();
        s.snapshot_poses();
        s
    }

    /// Set up the camera renderer for the world's scenario (none without cameras) and render
    /// the frames due now.
    fn start_cameras(&mut self) {
        self.cameras = None;
        self.replay_frame = None;
        if !camera::has_cameras(self.world.scenario()) {
            return;
        }
        match camera::gpu() {
            Ok(ctx) => self.cameras = Some(Mutex::new(Cameras::new(ctx, self.world.scenario()))),
            Err(e) => warn!("no camera images: {e}"),
        }
        self.capture();
    }

    /// Render and deliver the camera frames due at the current tick (live).
    fn capture(&mut self) {
        let Some(Ok(cameras)) = self.cameras.as_mut().map(Mutex::get_mut) else { return };
        if self.replay.is_none()
            && let Err(e) = cameras.update(&mut self.world)
        {
            warn!("camera images off: {e}");
            self.cameras = None;
        }
        if let Some(Ok(r)) = self.recorder.as_mut().map(Mutex::get_mut) {
            r.on_frames(&self.world);
        }
    }

    /// The image of camera `sensor` of `agent` with its capture time: live the frame its
    /// policy sees (noise and latency included), in a replay rendered from the recorded state
    /// at the playback time.
    pub fn camera_image(&mut self, agent: usize, sensor: usize) -> Option<(f64, &CameraImage)> {
        if let Some(r) = &self.replay {
            let key = (r.episode, r.time.to_bits(), agent, sensor);
            if self.replay_frame.as_ref().is_none_or(|(k, _)| *k != key) {
                let cameras = self.cameras.as_mut()?.get_mut().ok()?;
                let image = match cameras.render(&self.world, agent, sensor) {
                    Ok(image) => image?,
                    Err(e) => {
                        warn!("camera images off: {e}");
                        self.cameras = None;
                        return None;
                    }
                };
                self.replay_frame = Some((key, image));
            }
            let time = r.time;
            return self.replay_frame.as_ref().map(|(_, image)| (time, image));
        }
        match self.world.agents().get(agent)?.sensors.get(sensor)? {
            Sensor::Camera(c) => c.latest().map(|f| (f.time, &f.value)),
            _ => None,
        }
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
        self.ride_speed = 0.0;
        (self.airspeed, self.throttle) = (None, None);
        self.pilot = self.pilot.min(self.world.agents().len() - 1);
        self.latched = vec![Events::NONE; self.world.agents().len()];
        self.accumulator = 0.0;
        self.episodes += 1;
        if let Some(a) = &mut self.autopilot {
            a.ended = None;
        }
        self.start_cameras();
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
        self.capture();
        self.ride_speed = 0.0;
        (self.airspeed, self.throttle) = (None, None);
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
            // Scripted agents drive themselves.
            _ if self.world.agent(self.pilot).driver.is_some() => None,
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

    /// The `attitude` action map of the followed aircraft's group: its airspeed range and
    /// limits scale the keys.
    pub fn flight_map(&self) -> Option<FixedWingActionMap> {
        let agent = self.world.agent(self.pilot);
        let group = &self.world.scenario().groups[agent.group];
        let controller = agent.controller.as_fixed_wing()?;
        FixedWingActionMap::new(FixedWingActionMode::Attitude, &group.spec.fixed_wing_action_limits, controller).ok()
    }

    /// The keys' setpoint for the followed aircraft in the pilot mode (see [`PilotMode`]).
    pub fn flight_setpoint(&self) -> FixedWingSetpoint {
        let agent = self.world.agent(self.pilot);
        let (Some(map), Some(f), Some(c)) =
            (self.flight_map(), agent.vehicle.as_fixed_wing(), agent.controller.as_fixed_wing())
        else {
            return FixedWingSetpoint::Surfaces(Default::default());
        };
        let l = map.limits();
        let [forward, left, up, yaw] = self.stick;
        let airspeed = self.airspeed.unwrap_or_else(|| f.flow().airspeed);
        let [lo, hi] = map.airspeed_range();
        let airspeed = airspeed.clamp(lo, hi);
        match self.pilot_mode {
            PilotMode::Velocity => FixedWingSetpoint::Guidance {
                lateral: Lateral::CourseRate(left * l.course_rate),
                vertical: Vertical::ClimbRate(up * map.climb_rate()),
                airspeed,
            },
            PilotMode::Attitude => FixedWingSetpoint::Attitude {
                roll: -left * l.roll,
                pitch: c.model().trim_at(airspeed).alpha - forward * l.pitch,
                airspeed,
            },
            PilotMode::Rates => FixedWingSetpoint::Rates {
                rates: DVec3::new(-left, -forward, -yaw) * l.rate,
                throttle: self.throttle.unwrap_or(f.input().throttle),
            },
        }
    }

    /// Move the aircraft's airspeed setpoint (or throttle) with the keys by `dt` of simulated
    /// time.
    fn update_flight(&mut self, dt: f64) {
        let Some(f) = self.world.agent(self.pilot).vehicle.as_fixed_wing() else { return };
        let (airspeed, throttle) = (f.flow().airspeed, f.input().throttle);
        let Some(map) = self.flight_map() else { return };
        let [lo, hi] = map.airspeed_range();
        let key = if self.pilot_mode == PilotMode::Velocity { self.stick[0] } else { self.stick[2] };
        let v = self.airspeed.unwrap_or(airspeed).clamp(lo, hi);
        self.airspeed = Some((v + key * AIRSPEED_RATE * dt).clamp(lo, hi));
        let t = self.throttle.unwrap_or(throttle);
        self.throttle = Some((t + self.stick[2] * THROTTLE_RATE * dt).clamp(0.0, 1.0));
    }

    /// The followed helicopter's `velocity` action map (its group's limits): its speeds and
    /// limits scale the keys.
    pub fn heli_map(&self) -> Option<HelicopterActionMap> {
        let agent = self.world.agent(self.pilot);
        let group = &self.world.scenario().groups[agent.group];
        let (h, c) = (agent.vehicle.as_helicopter()?, agent.controller.as_helicopter()?);
        HelicopterActionMap::new(HelicopterActionMode::Velocity, &group.spec.helicopter_action_limits, h.def(), c).ok()
    }

    /// The keys' setpoint for the followed helicopter in the pilot mode: `velocity` as the
    /// action map has it (W/S forward, A/D sideways, Space/Shift climb, Q/E yaw); `attitude`
    /// tilts from the trim attitude at the airspeed (W forward is nose down, A banks left) and
    /// `rates` flies body rates, both with the trim collective raised or lowered by
    /// Space/Shift.
    pub fn heli_setpoint(&self) -> HelicopterSetpoint {
        let agent = self.world.agent(self.pilot);
        let (Some(map), Some(h), Some(c)) =
            (self.heli_map(), agent.vehicle.as_helicopter(), agent.controller.as_helicopter())
        else {
            return HelicopterSetpoint::Sticks(Default::default());
        };
        let [forward, left, up, yaw] = self.stick;
        if self.pilot_mode == PilotMode::Velocity {
            return map.setpoint(&[forward, left, up, yaw]);
        }
        let l = map.limits();
        let (trim, pitch, roll) = c.trim_at(h.flow().velocity.x);
        let collective = (trim.collective + COLLECTIVE_KEY * up).clamp(-1.0, 1.0);
        if self.pilot_mode == PilotMode::Attitude {
            HelicopterSetpoint::Attitude {
                roll: roll - left * l.roll,
                pitch: pitch - forward * l.pitch,
                yaw_rate: yaw * l.yaw_rate,
                collective,
            }
        } else {
            HelicopterSetpoint::Rates { rates: DVec3::new(-left, forward, yaw) * l.rates, collective }
        }
    }

    /// The followed tiltrotor's `velocity` action map (its group's limits): its speeds and
    /// limits scale the keys.
    pub fn tilt_map(&self) -> Option<TiltrotorActionMap> {
        let agent = self.world.agent(self.pilot);
        let group = &self.world.scenario().groups[agent.group];
        let (t, c) = (agent.vehicle.as_tiltrotor()?, agent.controller.as_tiltrotor()?);
        TiltrotorActionMap::new(TiltrotorActionMode::Velocity, &group.spec.tiltrotor_action_limits, t.def(), c).ok()
    }

    /// The range of the tiltrotor's speed setpoint in the pilot mode (m/s): backwards to
    /// forwards in `velocity`, an airspeed from 0 in `attitude`.
    fn tilt_speed_range(&self, map: &TiltrotorActionMap) -> [f64; 2] {
        let [forward, side, _] = map.speeds();
        if self.pilot_mode == PilotMode::Velocity { [-side, forward] } else { [0.0, forward] }
    }

    /// The keys' setpoint for the followed tiltrotor. `velocity`: W/S move the forward speed
    /// setpoint (the mounts convert with it), A/D fly sideways in hover and turn on the wing,
    /// Space/Shift climb, Q/E yaw. `attitude`: W/S move the airspeed setpoint, A/D bank, the
    /// pitch is the schedule's level-flight pitch at the airspeed, Space/Shift climb, Q/E yaw.
    pub fn tilt_setpoint(&self) -> TiltrotorSetpoint {
        let agent = self.world.agent(self.pilot);
        let (Some(map), Some(t), Some(c)) =
            (self.tilt_map(), agent.vehicle.as_tiltrotor(), agent.controller.as_tiltrotor())
        else {
            return TiltrotorSetpoint::Raw(Default::default());
        };
        let [_, left, up, yaw] = self.stick;
        let [lo, hi] = self.tilt_speed_range(&map);
        let speed = self.airspeed.unwrap_or_else(|| self.tilt_speed_now(t)).clamp(lo, hi);
        let l = map.limits();
        let [_, side, vertical] = map.speeds();
        let schedule = c.schedule();
        if self.pilot_mode == PilotMode::Velocity {
            let wing = schedule.wing_share(t.flow().airspeed);
            TiltrotorSetpoint::Velocity {
                velocity: DVec3::new(speed, (1.0 - wing) * left * side, up * vertical),
                yaw_rate: (yaw + wing * left).clamp(-1.0, 1.0) * l.yaw_rate,
            }
        } else {
            TiltrotorSetpoint::Attitude {
                roll: -left * l.roll,
                pitch: schedule.pitch(speed),
                yaw_rate: yaw * l.yaw_rate,
                climb: up * l.climb,
                airspeed: speed,
            }
        }
    }

    /// The tiltrotor's speed now as its setpoint in the pilot mode measures it: forward speed
    /// in the heading frame (`velocity`) or airspeed (`attitude`).
    fn tilt_speed_now(&self, t: &autonomousim_vehicles::tiltrotor::Tiltrotor) -> f64 {
        if self.pilot_mode == PilotMode::Velocity {
            let heading = autonomousim_core::math::quat::yaw(t.orientation());
            (glam::DQuat::from_rotation_z(-heading) * t.lin_vel_world()).x
        } else {
            t.flow().airspeed
        }
    }

    /// Move the tiltrotor's speed setpoint with W/S by `dt` of simulated time.
    fn update_tilt(&mut self, dt: f64) {
        let Some(map) = self.tilt_map() else { return };
        let Some(t) = self.world.agent(self.pilot).vehicle.as_tiltrotor() else { return };
        let [lo, hi] = self.tilt_speed_range(&map);
        let v = self.airspeed.unwrap_or_else(|| self.tilt_speed_now(t)).clamp(lo, hi);
        self.airspeed = Some((v + self.stick[0] * AIRSPEED_RATE * dt).clamp(lo, hi));
    }

    /// The pilot mode's name for the followed vehicle.
    pub fn pilot_mode_name(&self) -> &'static str {
        self.pilot_mode.name_for(self.world.agent(self.pilot).vehicle.family())
    }

    /// Whether the followed vehicle is a single-track vehicle, ridden in `vk`.
    pub fn riding(&self) -> bool {
        self.world.agent(self.pilot).vehicle.as_wheeled().is_some_and(|w| w.def().is_single_track())
    }

    /// The lean of a full-stick turn of the followed single-track vehicle (see [`RIDE_LEAN`]).
    pub fn ride_lean(&self) -> f64 {
        let motorcycle = self
            .world
            .agent(self.pilot)
            .vehicle
            .as_wheeled()
            .is_some_and(|w| matches!(w.def().powertrain, PowertrainDef::Combustion(_)));
        RIDE_LEAN[usize::from(!motorcycle)]
    }

    /// The `vk` action map with which the keys ride the followed single-track vehicle: from
    /// its group's limits, leaning at most [`RIDE_LEAN`].
    pub fn ride_map(&self) -> Option<GroundActionMap> {
        let agent = self.world.agent(self.pilot);
        let group = &self.world.scenario().groups[agent.group];
        let def = agent.vehicle.as_wheeled()?.def();
        let lean = self.ride_lean();
        let mut limits = group.spec.ground_action_limits.clone();
        limits.lean = Some(limits.lean.map_or(lean, |l| l.min(lean)));
        GroundActionMap::new(GroundActionMode::Vk, &limits, def).ok()
    }

    /// The pilot's command for the followed vehicle: [`setpoint`](Self::setpoint) for a
    /// multirotor, [`flight_setpoint`](Self::flight_setpoint) for an aircraft and
    /// [`heli_setpoint`](Self::heli_setpoint) for a helicopter. For a ground vehicle, forward/back is the pedal (brake, then reverse) and
    /// the steering follows left/right; side drives turn by driving their sides apart. A
    /// single-track vehicle is ridden in `vk`: the speed setpoint and the curvature (left/right,
    /// at most what the lean allows at the speed).
    pub fn pilot_command(&self) -> Command {
        let agent = self.world.agent(self.pilot);
        match agent.vehicle.family() {
            Family::Multirotor => self.setpoint().into(),
            Family::FixedWing => self.flight_setpoint().into(),
            Family::Rotorcraft => self.heli_setpoint().into(),
            Family::Tiltrotor => self.tilt_setpoint().into(),
            Family::Wheeled => {
                if let Some(sp) = self.drive_command {
                    return sp.into();
                }
                if self.riding()
                    && let Some(map) = self.ride_map()
                {
                    let fade = (agent.vehicle.lin_vel_body().x / RIDE_TURN_SPEED).clamp(0.0, 1.0);
                    return map.setpoint(&[self.ride_speed, self.steer * fade]).into();
                }
                let forward = self.stick[0];
                if agent.controller.as_ground().is_some_and(|c| c.is_side_drive()) {
                    let (left, right) =
                        ((forward - self.steer).clamp(-1.0, 1.0), (forward + self.steer).clamp(-1.0, 1.0));
                    GroundSetpoint::Sides { left, right }.into()
                } else {
                    GroundSetpoint::Pedal { drive: forward, steering: self.steer, handbrake: self.handbrake, lean: 0.0 }
                        .into()
                }
            }
        }
    }

    /// Move the steering command toward the keys by `dt` of simulated time (and a rider's
    /// speed setpoint).
    fn update_steering(&mut self, dt: f64) {
        if self.world.agent(self.pilot).vehicle.family() == Family::FixedWing {
            self.update_flight(dt);
            return;
        }
        if self.world.agent(self.pilot).vehicle.family() == Family::Tiltrotor {
            self.update_tilt(dt);
            return;
        }
        if self.riding() {
            let rate = match (self.handbrake, self.stick[0]) {
                (true, _) => -RIDE_STOP,
                (false, f) if f > 0.0 => f * RIDE_ACCEL,
                (false, f) => f * RIDE_DECEL,
            };
            self.ride_speed = (self.ride_speed + rate * dt).clamp(0.0, 1.0);
            let step = RIDE_STEER_RATE * dt;
            self.steer += (self.stick[1] - self.steer).clamp(-step, step);
            return;
        }
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
            // Scripted groups (`driver`) at every policy step.
            if self.world.clock().tick.is_multiple_of(u64::from(self.world.scenario().decimation)) {
                self.world.drive();
            }
            for a in 0..self.world.agents().len() {
                self.world.agent_mut(a).events = Events::NONE;
            }
            self.world.tick();
            if let Some(a) = &self.autopilot {
                a.after_tick(&mut self.world, manual);
            }
            self.capture();
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
/// W/S pedal (brake, then reverse), A/D steering, Space handbrake; single-track vehicles: W/S
/// speed setpoint up/down, A/D curvature, Space stop. The free camera takes the flight keys.
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
        (sim.airspeed, sim.throttle) = (None, None);
    }
    if let Some(a) = &mut sim.autopilot
        && keys.just_pressed(KeyCode::KeyT)
    {
        a.flies_pilot = !a.flies_pilot;
    }
    if sim.replay.is_none() && !ground {
        if keys.just_pressed(KeyCode::KeyM) || mode {
            sim.pilot_mode = sim.pilot_mode.next();
            // Tiltrotors fly `velocity` and `attitude` only.
            if sim.pilot_mode == PilotMode::Rates && sim.world.agent(sim.pilot).vehicle.family() == Family::Tiltrotor {
                sim.pilot_mode = sim.pilot_mode.next();
            }
            sim.airspeed = None;
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
    use autonomousim_control::fixedwing::euler;
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

    /// A drone hovering with a down camera over a scripted car on a rural map: live, the car
    /// drives itself and the camera sees it; the recorded session replays the same images.
    #[test]
    fn camera_images_replay_as_seen_live() {
        use crate::replay::Replay;
        use autonomousim_sim::record::{RecorderConfig, Recording};
        camera::use_adapter(&autonomousim_render::AdapterChoice::Software);
        let sc = Scenario::from_toml(
            r#"
            name = "chase"
            map = { type = "rural", seed = 3, count = 1, cache = false }
            [[groups]]
            name = "drone"
            vehicle = "iris_like"
            action_mode = "velocity"
            sensors = [{ name = "down", type = "camera", width = 48, height = 40, fov_deg = 90.0, rate_hz = 25, mount = { rotation = [0.0, 1.5707963267948966, 0.0] } }]
            [[groups]]
            name = "cars"
            vehicle = "sedan_like"
            spawn = { on_road = true }
            driver = { type = "road" }
            "#,
        )
        .unwrap();
        let mut world = WorldInstance::new(Arc::new(sc.compile().unwrap()), Seed::from_u64(2));
        // The drone 12 m above the car.
        let car = world.agent(1).vehicle.position();
        world.agent_mut(0).vehicle.place(
            Pose::new(car + DVec3::Z * 12.0, glam::DQuat::IDENTITY),
            DVec3::ZERO,
            DVec3::ZERO,
        );
        let mut s = Sim::new(world);
        let path = std::env::temp_dir().join(format!("autonomousim-camera-replay-{}.mcap", std::process::id()));
        s.record(Recorder::create(&path, RecorderConfig { camera_hz: 25, ..Default::default() }).unwrap());
        let mut frames: Vec<(u64, f64, CameraImage)> = Vec::new();
        for _ in 0..120 {
            s.advance(1.0 / 60.0);
            let Sensor::Camera(c) = &s.world.agent(0).sensors[0] else { panic!("camera") };
            let f = c.latest().unwrap();
            if frames.last().is_none_or(|l| l.0 != f.tick) {
                frames.push((f.tick, f.time, f.value.clone()));
            }
        }
        s.finish_recording().unwrap();
        assert!(frames.len() >= 45, "{} frames", frames.len());
        assert!(s.world.agent(1).vehicle.position().distance(car) > 3.0, "the car drives itself");
        let vehicle = autonomousim_render::SemanticClass::Vehicle as u8;
        assert!(frames.iter().all(|(.., f)| f.class.contains(&vehicle)), "the camera sees the car");

        let recording = Recording::read(&path).unwrap();
        std::fs::remove_file(&path).ok();
        let world = WorldInstance::new(Arc::new(recording.compile().unwrap()), Seed::from_u64(0));
        let mut replay = Sim::replay(world, Replay::new(recording, 0));
        replay.paused = true;
        for (tick, time, live) in &frames {
            replay.replay.as_mut().unwrap().seek(*time);
            replay.advance(0.0);
            let shown = replay.camera_image(0, 0).expect("replayed image").1;
            assert!(shown == live, "frame of tick {tick} replays differently");
        }
    }

    /// Scripted traffic coming first: the viewer follows the first agent that is not scripted.
    #[test]
    fn follows_the_first_agent_not_scripted() {
        let sc = Scenario::from_toml(
            r#"
            name = "traffic"
            map = { type = "rural", seed = 3, count = 1, cache = false }
            [[groups]]
            name = "cars"
            count = 2
            vehicle = "sedan_like"
            spawn = { on_road = true }
            driver = { type = "road" }
            [[groups]]
            name = "drone"
            vehicle = "iris_like"
            "#,
        )
        .unwrap();
        let s = Sim::new(WorldInstance::new(Arc::new(sc.compile().unwrap()), Seed::from_u64(2)));
        assert_eq!(s.pilot, 2);
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

    /// An aircraft in the air, heading east at 1.5 V_s.
    fn aircraft_sim(vehicle: &str) -> Sim {
        let sc = Scenario::from_toml(&format!(
            r#"
            name = "fly"
            map = {{ type = "testworld", kind = "flat", size = 8000.0 }}
            [[groups]]
            vehicle = "{vehicle}"
            spawn = {{ region = [[-10.0, -10.0], [10.0, 10.0]], agl = [300.0, 300.0], yaw_deg = [0.0, 0.0], clearance = 0.0 }}
            "#
        ))
        .unwrap();
        Sim::new(WorldInstance::new(Arc::new(sc.compile().unwrap()), Seed::from_u64(1)))
    }

    #[test]
    fn keys_fly_an_aircraft() {
        let mut s = aircraft_sim("aerosonde_like");
        assert_eq!((s.pilot_mode, s.pilot_mode_name()), (PilotMode::Attitude, "attitude"));
        let z0 = s.world.agent(0).vehicle.position().z;
        // Hands off: straight and level, speeding up to the setpoint's floor of the range.
        let (_, turn) = drive(&mut s, [0.0; 4], 5.0);
        let v = &s.world.agent(0).vehicle;
        assert!(turn.abs() < 0.05 && (v.position().z - z0).abs() < 15.0, "turn {turn} z {}", v.position().z);
        // A: bank left and turn left; released, the wings level.
        let (_, turn) = drive(&mut s, [0.0, 1.0, 0.0, 0.0], 5.0);
        let roll = euler(s.world.agent(0).vehicle.orientation()).0;
        let limit = s.flight_map().unwrap().limits().roll;
        assert!(turn > 0.5 && (roll + limit).abs() < 0.1, "turn {turn} roll {roll}");
        drive(&mut s, [0.0; 4], 4.0);
        assert!(euler(s.world.agent(0).vehicle.orientation()).0.abs() < 0.05);
        // Space raises the airspeed setpoint, and the aircraft follows.
        let v0 = s.airspeed.unwrap();
        drive(&mut s, [0.0, 0.0, 1.0, 0.0], 2.0);
        assert!((s.airspeed.unwrap() - v0 - 2.0 * AIRSPEED_RATE).abs() < 0.2, "{v0} → {:?}", s.airspeed);
        drive(&mut s, [0.0; 4], 10.0);
        let f = s.world.agent(0).vehicle.as_fixed_wing().unwrap();
        assert!((f.flow().airspeed - s.airspeed.unwrap()).abs() < 1.0, "{} vs {:?}", f.flow().airspeed, s.airspeed);
        // Guidance: Space climbs.
        s.pilot_mode = PilotMode::Velocity;
        assert_eq!(s.pilot_mode_name(), "guidance");
        let z = s.world.agent(0).vehicle.position().z;
        drive(&mut s, [0.0, 0.0, 1.0, 0.0], 5.0);
        let climb = s.world.agent(0).vehicle.lin_vel_world().z;
        assert!(s.world.agent(0).vehicle.position().z > z + 5.0 && climb > 1.0, "climb {climb}");
        // Rates: Shift closes the throttle.
        s.pilot_mode = PilotMode::Rates;
        drive(&mut s, [0.0, 0.0, -1.0, 0.0], 3.0);
        assert!(
            s.throttle.unwrap() < 0.05 && s.world.agent(0).vehicle.as_fixed_wing().unwrap().input().throttle < 0.05
        );
        assert!(!s.latched[0].is_terminal(), "{:?}", s.latched[0]);
    }

    /// A helicopter hovering 50 m up, heading east.
    fn heli_sim(vehicle: &str) -> Sim {
        let sc = Scenario::from_toml(&format!(
            r#"
            name = "hover"
            map = {{ type = "testworld", kind = "flat", size = 4000.0 }}
            [[groups]]
            vehicle = "{vehicle}"
            spawn = {{ region = [[-10.0, -10.0], [10.0, 10.0]], agl = [50.0, 50.0], yaw_deg = [0.0, 0.0], clearance = 0.0 }}
            "#
        ))
        .unwrap();
        Sim::new(WorldInstance::new(Arc::new(sc.compile().unwrap()), Seed::from_u64(1)))
    }

    /// The keys fly a helicopter: in `velocity` it holds its place hands off, flies forward at
    /// the stick's share of the full speed and stops again; in `attitude` A banks it left from
    /// the trim and Space climbs; in `rates` E yaws it right.
    #[test]
    fn keys_fly_a_helicopter() {
        let mut s = heli_sim("xcell60_like");
        assert_eq!((s.pilot_mode, s.pilot_mode_name()), (PilotMode::Velocity, "velocity"));
        let full = s.heli_map().unwrap().speeds()[0];
        let (moved, turn) = drive(&mut s, [0.0; 4], 3.0);
        assert!(moved.abs() < 0.5 && turn.abs() < 0.05, "moved {moved}, turn {turn}");
        let (moved, _) = drive(&mut s, [0.5, 0.0, 0.0, 0.0], 15.0);
        let speed = s.world.agent(0).vehicle.lin_vel_world().x;
        assert!((speed - 0.5 * full).abs() < 0.5 && moved > 0.25 * full * 15.0, "speed {speed} of {full}");
        drive(&mut s, [0.0; 4], 15.0);
        let v = s.world.agent(0).vehicle.lin_vel_world();
        assert!(v.length() < 0.3, "{v}");
        s.pilot_mode = PilotMode::Attitude;
        let limit = s.heli_map().unwrap().limits().roll;
        let roll = |s: &Sim| euler(s.world.agent(0).vehicle.orientation()).0;
        let trim = s.world.agent(0).controller.as_helicopter().unwrap().trim_at(0.0).2;
        drive(&mut s, [0.0, 1.0, 0.0, 0.0], 1.5);
        assert!(roll(&s) < trim - 0.8 * limit, "roll {} trim {trim}", roll(&s));
        assert!(s.world.agent(0).vehicle.lin_vel_body().y > 1.0);
        drive(&mut s, [0.0; 4], 3.0);
        let h = s.world.agent(0).vehicle.as_helicopter().unwrap();
        let trim = s.world.agent(0).controller.as_helicopter().unwrap().trim_at(h.flow().velocity.x).2;
        assert!((roll(&s) - trim).abs() < 0.05, "roll {} trim {trim}", roll(&s));
        drive(&mut s, [0.0, 0.0, 1.0, 0.0], 2.0);
        let climb = s.world.agent(0).vehicle.lin_vel_world().z;
        assert!(climb > 0.5, "climb {climb}");
        s.pilot_mode = PilotMode::Rates;
        let (_, turn) = drive(&mut s, [0.0, 0.0, 0.0, -1.0], 1.0);
        assert!(turn < -0.5, "turn {turn}");
        assert!(!s.latched[0].is_terminal(), "{:?}", s.latched[0]);
    }

    /// Shift in `velocity` (descending at 1 m/s, below the crash speed) sets both helicopters
    /// down on their skids, upright and still.
    #[test]
    fn keys_land_a_helicopter() {
        for vehicle in ["xcell60_like", "bo105_like"] {
            let mut s = heli_sim(vehicle);
            let down = 1.0 / s.heli_map().unwrap().speeds()[2];
            drive(&mut s, [0.0, 0.0, -down, 0.0], 70.0);
            drive(&mut s, [0.0; 4], 5.0);
            let v = &s.world.agent(0).vehicle;
            let (roll, pitch, _) = euler(v.orientation());
            assert!(s.latched[0].contains(Events::LANDED), "{vehicle}: {:?}", s.latched[0]);
            assert!(!s.latched[0].is_terminal(), "{vehicle}: {:?}", s.latched[0]);
            assert!(roll.abs() < 0.1 && pitch.abs() < 0.1, "{vehicle}: roll {roll} pitch {pitch}");
            assert!(v.lin_vel_world().length() < 0.1, "{vehicle}: {}", v.lin_vel_world());
            // Space lifts it off again, level (no rollover from the time on the skids).
            let z = v.position().z;
            drive(&mut s, [0.0, 0.0, 0.5, 0.0], 8.0);
            let v = &s.world.agent(0).vehicle;
            let (roll, _, _) = euler(v.orientation());
            assert!(v.position().z > z + 3.0 && roll.abs() < 0.15, "{vehicle}: z {} roll {roll}", v.position().z);
            assert!(!s.latched[0].is_terminal(), "{vehicle}: {:?}", s.latched[0]);
        }
    }

    /// The keys fly a tiltrotor through both transitions: in `velocity` it hovers in place
    /// hands off; W raises the speed setpoint and the mounts convert onto the wing; A turns
    /// it left there; S brings the setpoint back to 0 and it converts back to a hover, the
    /// height held all along. In `attitude` A banks it left.
    #[test]
    fn keys_fly_a_tiltrotor() {
        let mut s = heli_sim("quadtilt_like");
        assert_eq!((s.pilot_mode, s.pilot_mode_name()), (PilotMode::Velocity, "velocity"));
        let z0 = s.world.agent(0).vehicle.position().z;
        let height = |s: &Sim| (s.world.agent(0).vehicle.position().z - z0).abs();
        let (moved, turn) = drive(&mut s, [0.0; 4], 3.0);
        assert!(moved.abs() < 0.5 && turn.abs() < 0.05, "moved {moved}, turn {turn}");
        let cruise = (1.8 * s.world.agent(0).controller.as_tiltrotor().unwrap().schedule().stall_speed()).round();
        drive(&mut s, [1.0, 0.0, 0.0, 0.0], cruise / AIRSPEED_RATE);
        assert!((s.airspeed.unwrap() - cruise).abs() < 0.2, "{:?}", s.airspeed);
        drive(&mut s, [0.0; 4], 20.0);
        let t = s.world.agent(0).vehicle.as_tiltrotor().unwrap();
        assert!((t.flow().airspeed - cruise).abs() < 1.0, "{} vs {cruise}", t.flow().airspeed);
        assert!(t.tilts().iter().all(|x| *x > 1.5) && height(&s) < 5.0, "{:?} dz {}", t.tilts(), height(&s));
        let (_, turn) = drive(&mut s, [0.0, 1.0, 0.0, 0.0], 4.0);
        let roll = euler(s.world.agent(0).vehicle.orientation()).0;
        assert!(turn > 0.5 && roll < -0.1, "turn {turn} roll {roll}");
        drive(&mut s, [0.0; 4], 5.0);
        drive(&mut s, [-1.0, 0.0, 0.0, 0.0], cruise / AIRSPEED_RATE);
        assert!(s.airspeed.unwrap().abs() < 0.2, "{:?}", s.airspeed);
        drive(&mut s, [0.0; 4], 30.0);
        let t = s.world.agent(0).vehicle.as_tiltrotor().unwrap();
        assert!(t.lin_vel_world().length() < 0.3, "{}", t.lin_vel_world());
        assert!(t.tilts().iter().all(|x| x.abs() < 0.05) && height(&s) < 5.0, "{:?} dz {}", t.tilts(), height(&s));
        s.pilot_mode = PilotMode::Attitude;
        s.airspeed = None;
        drive(&mut s, [0.0, 1.0, 0.0, 0.0], 2.0);
        let roll = euler(s.world.agent(0).vehicle.orientation()).0;
        assert!(roll < -0.1 && s.world.agent(0).vehicle.lin_vel_body().y > 0.5, "roll {roll}");
        assert!(!s.latched[0].is_terminal(), "{:?}", s.latched[0]);
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

    /// The keys ride a motorcycle in `vk`: W raises the speed setpoint, A turns it left (it
    /// leans into the turn), Space stops it on its feet.
    #[test]
    fn keys_ride_a_motorcycle() {
        let mut s = ground_sim("motorcycle_sport");
        assert!(s.riding());
        let full = s.ride_map().unwrap().speed();
        drive(&mut s, [0.0; 4], 0.5);
        drive(&mut s, [1.0, 0.0, 0.0, 0.0], 4.0);
        assert!((s.ride_speed - 4.0 * RIDE_ACCEL).abs() < 0.01, "setpoint {}", s.ride_speed);
        let (forward, turn) = drive(&mut s, [0.0; 4], 4.0);
        let speed = s.world.agent(0).vehicle.lin_vel_body().x;
        assert!((speed - s.ride_speed * full).abs() < 1.0 && forward > 20.0, "speed {speed}, moved {forward}");
        assert!(turn.abs() < 0.1, "turn {turn}");
        let bike = |s: &Sim| s.world.agent(0).vehicle.as_wheeled().unwrap().feet_down();
        assert!(!bike(&s));
        let (_, turn) = drive(&mut s, [0.0, 0.5, 0.0, 0.0], 3.0);
        let roll = s.world.agent(0).vehicle.orientation().to_euler(glam::EulerRot::ZYX).2;
        assert!(turn > 0.3 && roll < -0.1, "turn {turn}, roll {roll}");
        drive(&mut s, [0.0; 4], 2.0);
        s.handbrake = true;
        drive(&mut s, [0.0; 4], 6.0);
        assert_eq!(s.ride_speed, 0.0);
        assert!(s.world.agent(0).vehicle.lin_vel_body().x.abs() < 0.3 && bike(&s));
        assert!(!s.latched[0].intersects(Events::TERMINAL), "{:?}", s.latched[0]);
    }

    /// Full-stick turns from the keys, at several speeds, lean both bikes within reach of
    /// their riders: no crash.
    #[test]
    fn full_stick_turns_stay_upright() {
        for vehicle in ["motorcycle_sport", "bicycle_city"] {
            for speed in [0.2, 0.4, 0.6] {
                let mut s = ground_sim(vehicle);
                s.ride_speed = speed;
                drive(&mut s, [0.0; 4], 6.0);
                drive(&mut s, [0.0, 1.0, 0.0, 0.0], 6.0);
                drive(&mut s, [0.0, -1.0, 0.0, 0.0], 4.0);
                let roll = s.world.agent(0).vehicle.orientation().to_euler(glam::EulerRot::ZYX).2;
                assert!(roll.abs() < 0.6, "{vehicle} at {speed}: roll {roll}");
                assert!(!s.latched[0].intersects(Events::TERMINAL), "{vehicle} at {speed}: {:?}", s.latched[0]);
            }
        }
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
        s.drive_command = Some(GroundSetpoint::Pedal { drive: -0.5, steering: 0.0, handbrake: true, lean: 0.0 });
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
