//! The bridge: runs a [`WorldInstance`] and publishes it as a ROS 2 node.
//!
//! - `/clock` (sim time, continuous across episodes), `/tf` (`map` → `agent<id>/base_link` at the
//!   odometry rate), `/tf_static` (sensor mounts, latched), `/autonomousim/meta` (latched JSON).
//! - Per bridged agent (the agents of groups without a scripted driver), under `/agent<id>/`:
//!   `odom` (`nav_msgs/Odometry`: pose in `map`, twist in `base_link`), `joint_states` (wheels
//!   and steering of wheeled vehicles in full physics), `events` (`std_msgs/UInt32`, the event
//!   bits of each policy step that raised any) and one topic per sensor, published at the
//!   sensor's own rate and stamped with its measurement time: `imu` (`sensor_msgs/Imu`), `gps`
//!   (`NavSatFix`), `baro` (`FluidPressure`), `mag` (`MagneticField`), `range` (`Range`),
//!   named after the sensor (`/agent0/<name>`), in the frame `agent<id>/<name>`.
//! - Commands, per bridged agent not flown by a policy: `cmd_vel` (`geometry_msgs/Twist`;
//!   multirotors: velocity in the heading frame and heading rate; wheeled vehicles: forward
//!   speed and yaw rate) and `action` (`std_msgs/Float32MultiArray`, the normalized action of
//!   the group's mode). Commands are held; an agent without one for `command_timeout` holds
//!   still (hovers where it is, brakes). In lockstep each policy step waits until every
//!   commanded agent has sent a command since the last `/clock`.
//! - Services: `/autonomousim/reset` (`std_srvs/Trigger`: a new episode) and
//!   `/autonomousim/pause` (`std_srvs/SetBool`: simulated time stands while paused).

use crate::msgs::builtin_interfaces::Time;
use crate::msgs::geometry_msgs::{Quaternion, Transform, TransformStamped, Twist, Vector3};
use crate::msgs::nav_msgs::Odometry;
use crate::msgs::rosgraph_msgs::Clock;
use crate::msgs::sensor_msgs::{FluidPressure, Imu, JointState, MagneticField, NavSatFix, NavSatStatus, Range};
use crate::msgs::std_msgs::{Float32MultiArray, Header, StringMsg, UInt32};
use crate::msgs::std_srvs::{SetBoolRequest, SetBoolResponse, TriggerRequest, TriggerResponse};
use crate::msgs::tf2_msgs::TFMessage;
use crate::msgs::{Array, RosMessage};
use crate::node::{Publisher, RosNode, Server, Subscription, qos};
use autonomousim_control::Command;
use autonomousim_control::ground::GroundSetpoint;
use autonomousim_control::multirotor::{Frame, Setpoint, YawCommand};
use autonomousim_core::rng::Seed;
use autonomousim_sensors::Sensor;
use autonomousim_sim::policy::Policy;
use autonomousim_sim::{CompiledScenario, WorldInstance};
use glam::{DQuat, DVec3};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// How the bridge paces the simulation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Pacing {
    /// Simulated time runs at this multiple of wall-clock time.
    Realtime(f64),
    /// As fast as it steps.
    Fast,
}

#[derive(Clone, Debug)]
pub struct BridgeConfig {
    /// DDS domain (`ROS_DOMAIN_ID`).
    pub domain_id: u16,
    pub pacing: Pacing,
    /// Odometry and `/tf` rate (Hz; 0: the policy rate). Must divide the policy rate.
    pub odom_hz: u32,
    /// Episodes end after this much simulated time (s; none: when every bridged agent is
    /// disabled).
    pub episode_time: Option<f64>,
    /// Episode seed of the first episode.
    pub seed: u64,
    /// Simulated time (s) after an agent's last command when it is made to hold still.
    pub command_timeout: f64,
    /// Wait before each policy step for a command from every commanded agent, at most this
    /// long (wall clock, s); none: step without waiting.
    pub lockstep: Option<f64>,
}

impl Default for BridgeConfig {
    fn default() -> Self {
        Self {
            domain_id: 0,
            pacing: Pacing::Realtime(1.0),
            odom_hz: 0,
            episode_time: None,
            seed: 0,
            command_timeout: 0.5,
            lockstep: None,
        }
    }
}

/// A best-effort send: a full or broken writer drops the message rather than stall the
/// simulation (DDS reports nothing the bridge could act on).
fn send<M: RosMessage>(publisher: &Publisher<M>, msg: M) {
    let _ = publisher.publish(msg);
}

pub fn vector(v: DVec3) -> Vector3 {
    Vector3 { x: v.x, y: v.y, z: v.z }
}

pub fn quaternion(q: DQuat) -> Quaternion {
    Quaternion { x: q.x, y: q.y, z: q.z, w: q.w }
}

fn header(t: f64, frame: &str) -> Header {
    Header { stamp: Time::from_secs(t), frame_id: frame.to_string() }
}

/// The frame of agent `id`'s body.
pub fn base_frame(id: u32) -> String {
    format!("agent{id}/base_link")
}

enum SensorTopic {
    Imu(Publisher<Imu>),
    Gps(Publisher<NavSatFix>),
    Baro(Publisher<FluidPressure>),
    Mag(Publisher<MagneticField>),
    Range(Publisher<Range>, f32, f32),
}

struct SensorPub {
    /// Index in the agent's sensors.
    sensor: usize,
    frame: String,
    topic: SensorTopic,
    /// Physics tick of the last reading published.
    last: Option<u64>,
}

impl SensorPub {
    /// Publish the sensor's latest reading if it is new (`offset`: time of earlier episodes).
    fn publish(&mut self, sensor: &Sensor, offset: f64) {
        macro_rules! fresh {
            ($s:expr) => {{
                let Some(r) = $s.latest() else { return };
                if self.last == Some(r.tick) {
                    return;
                }
                self.last = Some(r.tick);
                r
            }};
        }
        match (&self.topic, sensor) {
            (SensorTopic::Imu(p), Sensor::Imu(s)) => {
                let r = fresh!(s);
                // No orientation estimate: covariance[0] = −1 (REP-145).
                let mut orientation_covariance = Array::default();
                orientation_covariance.0[0] = -1.0;
                send(
                    p,
                    Imu {
                        header: header(offset + r.time, &self.frame),
                        orientation: Quaternion::default(),
                        orientation_covariance,
                        angular_velocity: vector(r.value.gyro),
                        angular_velocity_covariance: Array::default(),
                        linear_acceleration: vector(r.value.accel),
                        linear_acceleration_covariance: Array::default(),
                    },
                );
            }
            (SensorTopic::Gps(p), Sensor::Gps(s)) => {
                let r = fresh!(s);
                let g = &r.value;
                let (h, v) = (g.eph * g.eph, g.epv * g.epv);
                send(
                    p,
                    NavSatFix {
                        header: header(offset + r.time, &self.frame),
                        status: NavSatStatus { status: NavSatStatus::STATUS_FIX, service: NavSatStatus::SERVICE_GPS },
                        latitude: g.geodetic.lat_deg,
                        longitude: g.geodetic.lon_deg,
                        altitude: g.geodetic.alt,
                        position_covariance: Array([h, 0.0, 0.0, 0.0, h, 0.0, 0.0, 0.0, v]),
                        position_covariance_type: NavSatFix::COVARIANCE_TYPE_DIAGONAL_KNOWN,
                    },
                );
            }
            (SensorTopic::Baro(p), Sensor::Baro(s)) => {
                let r = fresh!(s);
                let fluid_pressure = r.value.pressure;
                send(p, FluidPressure { header: header(offset + r.time, &self.frame), fluid_pressure, variance: 0.0 });
            }
            (SensorTopic::Mag(p), Sensor::Mag(s)) => {
                let r = fresh!(s);
                send(
                    p,
                    MagneticField {
                        header: header(offset + r.time, &self.frame),
                        magnetic_field: vector(r.value.field),
                        magnetic_field_covariance: Array::default(),
                    },
                );
            }
            (SensorTopic::Range(p, min, max), Sensor::Rangefinder(s)) => {
                let r = fresh!(s);
                // No return within range: +inf (REP-117).
                let range = r.value.range.map_or(f32::INFINITY, |d| d as f32);
                send(
                    p,
                    Range {
                        header: header(offset + r.time, &self.frame),
                        radiation_type: Range::INFRARED,
                        field_of_view: 0.0,
                        min_range: *min,
                        max_range: *max,
                        range,
                        variance: 0.0,
                    },
                );
            }
            _ => unreachable!("sensor topics are built from the sensors"),
        }
    }
}

/// A command for one agent, as the bridge receives it.
#[derive(Clone, Debug, PartialEq)]
pub enum AgentCommand {
    /// `cmd_vel`: multirotors fly `linear` (heading frame: x forward, y left, z up; m/s) and
    /// turn at `angular.z` (rad/s); wheeled vehicles drive at `linear.x` and yaw at
    /// `angular.z` (a steered vehicle turns on the path curvature `angular.z / linear.x`).
    Twist(Twist),
    /// The normalized action of the group's action mode.
    Action(Vec<f32>),
}

struct AgentPubs {
    /// Index in the world.
    agent: usize,
    id: u32,
    frame: String,
    odom: Publisher<Odometry>,
    joints: Option<Publisher<JointState>>,
    events: Publisher<UInt32>,
    sensors: Vec<SensorPub>,
    cmd_vel: Subscription<Twist>,
    action: Subscription<Float32MultiArray>,
    /// Flown by a policy (commands are ignored).
    flown: bool,
    /// Bridge time of the last command; none since the episode began.
    last_command: Option<f64>,
    /// Holding still (no command for `command_timeout`).
    holding: bool,
    /// A command arrived since the last policy step.
    fresh: bool,
}

/// A sensor's static transform from the body (`base_link`); a rangefinder's frame has its x
/// axis along the beam, as `sensor_msgs/Range` expects.
fn mount_transform(sensor: &Sensor) -> Option<(DVec3, DQuat)> {
    let (mount, extra) = match sensor {
        Sensor::Imu(s) => (s.config().mount, DQuat::IDENTITY),
        Sensor::Gps(s) => (s.config().mount, DQuat::IDENTITY),
        Sensor::Baro(s) => (s.config().mount, DQuat::IDENTITY),
        Sensor::Mag(s) => (s.config().mount, DQuat::IDENTITY),
        // The beam direction alone orients a rangefinder (its mount rotation is unused).
        Sensor::Rangefinder(s) => {
            let c = s.config();
            return Some((c.mount.position, DQuat::from_rotation_arc(DVec3::X, c.direction.normalize())));
        }
        Sensor::Pitot(_) | Sensor::Lidar(_) | Sensor::Camera(_) | Sensor::GroundTruth(_) => return None,
    };
    Some((mount.position, mount.quat() * extra))
}

/// A bridged simulation.
pub struct Bridge {
    world: WorldInstance,
    config: BridgeConfig,
    _node: RosNode,
    clock: Publisher<Clock>,
    tf: Publisher<TFMessage>,
    _tf_static: Publisher<TFMessage>,
    _meta: Publisher<StringMsg>,
    agents: Vec<AgentPubs>,
    /// Policy steps per odometry message.
    odom_divider: u64,
    /// Simulated time of the episodes before the current one (s).
    offset: f64,
    /// Exported policies flying groups: (group, policy).
    policies: Vec<(usize, Policy)>,
    obs: Vec<f32>,
    action: Vec<f32>,
    /// Wall-clock time and simulated time at which pacing started.
    paced_from: Option<(Instant, f64)>,
    episodes: u64,
    reset_server: Server<TriggerRequest, TriggerResponse>,
    pause_server: Server<SetBoolRequest, SetBoolResponse>,
    paused: bool,
    reset_requested: bool,
    /// Any command received yet (lockstep waits without a time limit until then).
    commanded: bool,
    steps_total: u64,
}

impl Bridge {
    pub fn new(scenario: Arc<CompiledScenario>, config: BridgeConfig) -> anyhow::Result<Self> {
        let policy_hz = scenario.spec.policy_hz;
        let odom_hz = if config.odom_hz == 0 { policy_hz } else { config.odom_hz };
        if odom_hz == 0 || !policy_hz.is_multiple_of(odom_hz) {
            anyhow::bail!("the odometry rate ({odom_hz} Hz) must divide the policy rate ({policy_hz} Hz)");
        }
        let world = WorldInstance::new(scenario.clone(), Seed::from_u64(config.seed));
        let mut node = RosNode::new("/", "autonomousim", config.domain_id)?;
        let clock =
            node.publisher::<Clock>("/clock", qos::RELIABLE.history(ros2_client::qos::History::KeepLast { depth: 1 }))?;
        let tf = node.publisher::<TFMessage>("/tf", qos::RELIABLE)?;
        let tf_static = node.publisher::<TFMessage>("/tf_static", qos::LATCHED)?;
        let meta = node.publisher::<StringMsg>("/autonomousim/meta", qos::LATCHED)?;
        let mut agents = Vec::new();
        let mut statics = Vec::new();
        let mut meta_agents = Vec::new();
        for (i, a) in world.agents().iter().enumerate() {
            let g = &scenario.groups[a.group];
            if g.scripted() {
                continue;
            }
            let ns = format!("/agent{}", a.id);
            let frame = base_frame(a.id);
            let mut sensors = Vec::new();
            let mut topics =
                vec![format!("{ns}/odom"), format!("{ns}/events"), format!("{ns}/cmd_vel"), format!("{ns}/action")];
            for (k, (s, spec)) in a.sensors.iter().zip(&g.spec.sensors).enumerate() {
                let Some((position, rotation)) = mount_transform(s) else { continue };
                let topic = format!("{ns}/{}", spec.name);
                let sensor_frame = format!("agent{}/{}", a.id, spec.name);
                let t = match s {
                    Sensor::Imu(_) => SensorTopic::Imu(node.publisher(&topic, qos::SENSOR)?),
                    Sensor::Gps(_) => SensorTopic::Gps(node.publisher(&topic, qos::SENSOR)?),
                    Sensor::Baro(_) => SensorTopic::Baro(node.publisher(&topic, qos::SENSOR)?),
                    Sensor::Mag(_) => SensorTopic::Mag(node.publisher(&topic, qos::SENSOR)?),
                    Sensor::Rangefinder(r) => {
                        let c = r.config();
                        SensorTopic::Range(node.publisher(&topic, qos::SENSOR)?, c.min_range as f32, c.max_range as f32)
                    }
                    _ => unreachable!("mount_transform filters the others"),
                };
                statics.push(TransformStamped {
                    header: header(0.0, &frame),
                    child_frame_id: sensor_frame.clone(),
                    transform: Transform { translation: vector(position), rotation: quaternion(rotation) },
                });
                topics.push(topic);
                sensors.push(SensorPub { sensor: k, frame: sensor_frame, topic: t, last: None });
            }
            let joints = if a.vehicle.as_wheeled().is_some() {
                topics.push(format!("{ns}/joint_states"));
                Some(node.publisher(&format!("{ns}/joint_states"), qos::RELIABLE)?)
            } else {
                None
            };
            meta_agents.push(serde_json::json!({
                "id": a.id, "group": g.spec.name, "frame": frame, "topics": topics,
            }));
            agents.push(AgentPubs {
                agent: i,
                id: a.id,
                odom: node.publisher(&format!("{ns}/odom"), qos::RELIABLE)?,
                events: node.publisher(&format!("{ns}/events"), qos::RELIABLE)?,
                joints,
                frame,
                sensors,
                cmd_vel: node.subscription(&format!("{ns}/cmd_vel"), qos::RELIABLE)?,
                action: node.subscription(&format!("{ns}/action"), qos::RELIABLE)?,
                flown: false,
                last_command: None,
                holding: false,
                fresh: false,
            });
        }
        send(&tf_static, TFMessage { transforms: statics });
        let meta_json = serde_json::json!({
            "format": 1,
            "scenario": scenario.spec,
            "map_hashes": scenario.map_hashes.iter().map(|h| h.hex()).collect::<Vec<_>>(),
            "physics_hz": scenario.spec.physics_hz,
            "policy_hz": policy_hz,
            "odom_hz": odom_hz,
            "agents": meta_agents,
        });
        send(&meta, StringMsg { data: meta_json.to_string() });
        let reset_server = node.server("/autonomousim/reset", "std_srvs/Trigger")?;
        let pause_server = node.server("/autonomousim/pause", "std_srvs/SetBool")?;
        Ok(Self {
            world,
            config,
            _node: node,
            clock,
            tf,
            _tf_static: tf_static,
            _meta: meta,
            agents,
            odom_divider: u64::from(policy_hz / odom_hz),
            offset: 0.0,
            policies: Vec::new(),
            obs: Vec::new(),
            action: Vec::new(),
            paced_from: None,
            episodes: 0,
            reset_server,
            pause_server,
            paused: false,
            reset_requested: false,
            commanded: false,
            steps_total: 0,
        })
    }

    /// Let an exported policy fly the agents of `group`.
    pub fn fly(&mut self, group: usize, policy: Policy) -> anyhow::Result<()> {
        let g = &self.world.scenario().groups[group];
        if (g.obs_dim(), g.act_dim()) != (policy.obs_dim(), policy.act_dim()) || policy.image_shape().is_some() {
            anyhow::bail!(
                "group {:?} observes {} values and takes {} actions, the policy {} and {} (pixel policies are not supported)",
                g.spec.name,
                g.obs_dim(),
                g.act_dim(),
                policy.obs_dim(),
                policy.act_dim()
            );
        }
        let first = g.first_agent..g.first_agent + g.spec.count;
        for a in &mut self.agents {
            a.flown |= first.contains(&a.agent);
        }
        self.policies.push((group, policy));
        Ok(())
    }

    /// Command bridged agent `id` (what its `cmd_vel` or `action` topic does).
    pub fn command(&mut self, id: u32, command: AgentCommand) -> anyhow::Result<()> {
        let t = self.time();
        let k =
            self.agents.iter().position(|a| a.id == id).ok_or_else(|| anyhow::anyhow!("agent {id} is not bridged"))?;
        let a = &mut self.agents[k];
        if a.flown {
            anyhow::bail!("agent {id} is flown by a policy");
        }
        let agent = self.world.agent(a.agent);
        match command {
            AgentCommand::Action(action) => {
                let dim = self.world.scenario().groups[agent.group].act_dim();
                if action.len() != dim {
                    anyhow::bail!("agent {id} takes {dim} action values, not {}", action.len());
                }
                self.world.set_action(a.agent, &action);
            }
            AgentCommand::Twist(twist) => {
                let (v, w) = (twist.linear, twist.angular.z);
                let command: Command = if agent.controller.as_multirotor().is_some() {
                    Setpoint::Velocity {
                        velocity: DVec3::new(v.x, v.y, v.z),
                        frame: Frame::Heading,
                        yaw: YawCommand::Rate(w),
                    }
                    .into()
                } else if let Some(c) = agent.controller.as_ground() {
                    if c.is_side_drive() {
                        GroundSetpoint::SpeedYawRate { speed: v.x, yaw_rate: w }.into()
                    } else {
                        // The curvature of that yaw rate; near standstill, that of a crawl.
                        let speed = if v.x.abs() < 0.1 { 0.1 } else { v.x };
                        GroundSetpoint::SpeedCurvature { speed: v.x, curvature: w / speed }.into()
                    }
                } else {
                    anyhow::bail!("cmd_vel drives multirotors and wheeled vehicles; agent {id} takes `action`");
                };
                self.world.set_command(a.agent, command);
            }
        }
        a.last_command = Some(t);
        a.holding = false;
        a.fresh = true;
        self.commanded = true;
        Ok(())
    }

    /// Simulated time stands still.
    pub fn paused(&self) -> bool {
        self.paused
    }

    pub fn set_paused(&mut self, paused: bool) {
        self.paused = paused;
    }

    /// Start a new episode after the next step (or now, while paused).
    pub fn request_reset(&mut self) {
        self.reset_requested = true;
    }

    /// Take the commands that arrived (the latest of each agent) and answer service calls.
    fn poll(&mut self) {
        let mut received = Vec::new();
        for a in &self.agents {
            let mut last = None;
            while let Ok(Some((t, _))) = a.cmd_vel.take() {
                last = Some(AgentCommand::Twist(t));
            }
            while let Ok(Some((m, _))) = a.action.take() {
                last = Some(AgentCommand::Action(m.data));
            }
            if let Some(c) = last.filter(|_| !a.flown) {
                received.push((a.id, c));
            }
        }
        for (id, c) in received {
            if let Err(e) = self.command(id, c) {
                eprintln!("command ignored: {e}");
            }
        }
        while let Ok(Some((id, _))) = self.reset_server.receive_request() {
            self.reset_requested = true;
            let message = format!("episode {} begins after this step", self.episodes + 2);
            let _ = self.reset_server.send_response(id, TriggerResponse { success: true, message });
        }
        while let Ok(Some((id, req))) = self.pause_server.receive_request() {
            self.paused = req.data;
            let message = if req.data { "paused" } else { "running" }.to_string();
            let _ = self.pause_server.send_response(id, SetBoolResponse { success: true, message });
        }
    }

    /// Lockstep: wait until every commanded agent has a fresh command (at most `limit` s once
    /// any command has come; until then, without a limit), sending state and clock again now
    /// and then for controllers that join late.
    fn wait_for_commands(&mut self, limit: f64) {
        let start = Instant::now();
        let mut resent = start;
        let mut hinted = false;
        loop {
            self.poll();
            let waiting = self.agents.iter().any(|a| !a.flown && !a.fresh);
            if !waiting || self.paused || self.reset_requested {
                return;
            }
            let now = Instant::now();
            if self.commanded && (now - start).as_secs_f64() > limit {
                let late: Vec<String> =
                    self.agents.iter().filter(|a| !a.flown && !a.fresh).map(|a| format!("agent{}", a.id)).collect();
                eprintln!("lockstep: no command from {} within {limit} s; stepping", late.join(", "));
                return;
            }
            if !self.commanded && !hinted && (now - start).as_secs_f64() > 2.0 {
                eprintln!("lockstep: waiting for commands (cmd_vel or action) of every bridged agent");
                hinted = true;
            }
            if now - resent > Duration::from_millis(500) {
                self.resend();
                resent = now;
            }
            std::thread::sleep(Duration::from_micros(200));
        }
    }

    /// The current state and clock again (for controllers that joined after they were sent).
    fn resend(&self) {
        let t = self.time();
        self.publish_state(t);
        send(&self.clock, Clock { clock: Time::from_secs(t) });
    }

    /// Agents without a recent command hold still.
    fn hold_idle(&mut self) {
        let t = self.time();
        for a in &mut self.agents {
            let idle = a.last_command.is_none_or(|c| t - c > self.config.command_timeout + 1e-9);
            if !a.flown && !a.holding && idle {
                let hold = Command::hold(&self.world.agent(a.agent).vehicle);
                self.world.set_command(a.agent, hold);
                a.holding = true;
            }
        }
    }

    fn new_episode(&mut self) {
        self.offset = self.time();
        self.world.reset(None);
        self.episodes += 1;
        self.reset_requested = false;
        for a in &mut self.agents {
            a.last_command = None;
            a.holding = false;
            for s in &mut a.sensors {
                s.last = None;
            }
        }
    }

    pub fn world(&self) -> &WorldInstance {
        &self.world
    }

    /// Simulated time since the bridge started, across episodes (s): the `/clock`.
    pub fn time(&self) -> f64 {
        self.offset + self.world.time()
    }

    pub fn episodes(&self) -> u64 {
        self.episodes
    }

    /// One policy step: commands and service calls are taken (in lockstep, after waiting for
    /// them), policies act, idle agents hold, the world steps (sensors published as their
    /// readings come), then odometry, transforms, joints, events and the clock; an episode
    /// that ended is reset. Paces to wall-clock time when asked. While paused, it only answers
    /// service calls (and returns after a few milliseconds).
    pub fn step(&mut self) {
        if self.steps_total == 0 && self.config.lockstep.is_some() {
            // The state and clock the controllers answer for the first step.
            self.resend();
        }
        match self.config.lockstep {
            Some(limit) if !self.paused => self.wait_for_commands(limit),
            _ => self.poll(),
        }
        if self.paused {
            if self.reset_requested {
                self.new_episode();
                send(&self.clock, Clock { clock: Time::from_secs(self.time()) });
            }
            self.paced_from = None;
            std::thread::sleep(Duration::from_millis(5));
            return;
        }
        for (group, policy) in &mut self.policies {
            let g = &self.world.scenario().groups[*group];
            let (n, od, ad) = (g.spec.count, g.obs_dim(), g.act_dim());
            self.obs.resize(n * od, 0.0);
            self.action.resize(n * ad, 0.0);
            self.world.observe(*group, &mut self.obs);
            for (o, a) in self.obs.chunks_exact(od).zip(self.action.chunks_exact_mut(ad)) {
                policy.act(o, a);
            }
            self.world.set_actions(*group, &self.action);
        }
        self.hold_idle();
        let (agents, offset) = (&mut self.agents, self.offset);
        self.world.step_with(&mut |w| {
            for a in agents.iter_mut() {
                let sensors = &w.agent(a.agent).sensors;
                for s in &mut a.sensors {
                    s.publish(&sensors[s.sensor], offset);
                }
            }
        });
        self.steps_total += 1;
        for a in &mut self.agents {
            a.fresh = false;
        }
        let t = self.time();
        // State before the clock: a controller answering a tick has the state of that tick.
        if self.world.steps().is_multiple_of(self.odom_divider) {
            self.publish_state(t);
        }
        for a in &self.agents {
            let events = self.world.agent(a.agent).events;
            if !events.is_empty() {
                send(&a.events, UInt32 { data: events.0 });
            }
        }
        if self.episode_over() || self.reset_requested {
            self.new_episode();
        }
        send(&self.clock, Clock { clock: Time::from_secs(t) });
        self.pace();
    }

    /// Policy steps taken, across episodes.
    pub fn steps(&self) -> u64 {
        self.steps_total
    }

    fn episode_over(&self) -> bool {
        let timed_out = self.config.episode_time.is_some_and(|e| self.world.time() >= e - 1e-9);
        let all_disabled = !self.agents.is_empty() && self.agents.iter().all(|a| self.world.agent(a.agent).disabled);
        timed_out || all_disabled
    }

    fn publish_state(&self, t: f64) {
        let mut transforms = Vec::with_capacity(self.agents.len());
        for a in &self.agents {
            let agent = self.world.agent(a.agent);
            let v = &agent.vehicle;
            let (p, q) = (v.position(), v.orientation());
            let mut odom = Odometry { header: header(t, "map"), child_frame_id: a.frame.clone(), ..Default::default() };
            odom.pose.pose.position = crate::msgs::geometry_msgs::Point { x: p.x, y: p.y, z: p.z };
            odom.pose.pose.orientation = quaternion(q);
            odom.twist.twist.linear = vector(v.lin_vel_body());
            odom.twist.twist.angular = vector(v.ang_vel_body());
            send(&a.odom, odom);
            transforms.push(TransformStamped {
                header: header(t, "map"),
                child_frame_id: a.frame.clone(),
                transform: Transform { translation: vector(p), rotation: quaternion(q) },
            });
            if let (Some(p), Some(w)) = (&a.joints, v.as_wheeled()) {
                let mut js = JointState { header: header(t, &a.frame), ..Default::default() };
                for (k, wheel) in w.wheels().enumerate() {
                    js.name.push(format!("wheel{k}"));
                    js.position.push(wheel.spin_angle);
                    js.velocity.push(wheel.spin);
                    js.name.push(format!("steer{k}"));
                    js.position.push(wheel.steer);
                    js.velocity.push(0.0);
                }
                send(p, js);
            }
        }
        send(&self.tf, TFMessage { transforms });
    }

    fn pace(&mut self) {
        let Pacing::Realtime(factor) = self.config.pacing else { return };
        let now = Instant::now();
        let (wall0, sim0) = *self.paced_from.get_or_insert((now, self.time()));
        let due = wall0 + Duration::from_secs_f64(((self.time() - sim0) / factor).max(0.0));
        if due > now {
            std::thread::sleep(due - now);
        } else if now - due > Duration::from_secs(1) {
            // Fell behind by more than a second: carry on from here instead of racing.
            self.paced_from = Some((now, self.time()));
        }
    }

    /// Step until the clock reaches `until` (s, bridge time).
    pub fn run_until(&mut self, until: f64) {
        while self.time() < until - 1e-9 {
            self.step();
        }
    }
}
