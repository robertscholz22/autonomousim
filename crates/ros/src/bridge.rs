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

use crate::msgs::builtin_interfaces::Time;
use crate::msgs::geometry_msgs::{Quaternion, Transform, TransformStamped, Vector3};
use crate::msgs::nav_msgs::Odometry;
use crate::msgs::rosgraph_msgs::Clock;
use crate::msgs::sensor_msgs::{FluidPressure, Imu, JointState, MagneticField, NavSatFix, NavSatStatus, Range};
use crate::msgs::std_msgs::{Header, StringMsg, UInt32};
use crate::msgs::tf2_msgs::TFMessage;
use crate::msgs::{Array, RosMessage};
use crate::node::{Publisher, RosNode, qos};
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
}

impl Default for BridgeConfig {
    fn default() -> Self {
        Self { domain_id: 0, pacing: Pacing::Realtime(1.0), odom_hz: 0, episode_time: None, seed: 0 }
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

struct AgentPubs {
    /// Index in the world.
    agent: usize,
    frame: String,
    odom: Publisher<Odometry>,
    joints: Option<Publisher<JointState>>,
    events: Publisher<UInt32>,
    sensors: Vec<SensorPub>,
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
            let mut topics = vec![format!("{ns}/odom"), format!("{ns}/events")];
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
                odom: node.publisher(&format!("{ns}/odom"), qos::RELIABLE)?,
                events: node.publisher(&format!("{ns}/events"), qos::RELIABLE)?,
                joints,
                frame,
                sensors,
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
        self.policies.push((group, policy));
        Ok(())
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

    /// One policy step: policies act, the world steps (sensors published as their readings
    /// come), then the clock, odometry, transforms, joints and events; an episode that ended
    /// is reset. Paces to wall-clock time when asked.
    pub fn step(&mut self) {
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
        let (agents, offset) = (&mut self.agents, self.offset);
        self.world.step_with(&mut |w| {
            for a in agents.iter_mut() {
                let sensors = &w.agent(a.agent).sensors;
                for s in &mut a.sensors {
                    s.publish(&sensors[s.sensor], offset);
                }
            }
        });
        let t = self.time();
        send(&self.clock, Clock { clock: Time::from_secs(t) });
        if self.world.steps().is_multiple_of(self.odom_divider) {
            self.publish_state(t);
        }
        for a in &self.agents {
            let events = self.world.agent(a.agent).events;
            if !events.is_empty() {
                send(&a.events, UInt32 { data: events.0 });
            }
        }
        if self.episode_over() {
            self.offset = t;
            self.world.reset(None);
            self.episodes += 1;
            for s in self.agents.iter_mut().flat_map(|a| &mut a.sensors) {
                s.last = None;
            }
        }
        self.pace();
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
