//! Recording of world instances to MCAP, with JSON messages that open in Foxglove.
//!
//! | Topic | When | Content |
//! |---|---|---|
//! | `/meta` | once | scenario, map pool (metadata and content hashes), vehicle definitions, rates, agent list |
//! | `/episode` | every reset | episode number and seed, map index, environment, spawn poses, goals and routes (lane points of agents with `route` goals, planned paths of `path` goals) |
//! | `/agent/<id>/state` | `state_hz` | time, pose, velocity, rates, wind, goal, events; rotor speeds (multirotors) or steering, wheels, powertrain and the joints of trailers and the rider's lean (ground vehicles); two-wheelers add `steer_torque` and `feet` |
//! | `/agent/<id>/pose` | `state_hz` | the pose as `foxglove.PoseInFrame` (frame `world`) |
//! | `/agent/<id>/action` | each action | the normalised action |
//! | `/agent/<id>/lidar` | each scan, if enabled | sensor pose and ranges |
//! | `/agent/<id>/camera/<sensor>` | frames captured at multiples of `1/camera_hz`, if enabled | the visible RGB image as `foxglove.RawImage` (`rgb8`, base64) |
//! | `/npcs` | `state_hz` (scenarios with scripted groups) | the scripted agents (NPCs) packed into one message instead of their own `state`, `pose` and `action` channels, exactly: time, agent ids, poses, velocities, body rates, steering angles, kinematic or full physics, event bits, disabled flags, trailer joints and wheels (spin, angle, steering, travel) ([`RecordedNpcs`]) |
//! | `/pedestrians` | `pedestrian_hz` and at resets (scenarios with pedestrians) | the crowd packed into one message, rounded to cm, cm/s and mrad: time, then per pedestrian position (x, y, z), velocity (x, y), heading, height and state ([`RecordedPedestrians`]) |
//! | `/route` | when a scripted agent's route changes (scenarios with scripted groups) | time, agent id and the route's lane points |
//! | `/events` | when an agent gets new event bits | agent id and event names |
//! | `/signals` | at resets and whenever a controller's phase or light changes (map pools with traffic signals) | time, controller, junction node, phase and light (`green`, `amber`, `red`) |
//!
//! `/episode` carries the signal offsets of the episode (`signals`, s per controller) on maps
//! with traffic signals; the state follows from them and the time
//! ([`traffic`](crate::traffic)).
//!
//! Vehicle definitions in `/meta` are a multirotor's fields alone (as in the first recordings),
//! or other families' definitions with their `type` tag.
//!
//! The map is not stored: a reader rebuilds it from the scenario's map source. Log times are
//! simulated time since the recording started, continuing across episodes.
//!
//! [`Recording`] reads a file back into typed episodes (states, actions, scans, events) and
//! rebuilds its scenario with the maps checked against their recorded hashes.

use crate::SimError;
use crate::pedestrians::PedState;
use crate::scenario::{CompiledScenario, Goal, Scenario};
use crate::world::{STATE_FIELDS, WorldInstance};
use autonomousim_sensors::Sensor;
use autonomousim_vehicles::{SharedDef, Vehicle, VehicleDef};
use autonomousim_world::{Light, Polyline};
use glam::{DQuat, DVec2, DVec3};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::io::{BufWriter, Seek, Write};
use std::path::Path;
use std::sync::Arc;

/// Destination of recorded messages (an MCAP file now; a live viewer connection later).
pub trait TelemetrySink: Send {
    /// Register a topic with a JSON schema; returns its channel id.
    fn add_channel(&mut self, topic: &str, schema_name: &str, schema: &str) -> Result<u16, SimError>;
    fn write(&mut self, channel: u16, log_time_ns: u64, data: &[u8]) -> Result<(), SimError>;
    fn finish(&mut self) -> Result<(), SimError>;
    /// Whether messages reach anyone now; if not, the recorder writes only `/meta`, `/episode`
    /// and `/route` (a stream without viewers).
    fn live(&self) -> bool {
        true
    }
}

/// Writes an MCAP file (zstd chunks, JSON messages with `jsonschema` schemas).
pub struct McapSink<W: Write + Seek + Send> {
    writer: mcap::Writer<W>,
    sequence: u32,
}

fn mcap_err(e: mcap::McapError) -> SimError {
    SimError::Record(e.to_string())
}

impl McapSink<BufWriter<std::fs::File>> {
    pub fn create(path: impl AsRef<Path>) -> Result<Self, SimError> {
        Self::new(BufWriter::new(std::fs::File::create(path)?))
    }
}

impl<W: Write + Seek + Send> McapSink<W> {
    pub fn new(w: W) -> Result<Self, SimError> {
        let writer = mcap::WriteOptions::new().profile("").library("autonomousim").create(w).map_err(mcap_err)?;
        Ok(Self { writer, sequence: 0 })
    }
}

impl<W: Write + Seek + Send> TelemetrySink for McapSink<W> {
    fn add_channel(&mut self, topic: &str, schema_name: &str, schema: &str) -> Result<u16, SimError> {
        let schema_id = self.writer.add_schema(schema_name, "jsonschema", schema.as_bytes()).map_err(mcap_err)?;
        self.writer.add_channel(schema_id, topic, "json", &BTreeMap::new()).map_err(mcap_err)
    }

    fn write(&mut self, channel: u16, log_time_ns: u64, data: &[u8]) -> Result<(), SimError> {
        self.sequence = self.sequence.wrapping_add(1);
        let header = mcap::records::MessageHeader {
            channel_id: channel,
            sequence: self.sequence,
            log_time: log_time_ns,
            publish_time: log_time_ns,
        };
        self.writer.write_to_known_channel(&header, data).map_err(mcap_err)
    }

    fn finish(&mut self) -> Result<(), SimError> {
        self.writer.finish().map(|_| ()).map_err(mcap_err)
    }
}

/// Keeps messages in memory (tests, and handing recordings to other threads).
#[derive(Clone, Debug, Default)]
pub struct MemorySink {
    /// Topic of each channel id.
    pub topics: Vec<String>,
    /// `(channel, log time ns, JSON)`.
    pub messages: Vec<(u16, u64, Vec<u8>)>,
}

impl MemorySink {
    /// Parsed messages of one topic.
    pub fn topic(&self, topic: &str) -> Vec<(u64, Value)> {
        let Some(id) = self.topics.iter().position(|t| t == topic) else { return Vec::new() };
        self.messages
            .iter()
            .filter(|m| m.0 as usize == id)
            .map(|m| (m.1, serde_json::from_slice(&m.2).expect("valid JSON")))
            .collect()
    }
}

impl TelemetrySink for MemorySink {
    fn add_channel(&mut self, topic: &str, _: &str, _: &str) -> Result<u16, SimError> {
        self.topics.push(topic.into());
        Ok((self.topics.len() - 1) as u16)
    }

    fn write(&mut self, channel: u16, log_time_ns: u64, data: &[u8]) -> Result<(), SimError> {
        self.messages.push((channel, log_time_ns, data.to_vec()));
        Ok(())
    }

    fn finish(&mut self) -> Result<(), SimError> {
        Ok(())
    }
}

/// Shares a [`MemorySink`] so that it can be inspected after the recorder is gone.
impl TelemetrySink for std::sync::Arc<std::sync::Mutex<MemorySink>> {
    fn add_channel(&mut self, topic: &str, schema_name: &str, schema: &str) -> Result<u16, SimError> {
        self.lock().expect("sink lock").add_channel(topic, schema_name, schema)
    }

    fn write(&mut self, channel: u16, log_time_ns: u64, data: &[u8]) -> Result<(), SimError> {
        self.lock().expect("sink lock").write(channel, log_time_ns, data)
    }

    fn finish(&mut self) -> Result<(), SimError> {
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RecorderConfig {
    /// Rate of the state and pose messages (must divide the physics rate).
    pub state_hz: u32,
    /// Record LiDAR scans.
    pub lidar: bool,
    /// Record the camera frames captured at multiples of `1 / camera_hz` (0: none; must
    /// divide the physics rate). Replays re-render images from the recorded states; these are
    /// for other tools (e.g. Foxglove).
    #[serde(skip_serializing_if = "is_zero_hz")]
    pub camera_hz: u32,
    /// Rate of the `/pedestrians` messages (0: none; must divide the physics rate).
    pub pedestrian_hz: u32,
}

fn is_zero_hz(x: &u32) -> bool {
    *x == 0
}

impl Default for RecorderConfig {
    fn default() -> Self {
        Self { state_hz: 50, lidar: false, camera_hz: 0, pedestrian_hz: 5 }
    }
}

const OBJECT_SCHEMA: &str = r#"{"type":"object"}"#;

const RAW_IMAGE_SCHEMA: &str = r#"{"title":"foxglove.RawImage","type":"object","properties":{"timestamp":{"type":"object","properties":{"sec":{"type":"integer","minimum":0},"nsec":{"type":"integer","minimum":0,"maximum":999999999}}},"frame_id":{"type":"string"},"width":{"type":"integer","minimum":0},"height":{"type":"integer","minimum":0},"encoding":{"type":"string"},"step":{"type":"integer","minimum":0},"data":{"type":"string","contentEncoding":"base64"}}}"#;

/// A camera frame channel: the sensor, its channel and the tick of the last frame written.
#[derive(Clone, Copy, Debug)]
struct CameraChannel {
    sensor: usize,
    channel: u16,
    last: Option<u64>,
}

const POSE_IN_FRAME_SCHEMA: &str = r#"{"title":"foxglove.PoseInFrame","type":"object","properties":{"timestamp":{"type":"object","properties":{"sec":{"type":"integer","minimum":0},"nsec":{"type":"integer","minimum":0,"maximum":999999999}}},"frame_id":{"type":"string"},"pose":{"type":"object","properties":{"position":{"type":"object","properties":{"x":{"type":"number"},"y":{"type":"number"},"z":{"type":"number"}}},"orientation":{"type":"object","properties":{"x":{"type":"number"},"y":{"type":"number"},"z":{"type":"number"},"w":{"type":"number"}}}}}}}"#;

#[derive(Clone, Copy, Debug)]
struct AgentChannels {
    /// `None` for scripted agents (NPCs; in `/npcs`).
    state: Option<u16>,
    pose: Option<u16>,
    action: Option<u16>,
    lidar: Option<u16>,
}

/// Records one [`WorldInstance`]: call [`on_reset`](Self::on_reset) after every reset,
/// [`on_actions`](Self::on_actions) after setting actions and [`on_tick`](Self::on_tick)
/// after every physics tick (e.g. through [`WorldInstance::step_with`]).
///
/// Errors do not interrupt the simulation: the first one is kept and returned by
/// [`finish`](Self::finish).
pub struct Recorder {
    sink: Box<dyn TelemetrySink>,
    config: RecorderConfig,
    divider: u64,
    physics_hz: u64,
    /// Physics ticks since the recording started.
    ticks: u64,
    episodes: u64,
    meta: Option<(u16, u16, u16)>,
    agents: Vec<AgentChannels>,
    /// Per agent (empty unless `camera_hz` is set).
    cameras: Vec<Vec<CameraChannel>>,
    camera_divider: u64,
    /// The `/route` channel (scenarios with scripted groups) and the route last written per
    /// agent.
    route: Option<u16>,
    routes: Vec<Option<Arc<Polyline>>>,
    /// The `/signals` channel (map pools with signals) and the (phase, light) last written
    /// per controller.
    signals: Option<u16>,
    lights: Vec<Option<(usize, Light)>>,
    /// The `/npcs` channel (scenarios with scripted groups).
    npcs: Option<u16>,
    /// The `/pedestrians` channel (scenarios with pedestrians) and its rate divider.
    pedestrians: Option<u16>,
    pedestrian_divider: u64,
    seen: Vec<u32>,
    error: Option<SimError>,
}

impl Recorder {
    pub fn new(sink: Box<dyn TelemetrySink>, config: RecorderConfig) -> Self {
        Self {
            sink,
            config,
            divider: 1,
            physics_hz: 1,
            ticks: 0,
            episodes: 0,
            meta: None,
            agents: Vec::new(),
            cameras: Vec::new(),
            camera_divider: 1,
            route: None,
            routes: Vec::new(),
            signals: None,
            lights: Vec::new(),
            npcs: None,
            pedestrians: None,
            pedestrian_divider: 1,
            seen: Vec::new(),
            error: None,
        }
    }

    /// Record to an MCAP file.
    pub fn create(path: impl AsRef<Path>, config: RecorderConfig) -> Result<Self, SimError> {
        Ok(Self::new(Box::new(McapSink::create(path)?), config))
    }

    /// Serve the recording live to viewers on `addr` (`autonomousim-viewer attach`; port 0:
    /// any free port); returns the address they connect to.
    pub fn stream(
        addr: impl std::net::ToSocketAddrs,
        config: RecorderConfig,
    ) -> Result<(Self, std::net::SocketAddr), SimError> {
        let sink = crate::stream::StreamSink::bind(addr)?;
        let addr = sink.local_addr();
        Ok((Self::new(Box::new(sink), config), addr))
    }

    fn keep<T>(&mut self, r: Result<T, SimError>) -> Option<T> {
        match r {
            Ok(x) => Some(x),
            Err(e) => {
                self.error.get_or_insert(e);
                None
            }
        }
    }

    fn time_ns(&self) -> u64 {
        (u128::from(self.ticks) * 1_000_000_000 / u128::from(self.physics_hz)) as u64
    }

    fn send(&mut self, channel: u16, msg: &Value) {
        let t = self.time_ns();
        let data = serde_json::to_vec(msg).expect("JSON");
        let r = self.sink.write(channel, t, &data);
        self.keep(r);
    }

    /// Write `/meta` and register the channels (first call only).
    fn start(&mut self, w: &WorldInstance) -> Result<(u16, u16, u16), SimError> {
        let sc = w.scenario();
        self.physics_hz = u64::from(sc.spec.physics_hz);
        self.divider = u64::from(sc.clock.divider("recorder state", self.config.state_hz)?);
        let meta = self.sink.add_channel("/meta", "autonomousim.Meta", OBJECT_SCHEMA)?;
        let episode = self.sink.add_channel("/episode", "autonomousim.Episode", OBJECT_SCHEMA)?;
        let events = self.sink.add_channel("/events", "autonomousim.Events", OBJECT_SCHEMA)?;
        if sc.groups.iter().any(|g| g.scripted()) {
            self.route = Some(self.sink.add_channel("/route", "autonomousim.Route", OBJECT_SCHEMA)?);
            self.npcs = Some(self.sink.add_channel("/npcs", "autonomousim.Npcs", OBJECT_SCHEMA)?);
        }
        if sc.maps.iter().any(|m| m.roads().has_sections() && !m.roads().lanes().controllers().is_empty()) {
            self.signals = Some(self.sink.add_channel("/signals", "autonomousim.Signals", OBJECT_SCHEMA)?);
        }
        if sc.spec.pedestrians.count > 0 && self.config.pedestrian_hz > 0 {
            self.pedestrian_divider = u64::from(sc.clock.divider("recorder pedestrians", self.config.pedestrian_hz)?);
            self.pedestrians =
                Some(self.sink.add_channel("/pedestrians", "autonomousim.Pedestrians", OBJECT_SCHEMA)?);
        }
        if self.config.camera_hz > 0 {
            self.camera_divider = u64::from(sc.clock.divider("recorder camera", self.config.camera_hz)?);
        }
        for a in w.agents() {
            let p = format!("/agent/{}", a.id);
            let mut cameras = Vec::new();
            for (k, s) in a.sensors.iter().enumerate() {
                if matches!(s, Sensor::Camera(_)) && self.config.camera_hz > 0 {
                    let name = &sc.groups[a.group].spec.sensors[k].name;
                    let channel =
                        self.sink.add_channel(&format!("{p}/camera/{name}"), "foxglove.RawImage", RAW_IMAGE_SCHEMA)?;
                    cameras.push(CameraChannel { sensor: k, channel, last: None });
                }
            }
            self.cameras.push(cameras);
            let lidar = a.sensors.iter().any(|s| matches!(s, Sensor::Lidar(_))) && self.config.lidar;
            let own = !sc.groups[a.group].scripted();
            let mut channel = |topic: String, schema_name: &str, schema: &str| -> Result<Option<u16>, SimError> {
                if own { self.sink.add_channel(&topic, schema_name, schema).map(Some) } else { Ok(None) }
            };
            self.agents.push(AgentChannels {
                state: channel(format!("{p}/state"), "autonomousim.AgentState", OBJECT_SCHEMA)?,
                pose: channel(format!("{p}/pose"), "foxglove.PoseInFrame", POSE_IN_FRAME_SCHEMA)?,
                action: channel(format!("{p}/action"), "autonomousim.Action", OBJECT_SCHEMA)?,
                lidar: if lidar {
                    Some(self.sink.add_channel(&format!("{p}/lidar"), "autonomousim.LidarScan", OBJECT_SCHEMA)?)
                } else {
                    None
                },
            });
        }
        self.seen = vec![0; w.agents().len()];
        let agents: Vec<Value> = w
            .agents()
            .iter()
            .map(|a| {
                let g = &sc.groups[a.group];
                json!({"id": a.id, "group": g.spec.name, "index": a.id as usize - g.first_agent, "vehicle": g.def.name()})
            })
            .collect();
        let vehicles: BTreeMap<&str, Value> = sc
            .groups
            .iter()
            .map(|g| {
                let def = match &g.def {
                    SharedDef::Multirotor(m) => serde_json::to_value(&**m),
                    SharedDef::Wheeled(w) => serde_json::to_value(VehicleDef::Wheeled((**w).clone())),
                    SharedDef::FixedWing(f) => serde_json::to_value(VehicleDef::FixedWing((**f).clone())),
                    SharedDef::Helicopter(h) => serde_json::to_value(VehicleDef::Helicopter((**h).clone())),
                    SharedDef::Tiltrotor(t) => serde_json::to_value(VehicleDef::Tiltrotor((**t).clone())),
                };
                (g.spec.name.as_str(), def.expect("JSON"))
            })
            .collect();
        let maps: Vec<Value> = sc
            .maps
            .iter()
            .zip(&sc.map_hashes)
            .map(|(m, h)| {
                json!({
                    "name": m.meta.name,
                    "generator": m.meta.generator,
                    "generator_version": m.meta.generator_version,
                    "seed": m.meta.seed,
                    "geo_origin": m.meta.geo_origin,
                    "hash": h,
                })
            })
            .collect();
        let mut msg = json!({
            "format": 1,
            "scenario": sc.spec,
            "maps": maps,
            "physics_hz": sc.spec.physics_hz,
            "policy_hz": sc.spec.policy_hz,
            "state_hz": self.config.state_hz,
            "state_fields": STATE_FIELDS.iter().map(|(n, d)| json!([n, d])).collect::<Vec<_>>(),
            "agents": agents,
            "vehicles": vehicles,
        });
        if self.pedestrians.is_some() {
            msg["pedestrian_hz"] = json!(self.config.pedestrian_hz);
        }
        let data = serde_json::to_vec(&msg).expect("JSON");
        self.sink.write(meta, self.time_ns(), &data)?;
        Ok((meta, episode, events))
    }

    /// After a reset: `/meta` on the first call, then `/episode` and the initial state.
    pub fn on_reset(&mut self, w: &WorldInstance) {
        if self.meta.is_none() {
            let r = self.start(w);
            self.meta = self.keep(r);
        }
        let Some((_, episode, _)) = self.meta else { return };
        let seed: String = w.episode_seed().0.iter().map(|b| format!("{b:02x}")).collect();
        let agents: Vec<Value> = w
            .agents()
            .iter()
            .map(|a| {
                let mut m = json!({
                    "id": a.id,
                    "spawn": {"position": a.spawn.pos, "orientation": a.spawn.rot},
                    "goals": a.goals,
                });
                if !a.legs.is_empty() {
                    // The whole planned path (each leg starts where the previous one ends).
                    let path: Vec<_> = a
                        .legs
                        .iter()
                        .enumerate()
                        .flat_map(|(k, l)| l.points()[usize::from(k > 0)..].iter().copied())
                        .collect();
                    m["route"] = json!(path);
                } else if let Some(route) = &a.route {
                    m["route"] = json!(route.points());
                }
                m
            })
            .collect();
        let mut msg = json!({
            "episode": self.episodes,
            "seed": seed,
            "map": w.map_index(),
            "environment": w.env().config,
            "agents": agents,
        });
        if !w.signals().is_empty() {
            msg["signals"] = json!(w.signals().offsets());
        }
        self.episodes += 1;
        self.send(episode, &msg);
        self.routes = w.agents().iter().map(|a| a.route.clone()).collect();
        self.cameras.iter_mut().flatten().for_each(|c| c.last = None);
        self.seen.fill(0);
        self.lights.clear();
        self.write_signals(w);
        self.write_states(w);
        self.write_pedestrians(w);
    }

    /// `/signals` messages for the controllers whose phase or light changed.
    fn write_signals(&mut self, w: &WorldInstance) {
        let Some(ch) = self.signals else { return };
        if w.signals().is_empty() {
            return;
        }
        let lanes = w.map().roads().lanes();
        let time = w.time();
        self.lights.resize(lanes.controllers().len(), None);
        for (k, c) in lanes.controllers().iter().enumerate() {
            let s = w.signals().state(lanes, k, time);
            if self.lights[k] != Some(s) {
                self.lights[k] = Some(s);
                let light = match s.1 {
                    Light::Green => "green",
                    Light::Amber => "amber",
                    Light::Red => "red",
                };
                let msg = json!({"time": time, "controller": k, "junction": c.junction, "phase": s.0, "light": light});
                self.send(ch, &msg);
            }
        }
    }

    /// After new actions were set.
    pub fn on_actions(&mut self, w: &WorldInstance) {
        if self.meta.is_none() || !self.sink.live() {
            return;
        }
        let time = w.time();
        for a in w.agents() {
            if let Some(ch) = self.agents[a.id as usize].action {
                self.send(ch, &json!({"time": time, "action": a.action.as_slice()}));
            }
        }
    }

    /// After every physics tick.
    pub fn on_tick(&mut self, w: &WorldInstance) {
        let Some((_, _, events)) = self.meta else { return };
        self.ticks += 1;
        let time = w.time();
        let live = self.sink.live();
        for a in w.agents() {
            let i = a.id as usize;
            let bits = if live { a.events.0 } else { 0 };
            if bits & self.seen[i] != self.seen[i] {
                // A new policy step cleared the bits.
                self.seen[i] = 0;
            }
            if bits & !self.seen[i] != 0 {
                self.seen[i] |= bits;
                let names: Vec<&str> = a.events.names().collect();
                self.send(events, &json!({"time": time, "agent": a.id, "events": names}));
            }
            if let Some(ch) = self.agents[i].lidar
                && live
                && let Some(scan) = a.sensors.iter().find_map(|s| match s {
                    Sensor::Lidar(l) => l.latest().filter(|s| s.tick == w.clock().tick),
                    _ => None,
                })
            {
                let ranges: Vec<Value> =
                    scan.ranges.iter().map(|r| if r.is_finite() { json!(r) } else { Value::Null }).collect();
                let msg = json!({"time": scan.time, "position": scan.pose.pos, "orientation": scan.pose.rot, "ranges": ranges});
                self.send(ch, &msg);
            }
            if let Some(ch) = self.route
                && let Some(route) = &a.route
                && !self.routes[i].as_ref().is_some_and(|r| Arc::ptr_eq(r, route))
            {
                self.routes[i] = Some(route.clone());
                self.send(ch, &json!({"time": time, "agent": a.id, "route": route.points()}));
            }
        }
        if !live {
            return;
        }
        self.write_signals(w);
        self.on_frames(w);
        if w.clock().tick.is_multiple_of(self.divider) {
            self.write_states(w);
        }
        if w.clock().tick.is_multiple_of(self.pedestrian_divider) {
            self.write_pedestrians(w);
        }
    }

    /// After camera frames were delivered ([`WorldInstance::deliver`]) outside a tick: records
    /// those that became visible (also done by [`on_tick`](Self::on_tick)).
    pub fn on_frames(&mut self, w: &WorldInstance) {
        if self.meta.is_none() {
            return;
        }
        for a in w.agents() {
            let i = a.id as usize;
            for k in 0..self.cameras[i].len() {
                let c = self.cameras[i][k];
                let Sensor::Camera(cam) = &a.sensors[c.sensor] else { continue };
                let Some(frame) = cam.latest() else { continue };
                if c.last == Some(frame.tick) || !frame.tick.is_multiple_of(self.camera_divider) {
                    continue;
                }
                self.cameras[i][k].last = Some(frame.tick);
                // Stamped with the capture time (logged when it became visible).
                let age = u128::from(w.clock().tick.saturating_sub(frame.tick)) * 1_000_000_000;
                let t_ns = self.time_ns().saturating_sub((age / u128::from(self.physics_hz)) as u64);
                let image = &frame.value;
                let msg = json!({
                    "timestamp": {"sec": t_ns / 1_000_000_000, "nsec": t_ns % 1_000_000_000},
                    "frame_id": w.scenario().groups[a.group].spec.sensors[c.sensor].name,
                    "width": image.width,
                    "height": image.height,
                    "encoding": "rgb8",
                    "step": 3 * image.width,
                    "data": base64(&image.rgb),
                });
                self.send(c.channel, &msg);
            }
        }
    }

    fn write_states(&mut self, w: &WorldInstance) {
        let time = w.time();
        let t_ns = self.time_ns();
        let stamp = json!({"sec": t_ns / 1_000_000_000, "nsec": t_ns % 1_000_000_000});
        self.write_npcs(w);
        for a in w.agents() {
            let ch = self.agents[a.id as usize];
            let (Some(state_ch), Some(pose_ch)) = (ch.state, ch.pose) else { continue };
            let v = &a.vehicle;
            let (p, q) = (v.position(), v.orientation());
            let goal = a.goal();
            let mut msg = json!({
                "time": time,
                "tick": w.clock().tick,
                "position": p,
                "orientation": q,
                "velocity": q * v.lin_vel_body(),
                "rates": v.ang_vel_body(),
                "wind": a.air().wind,
                "goal": goal.position,
                "goal_yaw": goal.yaw,
                "events": a.events.0,
                "disabled": a.disabled,
            });
            let m = msg.as_object_mut().expect("object");
            match v {
                Vehicle::Multirotor(v) => {
                    m.insert("motors".into(), json!(v.motor_speeds()));
                }
                Vehicle::Wheeled(v) => {
                    let wheels: Vec<RecordedWheel> = v
                        .wheels()
                        .map(|w| RecordedWheel {
                            spin: w.spin,
                            spin_angle: w.spin_angle,
                            steer: w.steer,
                            travel: w.travel,
                            drive_torque: w.drive_torque,
                            brake_torque: w.brake_torque,
                            load: w.tire.fz,
                            kappa: w.tire.kappa,
                            tan_alpha: w.tire.tan_alpha,
                            fx: w.tire.fx,
                            fy: w.tire.fy,
                            sinkage: w.tire.sinkage,
                        })
                        .collect();
                    let pt = v.powertrain();
                    m.insert("steering".into(), json!(v.steering_angle()));
                    m.insert("wheels".into(), json!(wheels));
                    m.insert("gear".into(), json!(pt.gear));
                    m.insert("engine_speed".into(), json!(pt.engine_speed));
                    let joints = v.joints();
                    if !joints.is_empty() {
                        m.insert("joints".into(), json!(joints));
                    }
                    if let Some((_, w)) = v.def().steering_head() {
                        m.insert("steer_torque".into(), json!(v.wheel(w).steer_torque));
                    }
                    if v.def().feet.is_some() {
                        m.insert("feet".into(), json!(v.feet_down()));
                    }
                }
                Vehicle::FixedWing(v) => {
                    let flow = v.flow();
                    let loads: Vec<f64> = v.wheels().iter().map(|w| w.map_or(0.0, |c| c.normal_force)).collect();
                    m.insert("surfaces".into(), json!(v.surfaces()));
                    m.insert("throttle".into(), json!(v.input().throttle));
                    m.insert("rotor_speed".into(), json!(v.rotor_speed()));
                    m.insert("airspeed".into(), json!(flow.airspeed));
                    m.insert("alpha".into(), json!(flow.alpha));
                    m.insert("beta".into(), json!(flow.beta));
                    m.insert("gear_loads".into(), json!(loads));
                }
                Vehicle::Helicopter(v) => {
                    let (main, tail) = (v.main_rotor_state(), v.tail_rotor_state());
                    m.insert("controls".into(), json!(v.input().to_array()));
                    m.insert("pitches".into(), json!(v.pitches()));
                    m.insert("rotor_speed".into(), json!(v.rotor_speed()));
                    m.insert("engine_power".into(), json!(v.engine_power()));
                    m.insert("flap".into(), json!([main.flap, tail.flap]));
                    m.insert("coning".into(), json!([v.loads().main.coning, v.loads().tail.coning]));
                    m.insert("airspeed".into(), json!(v.flow().airspeed));
                }
                Vehicle::Tiltrotor(v) => {
                    let n = v.rotor_count();
                    let [a, e, r] = v.channels();
                    let flow = v.flow();
                    m.insert("motors".into(), json!(v.rotor_speeds()));
                    m.insert("throttles".into(), json!(&v.input().throttle[..n]));
                    m.insert("tilts".into(), json!(v.tilts()));
                    m.insert("surfaces".into(), json!([a, e, r, 0.0]));
                    m.insert("engine_power".into(), json!(v.electric_power()));
                    m.insert("airspeed".into(), json!(flow.airspeed));
                    m.insert("alpha".into(), json!(flow.alpha));
                    m.insert("beta".into(), json!(flow.beta));
                }
            }
            self.send(state_ch, &msg);
            let pose = json!({
                "timestamp": stamp,
                "frame_id": "world",
                "pose": {"position": xyz(p), "orientation": {"x": q.x, "y": q.y, "z": q.z, "w": q.w}},
            });
            self.send(pose_ch, &pose);
        }
    }

    /// One `/npcs` message with the state of every scripted agent (see [`RecordedNpcs`]).
    fn write_npcs(&mut self, w: &WorldInstance) {
        let Some(ch) = self.npcs else { return };
        let mut m = RecordedNpcs { time: w.time(), tick: w.clock().tick, ..Default::default() };
        for a in w.agents().iter().filter(|a| self.agents[a.id as usize].state.is_none()) {
            let v = &a.vehicle;
            let q = v.orientation();
            m.agents.push(a.id);
            m.pose.extend(v.position().to_array());
            m.pose.extend(q.to_array());
            m.velocity.extend((q * v.lin_vel_body()).to_array());
            m.rates.extend(v.ang_vel_body().to_array());
            let wheeled = v.as_wheeled();
            m.steering.push(wheeled.map_or(0.0, |v| v.steering_angle()));
            m.kinematic.push(a.is_kinematic());
            m.events.push(a.events.0);
            m.disabled.push(a.disabled);
            m.joints.push(wheeled.map_or(Vec::new(), |v| v.joints().to_vec()));
            m.wheels.push(
                wheeled.map_or(Vec::new(), |v| {
                    v.wheels().flat_map(|w| [w.spin, w.spin_angle, w.steer, w.travel]).collect()
                }),
            );
        }
        if m.joints.iter().all(Vec::is_empty) {
            m.joints.clear();
        }
        if m.wheels.iter().all(Vec::is_empty) {
            m.wheels.clear();
        }
        self.send(ch, &serde_json::to_value(&m).expect("JSON"));
    }

    /// One `/pedestrians` message with the crowd (see [`RecordedPedestrians`]).
    fn write_pedestrians(&mut self, w: &WorldInstance) {
        let Some(ch) = self.pedestrians else { return };
        let peds = &w.crowd().peds;
        let cm = |x: f64| (x * 100.0).round() as i32;
        let mut m = RecordedPedestrians { time: w.time(), ..Default::default() };
        for p in peds {
            m.position.extend([cm(p.pos.x), cm(p.pos.y), cm(p.z)]);
            m.velocity.extend([cm(p.vel.x), cm(p.vel.y)]);
            m.heading.push((p.heading * 1000.0).round() as i32);
            m.height.push(cm(p.height));
            m.state.push(state_code(p.state));
        }
        self.send(ch, &serde_json::to_value(&m).expect("JSON"));
    }

    /// Finish the file; returns the first error of the recording, if any.
    pub fn finish(mut self) -> Result<(), SimError> {
        let r = self.sink.finish();
        match self.error.take() {
            Some(e) => Err(e),
            None => r,
        }
    }
}

fn xyz(v: DVec3) -> Value {
    json!({"x": v.x, "y": v.y, "z": v.z})
}

/// Standard base64 with padding.
fn base64(data: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b = [chunk[0], chunk.get(1).copied().unwrap_or(0), chunk.get(2).copied().unwrap_or(0)];
        let n = u32::from(b[0]) << 16 | u32::from(b[1]) << 8 | u32::from(b[2]);
        for k in 0..4 {
            if k <= chunk.len() {
                out.push(char::from(ALPHABET[(n >> (18 - 6 * k) & 63) as usize]));
            } else {
                out.push('=');
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------- reading

/// A `/agent/<id>/state` message (or an agent's part of an `/npcs` one).
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
pub struct RecordedState {
    /// Time since the episode started (s).
    pub time: f64,
    pub tick: u64,
    pub position: DVec3,
    pub orientation: DQuat,
    /// World frame (m/s).
    pub velocity: DVec3,
    /// Body frame (rad/s).
    pub rates: DVec3,
    /// Rotor speeds (rad/s); empty for other families.
    #[serde(default)]
    pub motors: Vec<f64>,
    /// Ground vehicles: bicycle steering angle (rad), wheels, gear and engine (or first motor)
    /// speed (rad/s).
    #[serde(default)]
    pub steering: f64,
    #[serde(default)]
    pub wheels: Vec<RecordedWheel>,
    #[serde(default)]
    pub gear: i32,
    #[serde(default)]
    pub engine_speed: f64,
    /// Ground vehicles with trailers or a rider: the units' joint coordinates, then the
    /// rider's lean (`Wheeled::joints`).
    #[serde(default)]
    pub joints: Vec<f64>,
    /// Two-wheelers: the torque on the steering head, the rider's plus the damper's (N·m,
    /// positive left).
    #[serde(default)]
    pub steer_torque: f64,
    /// Two-wheelers: whether the feet are down.
    #[serde(default)]
    pub feet: bool,
    /// Fixed-wing aircraft: aileron, elevator, rudder and flap deflections (rad), throttle,
    /// propeller speed (rad/s), airspeed (m/s), angle of attack and sideslip (rad), and each
    /// landing gear's load (N).
    #[serde(default)]
    pub surfaces: [f64; 4],
    #[serde(default)]
    pub throttle: f64,
    #[serde(default)]
    pub rotor_speed: f64,
    #[serde(default)]
    pub airspeed: f64,
    #[serde(default)]
    pub alpha: f64,
    #[serde(default)]
    pub beta: f64,
    #[serde(default)]
    pub gear_loads: Vec<f64>,
    /// Helicopters (also `rotor_speed` and `airspeed`): pilot inputs (collective,
    /// longitudinal, lateral, pedal), blade pitches (collective, cyclic θ₁c and θ₁s, tail
    /// collective; rad), engine power (W), main and tail rotor flapping `[β₁c, β₁s]` and coning
    /// (rad).
    #[serde(default)]
    pub controls: [f64; 4],
    #[serde(default)]
    pub pitches: [f64; 4],
    #[serde(default)]
    pub engine_power: f64,
    #[serde(default)]
    pub flap: [[f64; 2]; 2],
    #[serde(default)]
    pub coning: [f64; 2],
    /// Tiltrotors (also `motors`, `surfaces` without the flap, `engine_power` drawn by the
    /// motors, `airspeed`, `alpha` and `beta`): throttles and mount tilts (rad).
    #[serde(default)]
    pub throttles: Vec<f64>,
    #[serde(default)]
    pub tilts: Vec<f64>,
    pub wind: DVec3,
    pub goal: DVec3,
    pub goal_yaw: f64,
    /// Event bits since the last policy step.
    pub events: u32,
    pub disabled: bool,
    /// Scripted agents (`/npcs`): whether it moves kinematically.
    #[serde(default)]
    pub kinematic: bool,
}

/// An `/npcs` message: the scripted agents' states, packed and exact (`pose` holds x, y, z,
/// then the quaternion x, y, z, w per agent, `velocity` the world velocity and `rates` the
/// body rates; `wheels` holds spin rate, spin angle, steering angle and travel per wheel;
/// `joints` and `wheels` are empty when no agent has any).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RecordedNpcs {
    pub time: f64,
    pub tick: u64,
    pub agents: Vec<u32>,
    pub pose: Vec<f64>,
    pub velocity: Vec<f64>,
    pub rates: Vec<f64>,
    /// Bicycle steering angle (rad; 0 but for ground vehicles).
    pub steering: Vec<f64>,
    /// Whether it moves kinematically (else in full physics).
    pub kinematic: Vec<bool>,
    pub events: Vec<u32>,
    pub disabled: Vec<bool>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub joints: Vec<Vec<f64>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub wheels: Vec<Vec<f64>>,
}

impl RecordedNpcs {
    /// Each agent's state (what a `/agent/<id>/state` message would hold of it for its pose,
    /// motion and visuals; tyre forces, powertrain and goals 0), with its id.
    pub fn states(&self) -> impl Iterator<Item = (usize, RecordedState)> + '_ {
        self.agents.iter().enumerate().map(|(k, &id)| {
            let p = &self.pose[7 * k..7 * k + 7];
            let v = |x: &[f64]| DVec3::new(x[3 * k], x[3 * k + 1], x[3 * k + 2]);
            let wheels = self.wheels.get(k).map_or(&[][..], |w| &w[..]);
            let state = RecordedState {
                time: self.time,
                tick: self.tick,
                position: DVec3::new(p[0], p[1], p[2]),
                orientation: DQuat::from_xyzw(p[3], p[4], p[5], p[6]),
                velocity: v(&self.velocity),
                rates: v(&self.rates),
                steering: self.steering[k],
                joints: self.joints.get(k).cloned().unwrap_or_default(),
                wheels: wheels
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|w| RecordedWheel {
                        spin: w[0],
                        spin_angle: w[1],
                        steer: w[2],
                        travel: w[3],
                        ..Default::default()
                    })
                    .collect(),
                events: self.events[k],
                disabled: self.disabled[k],
                kinematic: self.kinematic[k],
                ..Default::default()
            };
            (id as usize, state)
        })
    }
}

/// A `/pedestrians` message: the crowd, packed and rounded (`position` holds x, y, z in cm
/// per pedestrian, `velocity` x, y in cm/s, `heading` mrad, `height` cm; `state` is 0
/// walking, 1 waiting, 2 crossing, 3 dwelling, 4 hit).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RecordedPedestrians {
    pub time: f64,
    pub position: Vec<i32>,
    pub velocity: Vec<i32>,
    pub heading: Vec<i32>,
    pub height: Vec<i32>,
    pub state: Vec<u8>,
}

/// A pedestrian of a `/pedestrians` message.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RecordedPedestrian {
    pub position: DVec3,
    pub velocity: DVec2,
    pub heading: f64,
    pub height: f64,
    pub state: PedState,
}

impl RecordedPedestrians {
    pub fn len(&self) -> usize {
        self.state.len()
    }

    pub fn is_empty(&self) -> bool {
        self.state.is_empty()
    }

    /// Pedestrian `i`, if there is one.
    pub fn get(&self, i: usize) -> Option<RecordedPedestrian> {
        let m = |x: i32| f64::from(x) * 0.01;
        let p = self.position.get(3 * i..3 * i + 3)?;
        let v = self.velocity.get(2 * i..2 * i + 2)?;
        Some(RecordedPedestrian {
            position: DVec3::new(m(p[0]), m(p[1]), m(p[2])),
            velocity: DVec2::new(m(v[0]), m(v[1])),
            heading: f64::from(*self.heading.get(i)?) * 1e-3,
            height: m(*self.height.get(i)?),
            state: state_from_code(*self.state.get(i)?),
        })
    }
}

fn state_code(s: PedState) -> u8 {
    match s {
        PedState::Walking => 0,
        PedState::Waiting => 1,
        PedState::Crossing => 2,
        PedState::Dwelling => 3,
        PedState::Hit => 4,
    }
}

fn state_from_code(c: u8) -> PedState {
    match c {
        1 => PedState::Waiting,
        2 => PedState::Crossing,
        3 => PedState::Dwelling,
        4 => PedState::Hit,
        _ => PedState::Walking,
    }
}

/// A wheel in a ground vehicle's state message.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RecordedWheel {
    /// Spin rate (rad/s) and angle (rad), steering angle (rad) and suspension travel (m, bump
    /// positive).
    pub spin: f64,
    pub spin_angle: f64,
    pub steer: f64,
    pub travel: f64,
    /// Drive and brake torque (N·m) and tyre load (N).
    pub drive_torque: f64,
    pub brake_torque: f64,
    pub load: f64,
    /// Tyre slip (longitudinal κ, tan of the slip angle) and the longitudinal and lateral
    /// forces in the contact frame (N).
    pub kappa: f64,
    pub tan_alpha: f64,
    pub fx: f64,
    pub fy: f64,
    /// Sinkage of a track patch into soft soil (m; left out when 0).
    #[serde(skip_serializing_if = "is_zero")]
    pub sinkage: f64,
}

fn is_zero(x: &f64) -> bool {
    *x == 0.0
}

/// A `/agent/<id>/action` message: the normalised action held from `time` on.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct RecordedAction {
    pub time: f64,
    pub action: Vec<f64>,
}

/// A `/agent/<id>/lidar` message.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct RecordedScan {
    pub time: f64,
    /// Sensor pose in the world.
    pub position: DVec3,
    pub orientation: DQuat,
    /// Range of each beam (m); `None` without a return.
    pub ranges: Vec<Option<f64>>,
}

/// An `/events` message.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct RecordedEvents {
    pub time: f64,
    pub agent: usize,
    pub events: Vec<String>,
}

/// An agent of a recording (from `/meta`).
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct RecordedAgent {
    pub id: usize,
    pub group: String,
    /// Index within the group.
    pub index: usize,
    pub vehicle: String,
}

/// One episode of a recording; the per-agent lists are indexed by agent id.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RecordedEpisode {
    /// Episode number within the recording.
    pub number: u64,
    /// Episode seed (hex).
    pub seed: String,
    /// Map of the pool.
    pub map: usize,
    /// Start as time since the recording started (s).
    pub start: f64,
    pub goals: Vec<Vec<Goal>>,
    /// The lane each agent follows to its goals, if it has `route` goals, or a scripted
    /// agent's route at the start.
    pub routes: Vec<Option<Arc<Polyline>>>,
    /// Later routes of scripted agents: (time, route) in time order.
    pub route_updates: Vec<Vec<(f64, Arc<Polyline>)>>,
    pub states: Vec<Vec<RecordedState>>,
    pub actions: Vec<Vec<RecordedAction>>,
    pub scans: Vec<Vec<RecordedScan>>,
    pub events: Vec<RecordedEvents>,
    /// Offsets (s) of the map's signal controllers (empty without signals).
    pub signal_offsets: Vec<f64>,
    /// The crowd in time order (empty without pedestrians).
    pub pedestrians: Vec<RecordedPedestrians>,
}

impl RecordedEpisode {
    /// Time of the last recorded state (s).
    pub fn duration(&self) -> f64 {
        self.states.iter().filter_map(|s| s.last()).map(|s| s.time).fold(0.0, f64::max)
    }

    /// The route of `agent` at `time` (s since the episode started).
    pub fn route_at(&self, agent: usize, time: f64) -> Option<&Arc<Polyline>> {
        let updates = self.route_updates.get(agent).map_or(&[][..], |u| &u[..]);
        let k = updates.partition_point(|(t, _)| *t <= time);
        if k > 0 { Some(&updates[k - 1].1) } else { self.routes.get(agent)?.as_ref() }
    }
}

/// A recording read back from MCAP, for replays and analysis.
#[derive(Clone, Debug)]
pub struct Recording {
    /// The recorded scenario, with every default filled in.
    pub scenario: Scenario,
    /// Content hash of each map of the pool (hex).
    pub map_hashes: Vec<String>,
    pub physics_hz: u32,
    pub policy_hz: u32,
    pub state_hz: u32,
    /// Rate of the `/pedestrians` messages (0: none recorded).
    pub pedestrian_hz: u32,
    pub agents: Vec<RecordedAgent>,
    pub episodes: Vec<RecordedEpisode>,
    /// The `/meta` message as written.
    pub meta: Value,
}

fn record_err(what: impl std::fmt::Display) -> SimError {
    SimError::Record(what.to_string())
}

fn parse<T: serde::de::DeserializeOwned>(topic: &str, data: &[u8]) -> Result<T, SimError> {
    serde_json::from_slice(data).map_err(|e| record_err(format!("{topic}: {e}")))
}

impl Recording {
    pub fn read(path: impl AsRef<Path>) -> Result<Self, SimError> {
        Self::from_bytes(&std::fs::read(path)?)
    }

    /// Parse an MCAP file written by [`Recorder`]; messages are taken in file order.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, SimError> {
        let mut rec: Option<Recording> = None;
        for m in mcap::MessageStream::new(bytes).map_err(mcap_err)? {
            let m = m.map_err(mcap_err)?;
            let topic = m.channel.topic.as_str();
            match &mut rec {
                _ if topic == "/meta" => rec = Some(Self::from_meta(&m.data)?),
                Some(r) => r.push(topic, m.log_time, &m.data)?,
                None => return Err(record_err(format!("{topic} before /meta"))),
            }
        }
        rec.ok_or_else(|| record_err("no /meta message"))
    }

    /// A recording without episodes from its `/meta` message.
    pub fn from_meta(data: &[u8]) -> Result<Self, SimError> {
        #[derive(Deserialize)]
        struct Meta {
            format: u32,
            scenario: Scenario,
            maps: Vec<MapMeta>,
            physics_hz: u32,
            policy_hz: u32,
            state_hz: u32,
            #[serde(default)]
            pedestrian_hz: u32,
            agents: Vec<RecordedAgent>,
        }
        #[derive(Deserialize)]
        struct MapMeta {
            hash: String,
        }
        let meta: Value = parse("/meta", data)?;
        let parsed: Meta = serde_json::from_value(meta.clone()).map_err(|e| record_err(format!("/meta: {e}")))?;
        if parsed.format != 1 {
            return Err(record_err(format!("unsupported recording format {}", parsed.format)));
        }
        Ok(Recording {
            scenario: parsed.scenario,
            map_hashes: parsed.maps.into_iter().map(|m| m.hash).collect(),
            physics_hz: parsed.physics_hz,
            policy_hz: parsed.policy_hz,
            state_hz: parsed.state_hz,
            pedestrian_hz: parsed.pedestrian_hz,
            agents: parsed.agents,
            episodes: Vec::new(),
            meta,
        })
    }

    /// Add a message that follows `/meta` (`/episode` starts an episode; the others need
    /// one); topics a recording does not keep are skipped.
    pub fn push(&mut self, topic: &str, log_time_ns: u64, data: &[u8]) -> Result<(), SimError> {
        #[derive(Deserialize)]
        struct Episode {
            episode: u64,
            seed: String,
            map: usize,
            agents: Vec<EpisodeAgent>,
            #[serde(default)]
            signals: Vec<f64>,
        }
        #[derive(Deserialize)]
        struct EpisodeAgent {
            id: usize,
            goals: Vec<Goal>,
            #[serde(default)]
            route: Option<Vec<DVec3>>,
        }
        #[derive(Deserialize)]
        struct Route {
            time: f64,
            agent: usize,
            route: Vec<DVec3>,
        }
        let n = self.agents.len();
        if topic == "/episode" {
            let e: Episode = parse(topic, data)?;
            let mut goals = vec![Vec::new(); n];
            let mut routes = vec![None; n];
            for a in e.agents {
                if a.id < n {
                    goals[a.id] = a.goals;
                    routes[a.id] = a.route.map(|p| Arc::new(Polyline::new(p)));
                }
            }
            self.episodes.push(RecordedEpisode {
                number: e.episode,
                seed: e.seed,
                map: e.map,
                start: log_time_ns as f64 * 1e-9,
                goals,
                routes,
                route_updates: vec![Vec::new(); n],
                states: vec![Vec::new(); n],
                actions: vec![Vec::new(); n],
                scans: vec![Vec::new(); n],
                events: Vec::new(),
                signal_offsets: e.signals,
                pedestrians: Vec::new(),
            });
            return Ok(());
        }
        let Some(ep) = self.episodes.last_mut() else { return Err(record_err(format!("{topic} before /episode"))) };
        match topic {
            "/events" => ep.events.push(parse(topic, data)?),
            "/pedestrians" => ep.pedestrians.push(parse(topic, data)?),
            "/npcs" => {
                let npcs: RecordedNpcs = parse(topic, data)?;
                for (id, state) in npcs.states() {
                    if id >= n {
                        return Err(record_err(format!("{topic}: no agent {id} in /meta")));
                    }
                    ep.states[id].push(state);
                }
            }
            "/route" => {
                let u: Route = parse(topic, data)?;
                if u.agent >= n {
                    return Err(record_err(format!("{topic}: no agent {} in /meta", u.agent)));
                }
                ep.route_updates[u.agent].push((u.time, Arc::new(Polyline::new(u.route))));
            }
            _ => {
                let Some((id, kind)) = topic.strip_prefix("/agent/").and_then(|t| t.split_once('/')) else {
                    return Ok(());
                };
                let id: usize = id.parse().map_err(|_| record_err(format!("bad topic {topic}")))?;
                if id >= n {
                    return Err(record_err(format!("{topic}: no agent {id} in /meta")));
                }
                match kind {
                    "state" => ep.states[id].push(parse(topic, data)?),
                    "action" => ep.actions[id].push(parse(topic, data)?),
                    "lidar" => ep.scans[id].push(parse(topic, data)?),
                    _ => {}
                }
            }
        }
        Ok(())
    }

    /// Compile the recorded scenario (maps come from the cache or are generated again) and
    /// check that every map matches its recorded content hash.
    pub fn compile(&self) -> Result<CompiledScenario, SimError> {
        let compiled = self.scenario.clone().compile()?;
        let rebuilt: Vec<String> = compiled.map_hashes.iter().map(|h| h.to_string()).collect();
        if rebuilt != self.map_hashes {
            return Err(record_err(format!(
                "the rebuilt maps differ from the recorded ones (generator changed?): {rebuilt:?} != {:?}",
                self.map_hashes
            )));
        }
        Ok(compiled)
    }
}
