//! Recording of world instances to MCAP, with JSON messages that open in Foxglove.
//!
//! | Topic | When | Content |
//! |---|---|---|
//! | `/meta` | once | scenario, map pool (metadata and content hashes), vehicle definitions, rates, agent list |
//! | `/episode` | every reset | episode number and seed, map index, environment, spawn poses and goals |
//! | `/agent/<id>/state` | `state_hz` | time, pose, velocity, rates, wind, goal, events; rotor speeds (multirotors) or steering, wheels and powertrain (ground vehicles) |
//! | `/agent/<id>/pose` | `state_hz` | the pose as `foxglove.PoseInFrame` (frame `world`) |
//! | `/agent/<id>/action` | each action | the normalised action |
//! | `/agent/<id>/lidar` | each scan, if enabled | sensor pose and ranges |
//! | `/events` | when an agent gets new event bits | agent id and event names |
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
use crate::scenario::{CompiledScenario, Goal, Scenario};
use crate::world::{STATE_FIELDS, WorldInstance};
use autonomousim_sensors::Sensor;
use autonomousim_vehicles::{SharedDef, Vehicle, VehicleDef};
use glam::{DQuat, DVec3};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::io::{BufWriter, Seek, Write};
use std::path::Path;

/// Destination of recorded messages (an MCAP file now; a live viewer connection later).
pub trait TelemetrySink: Send {
    /// Register a topic with a JSON schema; returns its channel id.
    fn add_channel(&mut self, topic: &str, schema_name: &str, schema: &str) -> Result<u16, SimError>;
    fn write(&mut self, channel: u16, log_time_ns: u64, data: &[u8]) -> Result<(), SimError>;
    fn finish(&mut self) -> Result<(), SimError>;
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
}

impl Default for RecorderConfig {
    fn default() -> Self {
        Self { state_hz: 50, lidar: false }
    }
}

const OBJECT_SCHEMA: &str = r#"{"type":"object"}"#;

const POSE_IN_FRAME_SCHEMA: &str = r#"{"title":"foxglove.PoseInFrame","type":"object","properties":{"timestamp":{"type":"object","properties":{"sec":{"type":"integer","minimum":0},"nsec":{"type":"integer","minimum":0,"maximum":999999999}}},"frame_id":{"type":"string"},"pose":{"type":"object","properties":{"position":{"type":"object","properties":{"x":{"type":"number"},"y":{"type":"number"},"z":{"type":"number"}}},"orientation":{"type":"object","properties":{"x":{"type":"number"},"y":{"type":"number"},"z":{"type":"number"},"w":{"type":"number"}}}}}}}"#;

#[derive(Clone, Copy, Debug)]
struct AgentChannels {
    state: u16,
    pose: u16,
    action: u16,
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
            seen: Vec::new(),
            error: None,
        }
    }

    /// Record to an MCAP file.
    pub fn create(path: impl AsRef<Path>, config: RecorderConfig) -> Result<Self, SimError> {
        Ok(Self::new(Box::new(McapSink::create(path)?), config))
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
        for a in w.agents() {
            let p = format!("/agent/{}", a.id);
            let lidar = a.sensors.iter().any(|s| matches!(s, Sensor::Lidar(_))) && self.config.lidar;
            self.agents.push(AgentChannels {
                state: self.sink.add_channel(&format!("{p}/state"), "autonomousim.AgentState", OBJECT_SCHEMA)?,
                pose: self.sink.add_channel(&format!("{p}/pose"), "foxglove.PoseInFrame", POSE_IN_FRAME_SCHEMA)?,
                action: self.sink.add_channel(&format!("{p}/action"), "autonomousim.Action", OBJECT_SCHEMA)?,
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
        let msg = json!({
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
                json!({
                    "id": a.id,
                    "spawn": {"position": a.spawn.pos, "orientation": a.spawn.rot},
                    "goals": a.goals,
                })
            })
            .collect();
        let msg = json!({
            "episode": self.episodes,
            "seed": seed,
            "map": w.map_index(),
            "environment": w.env().config,
            "agents": agents,
        });
        self.episodes += 1;
        self.send(episode, &msg);
        self.seen.fill(0);
        self.write_states(w);
    }

    /// After new actions were set.
    pub fn on_actions(&mut self, w: &WorldInstance) {
        if self.meta.is_none() {
            return;
        }
        let time = w.time();
        for a in w.agents() {
            let ch = self.agents[a.id as usize].action;
            self.send(ch, &json!({"time": time, "action": a.action.as_slice()}));
        }
    }

    /// After every physics tick.
    pub fn on_tick(&mut self, w: &WorldInstance) {
        let Some((_, _, events)) = self.meta else { return };
        self.ticks += 1;
        let time = w.time();
        for a in w.agents() {
            let i = a.id as usize;
            let bits = a.events.0;
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
        }
        if w.clock().tick.is_multiple_of(self.divider) {
            self.write_states(w);
        }
    }

    fn write_states(&mut self, w: &WorldInstance) {
        let time = w.time();
        let t_ns = self.time_ns();
        let stamp = json!({"sec": t_ns / 1_000_000_000, "nsec": t_ns % 1_000_000_000});
        for a in w.agents() {
            let ch = self.agents[a.id as usize];
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
                            steer: w.steer,
                            travel: w.travel,
                            drive_torque: w.drive_torque,
                            brake_torque: w.brake_torque,
                            load: w.tire.fz,
                        })
                        .collect();
                    let pt = v.powertrain();
                    m.insert("steering".into(), json!(v.steering_angle()));
                    m.insert("wheels".into(), json!(wheels));
                    m.insert("gear".into(), json!(pt.gear));
                    m.insert("engine_speed".into(), json!(pt.engine_speed));
                }
            }
            self.send(ch.state, &msg);
            let pose = json!({
                "timestamp": stamp,
                "frame_id": "world",
                "pose": {"position": xyz(p), "orientation": {"x": q.x, "y": q.y, "z": q.z, "w": q.w}},
            });
            self.send(ch.pose, &pose);
        }
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

// ---------------------------------------------------------------------------- reading

/// A `/agent/<id>/state` message.
#[derive(Clone, Debug, PartialEq, Deserialize)]
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
    pub wind: DVec3,
    pub goal: DVec3,
    pub goal_yaw: f64,
    /// Event bits since the last policy step.
    pub events: u32,
    pub disabled: bool,
}

/// A wheel in a ground vehicle's state message.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RecordedWheel {
    /// Spin rate (rad/s), steering angle (rad) and suspension travel (m, bump positive).
    pub spin: f64,
    pub steer: f64,
    pub travel: f64,
    /// Drive and brake torque (N·m) and tyre load (N).
    pub drive_torque: f64,
    pub brake_torque: f64,
    pub load: f64,
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
    pub states: Vec<Vec<RecordedState>>,
    pub actions: Vec<Vec<RecordedAction>>,
    pub scans: Vec<Vec<RecordedScan>>,
    pub events: Vec<RecordedEvents>,
}

impl RecordedEpisode {
    /// Time of the last recorded state (s).
    pub fn duration(&self) -> f64 {
        self.states.iter().filter_map(|s| s.last()).map(|s| s.time).fold(0.0, f64::max)
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
        #[derive(Deserialize)]
        struct Meta {
            format: u32,
            scenario: Scenario,
            maps: Vec<MapMeta>,
            physics_hz: u32,
            policy_hz: u32,
            state_hz: u32,
            agents: Vec<RecordedAgent>,
        }
        #[derive(Deserialize)]
        struct MapMeta {
            hash: String,
        }
        #[derive(Deserialize)]
        struct Episode {
            episode: u64,
            seed: String,
            map: usize,
            agents: Vec<EpisodeAgent>,
        }
        #[derive(Deserialize)]
        struct EpisodeAgent {
            id: usize,
            goals: Vec<Goal>,
        }

        let mut rec: Option<Recording> = None;
        for m in mcap::MessageStream::new(bytes).map_err(mcap_err)? {
            let m = m.map_err(mcap_err)?;
            let topic = m.channel.topic.as_str();
            if topic == "/meta" {
                let meta: Value = parse(topic, &m.data)?;
                let parsed: Meta =
                    serde_json::from_value(meta.clone()).map_err(|e| record_err(format!("/meta: {e}")))?;
                if parsed.format != 1 {
                    return Err(record_err(format!("unsupported recording format {}", parsed.format)));
                }
                rec = Some(Recording {
                    scenario: parsed.scenario,
                    map_hashes: parsed.maps.into_iter().map(|m| m.hash).collect(),
                    physics_hz: parsed.physics_hz,
                    policy_hz: parsed.policy_hz,
                    state_hz: parsed.state_hz,
                    agents: parsed.agents,
                    episodes: Vec::new(),
                    meta,
                });
                continue;
            }
            let Some(r) = rec.as_mut() else { return Err(record_err(format!("{topic} before /meta"))) };
            let n = r.agents.len();
            if topic == "/episode" {
                let e: Episode = parse(topic, &m.data)?;
                let mut goals = vec![Vec::new(); n];
                for a in e.agents {
                    if let Some(g) = goals.get_mut(a.id) {
                        *g = a.goals;
                    }
                }
                r.episodes.push(RecordedEpisode {
                    number: e.episode,
                    seed: e.seed,
                    map: e.map,
                    start: m.log_time as f64 * 1e-9,
                    goals,
                    states: vec![Vec::new(); n],
                    actions: vec![Vec::new(); n],
                    scans: vec![Vec::new(); n],
                    events: Vec::new(),
                });
                continue;
            }
            let Some(ep) = r.episodes.last_mut() else { return Err(record_err(format!("{topic} before /episode"))) };
            if topic == "/events" {
                ep.events.push(parse(topic, &m.data)?);
                continue;
            }
            let Some((id, kind)) = topic.strip_prefix("/agent/").and_then(|t| t.split_once('/')) else { continue };
            let id: usize = id.parse().map_err(|_| record_err(format!("bad topic {topic}")))?;
            if id >= n {
                return Err(record_err(format!("{topic}: no agent {id} in /meta")));
            }
            match kind {
                "state" => ep.states[id].push(parse(topic, &m.data)?),
                "action" => ep.actions[id].push(parse(topic, &m.data)?),
                "lidar" => ep.scans[id].push(parse(topic, &m.data)?),
                _ => {}
            }
        }
        rec.ok_or_else(|| record_err("no /meta message"))
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
