//! rosbag2 export: a recording (`autonomousim_sim::record`, MCAP with JSON messages) as a
//! rosbag2 bag (a directory with `metadata.yaml` and one MCAP file of CDR messages with
//! `ros2msg` schemas) that `ros2 bag info`/`play` and Foxglove read.
//!
//! The topics are the bridge's ([`crate::bridge`]), stamped with simulated time (s since the
//! recording started; bag times are those as ns since the epoch, so `ros2 bag info` dates
//! them in 1970):
//!
//! - `/clock` at each recorded state, `/tf` and `/agent<id>/odom` (the state rate),
//!   `/agent<id>/joint_states` (wheeled vehicles), `/agent<id>/events` (states that carry
//!   event bits) for the agents of groups without a scripted driver;
//! - `/agent<id>/<lidar>` (`PointCloud2`, x, y, z: the kinds of the returns are not recorded)
//!   for recorded scans, `/agent<id>/<camera>/image` (`rgb8`) for recorded camera frames;
//! - `/agent<id>/route` (latched) at each episode's start;
//! - `/autonomousim/{npcs, pedestrians, signals}` markers: NPCs and signals at `markers_hz`,
//!   pedestrians as recorded;
//! - `/tf_static` (sensor mounts) and `/autonomousim/meta` (the recording's `/meta`), latched.

use crate::bridge::{base_frame, header, mount_transform, optical_rotation, point_cloud, quaternion, vector};
use crate::definitions;
use crate::markers;
use crate::msgs::RosMessage;
use crate::msgs::geometry_msgs::{Point, Transform, TransformStamped};
use crate::msgs::nav_msgs::Odometry;
use crate::msgs::rosgraph_msgs::Clock;
use crate::msgs::sensor_msgs::{Image, JointState};
use crate::msgs::std_msgs::{StringMsg, UInt32};
use crate::msgs::tf2_msgs::TFMessage;
use crate::msgs::visualization_msgs::MarkerArray;
use crate::msgs::{builtin_interfaces::Time, to_cdr};
use anyhow::Context;
use autonomousim_core::rng::Seed;
use autonomousim_sensors::Sensor;
use autonomousim_sim::WorldInstance;
use autonomousim_sim::record::Recording;
use autonomousim_sim::traffic::Signals;
use std::collections::{BTreeMap, BTreeSet};
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// How a topic's publisher offered its messages (recorded in the bag; `ros2 bag play` offers
/// them the same way).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Durability {
    /// Reliable, volatile, keep last 10.
    Volatile,
    /// Reliable, transient local, keep last 1 (latched).
    Latched,
}

impl Durability {
    /// rosbag2's YAML of the profile (an `offered_qos_profiles` list), each line indented by
    /// `indent`.
    fn yaml(self, indent: &str) -> String {
        let (depth, durability) = match self {
            Durability::Volatile => (10, "volatile"),
            Durability::Latched => (1, "transient_local"),
        };
        let never = |key: &str| [format!("  {key}:"), "    sec: 9223372036".into(), "    nsec: 854775807".into()];
        let mut lines = vec![
            "- history: keep_last".to_string(),
            format!("  depth: {depth}"),
            "  reliability: reliable".into(),
            format!("  durability: {durability}"),
        ];
        lines.extend(never("deadline"));
        lines.extend(never("lifespan"));
        lines.push("  liveliness: automatic".into());
        lines.extend(never("liveliness_lease_duration"));
        lines.push("  avoid_ros_namespace_conventions: false".into());
        lines.iter().map(|l| format!("{indent}{l}")).collect::<Vec<_>>().join("\n")
    }
}

struct BagTopic {
    name: String,
    /// `package/msg/Name`.
    ty: String,
    durability: Durability,
    channel: u16,
    count: u64,
}

/// Writes a rosbag2 bag (storage `mcap`): messages by topic, then [`finish`](Self::finish).
pub struct BagWriter {
    dir: PathBuf,
    file: String,
    writer: mcap::Writer<BufWriter<std::fs::File>>,
    schemas: BTreeMap<&'static str, u16>,
    topics: Vec<BagTopic>,
    by_name: BTreeMap<String, usize>,
    sequence: u32,
    /// First and last message time (ns).
    span: Option<(u64, u64)>,
}

/// What [`BagWriter::finish`] wrote: topics with their types and message counts.
#[derive(Clone, Debug, PartialEq)]
pub struct BagInfo {
    pub dir: PathBuf,
    pub topics: Vec<(String, String, u64)>,
    pub duration: f64,
}

/// `package/Name` → `package/msg/Name`.
fn full_type(ty: &str) -> String {
    match ty.split_once('/') {
        Some((p, n)) => format!("{p}/msg/{n}"),
        None => ty.to_string(),
    }
}

impl BagWriter {
    /// A new bag in directory `dir`, which must not exist yet (as with `ros2 bag record`).
    pub fn create(dir: impl AsRef<Path>) -> anyhow::Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        anyhow::ensure!(!dir.exists(), "{} exists", dir.display());
        std::fs::create_dir_all(&dir)?;
        let name = dir.file_name().and_then(|n| n.to_str()).unwrap_or("bag");
        let file = format!("{name}_0.mcap");
        let out = BufWriter::new(std::fs::File::create(dir.join(&file))?);
        let writer = mcap::WriteOptions::new()
            .profile("ros2")
            .library("autonomousim")
            .compression(Some(mcap::Compression::Zstd))
            .create(out)?;
        Ok(Self {
            dir,
            file,
            writer,
            schemas: BTreeMap::new(),
            topics: Vec::new(),
            by_name: BTreeMap::new(),
            sequence: 0,
            span: None,
        })
    }

    fn topic<M: RosMessage>(&mut self, name: &str, durability: Durability) -> anyhow::Result<usize> {
        if let Some(&k) = self.by_name.get(name) {
            let t = &self.topics[k];
            anyhow::ensure!(t.ty == full_type(M::TYPE), "{name}: {} and {}", t.ty, M::TYPE);
            return Ok(k);
        }
        let ty = full_type(M::TYPE);
        let schema = match self.schemas.get(M::TYPE) {
            Some(&id) => id,
            None => {
                let text = definitions::schema(M::TYPE).map_err(anyhow::Error::msg)?;
                let id = self.writer.add_schema(&ty, "ros2msg", text.as_bytes())?;
                self.schemas.insert(M::TYPE, id);
                id
            }
        };
        let metadata = BTreeMap::from([("offered_qos_profiles".to_string(), durability.yaml(""))]);
        let channel = self.writer.add_channel(schema, name, "cdr", &metadata)?;
        self.topics.push(BagTopic { name: name.to_string(), ty, durability, channel, count: 0 });
        self.by_name.insert(name.to_string(), self.topics.len() - 1);
        Ok(self.topics.len() - 1)
    }

    /// Writes `msg` on `topic` at `time` (s).
    pub fn write<M: RosMessage>(
        &mut self,
        topic: &str,
        durability: Durability,
        time: f64,
        msg: &M,
    ) -> anyhow::Result<()> {
        let k = self.topic::<M>(topic, durability)?;
        let ns = (time.max(0.0) * 1e9).round() as u64;
        self.sequence = self.sequence.wrapping_add(1);
        let header = mcap::records::MessageHeader {
            channel_id: self.topics[k].channel,
            sequence: self.sequence,
            log_time: ns,
            publish_time: ns,
        };
        self.writer.write_to_known_channel(&header, &to_cdr(msg))?;
        self.topics[k].count += 1;
        self.span = Some(self.span.map_or((ns, ns), |(a, b)| (a.min(ns), b.max(ns))));
        Ok(())
    }

    /// rosbag2's `metadata.yaml` of what was written.
    fn metadata(&self) -> String {
        fn line(y: &mut String, indent: usize, s: &str) {
            y.push_str(&" ".repeat(indent));
            y.push_str(s);
            y.push('\n');
        }
        let (start, end) = self.span.unwrap_or((0, 0));
        let count: u64 = self.topics.iter().map(|t| t.count).sum();
        let y = &mut String::new();
        line(y, 0, "rosbag2_bagfile_information:");
        line(y, 2, "version: 9");
        line(y, 2, "storage_identifier: mcap");
        line(y, 2, "duration:");
        line(y, 4, &format!("nanoseconds: {}", end - start));
        line(y, 2, "starting_time:");
        line(y, 4, &format!("nanoseconds_since_epoch: {start}"));
        line(y, 2, &format!("message_count: {count}"));
        line(y, 2, "topics_with_message_count:");
        for t in &self.topics {
            line(y, 4, "- topic_metadata:");
            line(y, 8, &format!("name: {}", t.name));
            line(y, 8, &format!("type: {}", t.ty));
            line(y, 8, "serialization_format: cdr");
            line(y, 8, "offered_qos_profiles:");
            line(y, 0, &t.durability.yaml(&" ".repeat(10)));
            line(y, 8, "type_description_hash: \"\"");
            line(y, 6, &format!("message_count: {}", t.count));
        }
        line(y, 2, "compression_format: \"\"");
        line(y, 2, "compression_mode: \"\"");
        line(y, 2, "relative_file_paths:");
        line(y, 4, &format!("- {}", self.file));
        line(y, 2, "files:");
        line(y, 4, &format!("- path: {}", self.file));
        line(y, 6, "starting_time:");
        line(y, 8, &format!("nanoseconds_since_epoch: {start}"));
        line(y, 6, "duration:");
        line(y, 8, &format!("nanoseconds: {}", end - start));
        line(y, 6, &format!("message_count: {count}"));
        line(y, 2, "custom_data: ~");
        line(y, 2, "ros_distro: lyrical");
        std::mem::take(y)
    }

    /// Writes the bag's metadata (into the MCAP file, as rosbag2 does, and `metadata.yaml`)
    /// and closes it.
    pub fn finish(mut self) -> anyhow::Result<BagInfo> {
        let yaml = self.metadata();
        // The record holds the YAML without its top-level key.
        let inner: String = yaml.lines().skip(1).map(|l| format!("{}\n", l.strip_prefix("  ").unwrap_or(l))).collect();
        let inner = inner.trim_end().to_string();
        self.writer.write_metadata(&mcap::records::Metadata {
            name: "rosbag2".into(),
            metadata: BTreeMap::from([("serialized_metadata".to_string(), inner)]),
        })?;
        self.writer.finish()?;
        std::fs::write(self.dir.join("metadata.yaml"), yaml)?;
        let (start, end) = self.span.unwrap_or((0, 0));
        Ok(BagInfo {
            dir: self.dir,
            topics: self.topics.iter().map(|t| (t.name.clone(), t.ty.clone(), t.count)).collect(),
            duration: (end - start) as f64 * 1e-9,
        })
    }
}

/// Settings of [`export`].
#[derive(Clone, Debug, PartialEq)]
pub struct ExportConfig {
    /// Rate of the NPC and signal markers (Hz; 0: none). NPC markers are written at the
    /// recorded states on this grid.
    pub markers_hz: u32,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self { markers_hz: 10 }
    }
}

/// Whether `time` (s) is on the grid of `hz` (to well below a physics tick).
fn on_grid(time: f64, hz: u32) -> bool {
    let x = time * f64::from(hz);
    (x - x.round()).abs() < 1e-6
}

/// Writes recording `rec` (from MCAP file `source`, read again for its camera frames) as a
/// rosbag2 bag in directory `dir` (which must not exist).
pub fn export(rec: &Recording, source: &Path, dir: &Path, config: &ExportConfig) -> anyhow::Result<BagInfo> {
    let scenario = Arc::new(rec.compile().context("rebuilding the recording's scenario")?);
    let mut world = WorldInstance::new(scenario.clone(), Seed::from_u64(0));
    let mut bag = BagWriter::create(dir)?;
    // What the bridge would bridge: the agents of groups without a scripted driver.
    let bridged: Vec<usize> =
        (0..world.agents().len()).filter(|&i| !scenario.groups[world.agent(i).group].scripted()).collect();
    let npcs: Vec<(usize, _)> =
        (0..world.agents().len()).filter(|i| !bridged.contains(i)).map(|i| (i, markers::body_box(&world, i))).collect();
    bag.write("/autonomousim/meta", Durability::Latched, 0.0, &StringMsg { data: rec.meta.to_string() })?;
    // Sensor mounts, and the sensors whose data the recording has.
    let mut statics = Vec::new();
    let mut lidars = BTreeMap::new();
    let mut cameras = BTreeMap::new();
    for &i in &bridged {
        let a = world.agent(i);
        let frame = base_frame(a.id);
        for (s, spec) in a.sensors.iter().zip(&scenario.groups[a.group].spec.sensors) {
            let Some((position, rotation)) = mount_transform(s) else { continue };
            let sensor_frame = format!("agent{}/{}", a.id, spec.name);
            statics.push(TransformStamped {
                header: header(0.0, &frame),
                child_frame_id: sensor_frame.clone(),
                transform: Transform { translation: vector(position), rotation: quaternion(rotation) },
            });
            match s {
                // The recorder keeps the first LiDAR's scans.
                Sensor::Lidar(l) if !lidars.contains_key(&a.id) => {
                    lidars
                        .insert(a.id, (format!("/agent{}/{}", a.id, spec.name), sensor_frame, l.directions().to_vec()));
                }
                Sensor::Camera(_) => {
                    let optical = format!("{sensor_frame}_optical");
                    statics.push(TransformStamped {
                        header: header(0.0, &sensor_frame),
                        child_frame_id: optical.clone(),
                        transform: Transform {
                            translation: vector(glam::DVec3::ZERO),
                            rotation: quaternion(optical_rotation()),
                        },
                    });
                    cameras.insert(
                        format!("/agent/{}/camera/{}", a.id, spec.name),
                        (format!("/agent{}/{}/image", a.id, spec.name), optical),
                    );
                }
                _ => {}
            }
        }
    }
    bag.write("/tf_static", Durability::Latched, 0.0, &TFMessage { transforms: statics })?;
    let ped_radius = scenario.spec.pedestrians.radius;
    let mut last_clock: Option<f64> = None;
    for ep in &rec.episodes {
        let t0 = ep.start;
        if world.map_index() != ep.map {
            world.set_map(ep.map);
        }
        world.set_signals(Signals::with_offsets(ep.signal_offsets.clone()));
        let heads = markers::heads(&world);
        for &i in &bridged {
            let id = world.agent(i).id;
            if let Some(route) = ep.routes.get(id as usize).and_then(|r| r.as_ref()) {
                bag.write(&format!("/agent{id}/route"), Durability::Latched, t0, &markers::path(t0, route.points()))?;
            }
        }
        // States: clock, odometry, TF, joints, events.
        let mut ticks = BTreeSet::new();
        for &i in &bridged {
            let id = world.agent(i).id;
            let frame = base_frame(id);
            let wheeled = world.agent(i).vehicle.as_wheeled().is_some();
            for s in ep.states.get(id as usize).map_or(&[][..], |s| &s[..]) {
                let t = t0 + s.time;
                ticks.insert(s.tick);
                let (p, q) = (s.position, s.orientation);
                let mut odom =
                    Odometry { header: header(t, "map"), child_frame_id: frame.clone(), ..Default::default() };
                odom.pose.pose.position = Point { x: p.x, y: p.y, z: p.z };
                odom.pose.pose.orientation = quaternion(q);
                odom.twist.twist.linear = vector(q.inverse() * s.velocity);
                odom.twist.twist.angular = vector(s.rates);
                bag.write(&format!("/agent{id}/odom"), Durability::Volatile, t, &odom)?;
                let tf = TransformStamped {
                    header: header(t, "map"),
                    child_frame_id: frame.clone(),
                    transform: Transform { translation: vector(p), rotation: quaternion(q) },
                };
                bag.write("/tf", Durability::Volatile, t, &TFMessage { transforms: vec![tf] })?;
                if wheeled {
                    let mut js = JointState { header: header(t, &frame), ..Default::default() };
                    for (k, w) in s.wheels.iter().enumerate() {
                        js.name.extend([format!("wheel{k}"), format!("steer{k}")]);
                        js.position.extend([w.spin_angle, w.steer]);
                        js.velocity.extend([w.spin, 0.0]);
                    }
                    bag.write(&format!("/agent{id}/joint_states"), Durability::Volatile, t, &js)?;
                }
                if s.events != 0 {
                    bag.write(&format!("/agent{id}/events"), Durability::Volatile, t, &UInt32 { data: s.events })?;
                }
            }
            if let Some((topic, sensor_frame, dirs)) = lidars.get(&id) {
                for scan in ep.scans.get(id as usize).map_or(&[][..], |s| &s[..]) {
                    let t = t0 + scan.time;
                    let points: Vec<_> =
                        scan.ranges.iter().zip(dirs).filter_map(|(r, d)| r.map(|r| (*d * r, 0))).collect();
                    bag.write(topic, Durability::Volatile, t, &point_cloud(header(t, sensor_frame), &points, false))?;
                }
            }
        }
        let dt = 1.0 / f64::from(rec.physics_hz);
        for &tick in &ticks {
            // An episode's first state has the time of the previous one's last.
            let t = t0 + tick as f64 * dt;
            if last_clock.is_none_or(|l| t > l + 0.5 * dt) {
                bag.write("/clock", Durability::Volatile, t, &Clock { clock: Time::from_secs(t) })?;
                last_clock = Some(t);
            }
        }
        // Markers.
        if config.markers_hz > 0 && !npcs.is_empty() {
            let mut by_tick: BTreeMap<u64, Vec<_>> = BTreeMap::new();
            for &(i, b) in &npcs {
                let id = world.agent(i).id;
                for s in ep.states.get(id as usize).map_or(&[][..], |s| &s[..]) {
                    if on_grid(s.time, config.markers_hz) {
                        by_tick.entry(s.tick).or_default().push(markers::npc(
                            t0 + s.time,
                            id,
                            s.position,
                            s.orientation,
                            b,
                            s.disabled,
                        ));
                    }
                }
            }
            for (tick, markers) in by_tick {
                bag.write("/autonomousim/npcs", Durability::Volatile, t0 + tick as f64 * dt, &MarkerArray { markers })?;
            }
        }
        for peds in &ep.pedestrians {
            let t = t0 + peds.time;
            let markers = (0..peds.len())
                .filter_map(|k| {
                    peds.get(k).map(|p| markers::pedestrian(t, k, p.position, p.heading, ped_radius, p.height, p.state))
                })
                .collect();
            bag.write("/autonomousim/pedestrians", Durability::Volatile, t, &MarkerArray { markers })?;
        }
        if config.markers_hz > 0 && !heads.is_empty() {
            let lanes = world.map().roads().lanes();
            let n = (ep.duration() * f64::from(config.markers_hz)).floor() as u64;
            for k in 0..=n {
                let local = k as f64 / f64::from(config.markers_hz);
                let t = t0 + local;
                let markers = heads
                    .iter()
                    .enumerate()
                    .map(|(j, h)| markers::signal(t, j, h, world.signals().light(lanes, h.connector, local)))
                    .collect();
                bag.write("/autonomousim/signals", Durability::Volatile, t, &MarkerArray { markers })?;
            }
        }
    }
    // Camera frames (`foxglove.RawImage` JSON) as they are in the recording.
    if !cameras.is_empty() {
        let bytes = std::fs::read(source)?;
        for m in mcap::MessageStream::new(&bytes)? {
            let m = m?;
            let Some((topic, frame)) = cameras.get(&m.channel.topic) else { continue };
            let v: serde_json::Value = serde_json::from_slice(&m.data)?;
            let t = m.log_time as f64 * 1e-9;
            let data = base64_decode(v["data"].as_str().unwrap_or_default())?;
            let (w, h) = (v["width"].as_u64().unwrap_or(0) as u32, v["height"].as_u64().unwrap_or(0) as u32);
            let image = Image {
                header: header(t, frame),
                height: h,
                width: w,
                encoding: "rgb8".into(),
                is_bigendian: 0,
                step: 3 * w,
                data,
            };
            bag.write(topic, Durability::Volatile, t, &image)?;
        }
    }
    bag.finish()
}

/// Standard base64 (with padding) decoded.
fn base64_decode(s: &str) -> anyhow::Result<Vec<u8>> {
    let value = |c: u8| -> anyhow::Result<u32> {
        Ok(match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => anyhow::bail!("not base64: {:?}", c as char),
        }
        .into())
    };
    let bytes = s.trim_end_matches('=').as_bytes();
    let mut out = Vec::with_capacity(bytes.len() * 3 / 4);
    for chunk in bytes.chunks(4) {
        let mut n = 0u32;
        for (k, &c) in chunk.iter().enumerate() {
            n |= value(c)? << (18 - 6 * k);
        }
        out.extend(&n.to_be_bytes()[1..chunk.len()]);
    }
    Ok(out)
}
