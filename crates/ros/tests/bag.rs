//! rosbag2 export: a recorded street run (two episodes) as a bag, read back with `mcap`.

use autonomousim_ros::bag::{ExportConfig, export};
use autonomousim_ros::definitions;
use autonomousim_ros::msgs::nav_msgs::{Odometry, Path};
use autonomousim_ros::msgs::rosgraph_msgs::Clock;
use autonomousim_ros::msgs::sensor_msgs::{Image, PointCloud2};
use autonomousim_ros::msgs::std_msgs::StringMsg;
use autonomousim_ros::msgs::tf2_msgs::TFMessage;
use autonomousim_ros::msgs::visualization_msgs::MarkerArray;
use autonomousim_ros::msgs::{RosMessage, from_cdr};
use std::collections::BTreeMap;

mod common;

#[test]
fn schemas_are_written_as_rosbag2_writes_them() {
    let s = definitions::schema("nav_msgs/Odometry").unwrap();
    assert!(s.starts_with(definitions::definition("nav_msgs/Odometry").unwrap()));
    let deps: Vec<&str> = s.lines().filter_map(|l| l.strip_prefix("MSG: ")).collect();
    // As rosbag2 wrote it for a recorded `nav_msgs/msg/Odometry` (depth first, sorted).
    assert_eq!(
        deps,
        [
            "geometry_msgs/PoseWithCovariance",
            "geometry_msgs/Pose",
            "geometry_msgs/Point",
            "geometry_msgs/Quaternion",
            "geometry_msgs/TwistWithCovariance",
            "geometry_msgs/Twist",
            "geometry_msgs/Vector3",
            "std_msgs/Header",
            "builtin_interfaces/Time",
        ]
    );
    assert!(s.contains(&format!("\n\n{}\nMSG: geometry_msgs/Pose\n", "=".repeat(80))));
    // Every message the bridge uses resolves.
    for ty in [
        "sensor_msgs/CameraInfo",
        "sensor_msgs/PointCloud2",
        "visualization_msgs/MarkerArray",
        "tf2_msgs/TFMessage",
        "nav_msgs/Path",
        "sensor_msgs/NavSatFix",
        "std_msgs/Float32MultiArray",
    ] {
        definitions::schema(ty).unwrap_or_else(|e| panic!("{ty}: {e}"));
    }
    assert_eq!(definitions::schema("visualization_msgs/MarkerArray").unwrap().matches("MSG: ").count(), 12);
}

/// Messages of a bag's MCAP file by topic: (log time, CDR bytes).
type Messages = BTreeMap<String, Vec<(u64, Vec<u8>)>>;

fn read_bag(dir: &std::path::Path) -> (mcap::Summary, Messages, Vec<u8>) {
    let file = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| p.extension().is_some_and(|e| e == "mcap"))
        .expect("an MCAP file");
    let bytes = std::fs::read(&file).unwrap();
    let summary = mcap::Summary::read(&bytes).unwrap().expect("a summary");
    let mut messages = Messages::new();
    for m in mcap::MessageStream::new(&bytes).unwrap() {
        let m = m.unwrap();
        messages.entry(m.channel.topic.clone()).or_default().push((m.log_time, m.data.to_vec()));
    }
    (summary, messages, bytes)
}

fn decode<M: RosMessage>(messages: &Messages, topic: &str) -> Vec<M> {
    messages.get(topic).map_or(&[][..], |m| &m[..]).iter().map(|(_, d)| from_cdr(d).unwrap()).collect()
}

#[test]
fn a_recording_exports_as_a_rosbag2_bag() {
    let tmp = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"));
    let (path, dir) = (tmp.join("street.mcap"), tmp.join("street_bag"));
    let _ = std::fs::remove_dir_all(&dir);
    let rec = common::record_street(&path);
    assert_eq!(rec.episodes.len(), 2);
    let info = export(&rec, &path, &dir, &ExportConfig::default()).unwrap();
    assert!(export(&rec, &path, &dir, &ExportConfig::default()).is_err(), "the directory exists");

    let (summary, messages, _) = read_bag(&dir);
    let count = |t: &str| messages.get(t).map_or(0, |m| m.len());
    // The file: rosbag2's profile, `ros2msg` schemas, CDR channels with their QoS.
    for c in summary.channels.values() {
        assert_eq!(c.message_encoding, "cdr");
        let schema = c.schema.as_ref().unwrap();
        assert_eq!(schema.encoding, "ros2msg");
        assert!(schema.name.contains("/msg/"), "{}", schema.name);
        assert!(c.metadata["offered_qos_profiles"].contains("reliability: reliable"));
    }
    let latched = |t: &str| {
        summary.channels.values().find(|c| c.topic == t).unwrap().metadata["offered_qos_profiles"]
            .contains("transient_local")
    };
    assert!(latched("/tf_static") && latched("/agent0/route") && !latched("/agent0/odom"));
    // metadata.yaml lists every topic with its count, as the export reported.
    let yaml = std::fs::read_to_string(dir.join("metadata.yaml")).unwrap();
    assert!(yaml.contains("storage_identifier: mcap") && yaml.contains("version: 9"));
    for (topic, ty, n) in &info.topics {
        assert_eq!(count(topic) as u64, *n, "{topic}");
        let entry = format!("name: {topic}\n        type: {ty}\n");
        assert!(yaml.contains(&entry), "{entry} not in\n{yaml}");
        assert!(yaml.contains(&format!("      message_count: {n}\n")), "{topic}: {n}");
    }

    // Odometry and clock: every recorded state of the ego car, at its time.
    let states: Vec<_> = rec.episodes.iter().flat_map(|e| e.states[0].iter().map(move |s| (e.start, s))).collect();
    let odom: Vec<Odometry> = decode(&messages, "/agent0/odom");
    assert_eq!(odom.len(), states.len());
    for (o, (start, s)) in odom.iter().zip(&states) {
        assert!((o.header.stamp.as_secs() - (start + s.time)).abs() < 1e-9);
        let p = &o.pose.pose.position;
        assert_eq!([p.x, p.y, p.z], s.position.to_array());
        assert_eq!(o.child_frame_id, "agent0/base_link");
    }
    let second = rec.episodes[1].start;
    assert!(second > 0.9 && odom.iter().any(|o| o.header.stamp.as_secs() >= second));
    let clock: Vec<f64> = decode::<Clock>(&messages, "/clock").iter().map(|c| c.clock.as_secs()).collect();
    // One per state time: an episode's first state has the time of the previous one's last.
    assert_eq!(clock.len(), states.len() - 1);
    assert!(clock.windows(2).all(|w| w[1] > w[0]), "{clock:?}");
    assert_eq!(count("/tf"), states.len());
    assert_eq!(count("/agent0/joint_states"), states.len());

    // LiDAR: the recorded scans' returns in the sensor frame.
    let scans: Vec<_> = rec.episodes.iter().flat_map(|e| &e.scans[0]).collect();
    let clouds: Vec<PointCloud2> = decode(&messages, "/agent0/lidar");
    assert!(scans.len() >= 7 && clouds.len() == scans.len(), "{} clouds, {} scans", clouds.len(), scans.len());
    for (c, s) in clouds.iter().zip(&scans) {
        assert_eq!(c.width as usize, s.ranges.iter().flatten().count());
        assert_eq!((c.point_step, c.header.frame_id.as_str()), (12, "agent0/lidar"));
    }
    // Camera frames as recorded (10 Hz).
    let images: Vec<Image> = decode(&messages, "/agent0/front/image");
    assert!(images.len() >= 14, "{} images", images.len());
    let i = &images[3];
    assert_eq!((i.width, i.height, i.encoding.as_str(), i.data.len()), (160, 120, "rgb8", 160 * 120 * 3));
    assert_eq!(i.header.frame_id, "agent0/front_optical");
    assert!(i.data.iter().any(|&x| x != i.data[0]), "a blank image");

    // Markers: NPCs at 10 Hz (1.5 s, two episodes), the recorded crowd, signals.
    let npcs: Vec<MarkerArray> = decode(&messages, "/autonomousim/npcs");
    assert!((15..=17).contains(&npcs.len()), "{} NPC arrays", npcs.len());
    assert!(npcs.iter().all(|m| m.markers.len() == 15));
    let peds: Vec<MarkerArray> = decode(&messages, "/autonomousim/pedestrians");
    assert_eq!(peds.len(), rec.episodes.iter().map(|e| e.pedestrians.len()).sum::<usize>());
    assert!(peds.iter().all(|m| m.markers.len() == 20));
    assert!(count("/autonomousim/signals") >= 15);
    // Latched: one route per episode, the sensor mounts, the recording's meta.
    let routes: Vec<Path> = decode(&messages, "/agent0/route");
    assert_eq!(routes.len(), 2);
    assert!(routes.iter().all(|r| r.poses.len() > 5));
    let statics: Vec<TFMessage> = decode(&messages, "/tf_static");
    let children: Vec<&str> = statics[0].transforms.iter().map(|t| t.child_frame_id.as_str()).collect();
    assert_eq!(children, ["agent0/lidar", "agent0/front", "agent0/front_optical"]);
    let meta: Vec<StringMsg> = decode(&messages, "/autonomousim/meta");
    assert_eq!(serde_json::from_str::<serde_json::Value>(&meta[0].data).unwrap(), rec.meta);
}
