//! CDR encodings against ROS 2 Lyrical's (fixtures/ros/cdr.json, from
//! tools/ros/gen_cdr_fixtures.py): every message type, byte for byte, both ways.

use autonomousim_ros::msgs::*;
use serde_json::Value;

fn fixtures() -> serde_json::Map<String, Value> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures/ros/cdr.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn hex(s: &str) -> Vec<u8> {
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
}

fn check<M: RosMessage + PartialEq + std::fmt::Debug>(all: &serde_json::Map<String, Value>) -> usize {
    let f = all.get(M::TYPE).unwrap_or_else(|| panic!("no fixture for {}", M::TYPE));
    let msg: M = serde_json::from_value(f["value"].clone()).unwrap_or_else(|e| panic!("{}: {e}", M::TYPE));
    let bytes = hex(f["cdr"].as_str().unwrap());
    // rmw leaves alignment padding uninitialized: ours is zero there.
    let ours = to_cdr(&msg);
    assert_eq!(ours.len(), bytes.len(), "{}: length", M::TYPE);
    for (i, (&a, &b)) in ours.iter().zip(&bytes).enumerate() {
        assert!(a == b || a == 0, "{}: byte {i}: {a} != {b}\n{ours:?}\n{bytes:?}", M::TYPE);
    }
    assert_eq!(from_cdr::<M>(&bytes).unwrap(), msg, "{}: decoding", M::TYPE);
    1
}

#[test]
fn every_message_encodes_as_ros_does() {
    let all = fixtures();
    let n = check::<builtin_interfaces::Time>(&all)
        + check::<builtin_interfaces::Duration>(&all)
        + check::<std_msgs::Header>(&all)
        + check::<std_msgs::StringMsg>(&all)
        + check::<std_msgs::UInt32>(&all)
        + check::<std_msgs::ColorRGBA>(&all)
        + check::<std_msgs::Float32MultiArray>(&all)
        + check::<geometry_msgs::Vector3>(&all)
        + check::<geometry_msgs::Point>(&all)
        + check::<geometry_msgs::Quaternion>(&all)
        + check::<geometry_msgs::Pose>(&all)
        + check::<geometry_msgs::PoseStamped>(&all)
        + check::<geometry_msgs::PoseWithCovariance>(&all)
        + check::<geometry_msgs::Twist>(&all)
        + check::<geometry_msgs::TwistWithCovariance>(&all)
        + check::<geometry_msgs::Transform>(&all)
        + check::<geometry_msgs::TransformStamped>(&all)
        + check::<nav_msgs::Odometry>(&all)
        + check::<nav_msgs::Path>(&all)
        + check::<sensor_msgs::Imu>(&all)
        + check::<sensor_msgs::NavSatFix>(&all)
        + check::<sensor_msgs::FluidPressure>(&all)
        + check::<sensor_msgs::MagneticField>(&all)
        + check::<sensor_msgs::Range>(&all)
        + check::<sensor_msgs::PointCloud2>(&all)
        + check::<sensor_msgs::Image>(&all)
        + check::<sensor_msgs::CompressedImage>(&all)
        + check::<sensor_msgs::JointState>(&all)
        + check::<tf2_msgs::TFMessage>(&all)
        + check::<rosgraph_msgs::Clock>(&all)
        + check::<visualization_msgs::Marker>(&all)
        + check::<visualization_msgs::MarkerArray>(&all)
        + check::<std_srvs::TriggerRequest>(&all)
        + check::<std_srvs::TriggerResponse>(&all)
        + check::<std_srvs::SetBoolRequest>(&all)
        + check::<std_srvs::SetBoolResponse>(&all);
    assert_eq!(n, all.len(), "a fixture without a check");
}

#[test]
fn time_from_seconds() {
    let t = builtin_interfaces::Time::from_secs(12.25);
    assert_eq!((t.sec, t.nanosec), (12, 250_000_000));
    assert_eq!(builtin_interfaces::Time::from_secs(-1.0), builtin_interfaces::Time::default());
    assert!((t.as_secs() - 12.25).abs() < 1e-12);
}
