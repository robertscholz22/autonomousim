//! Interop with ROS 2 Lyrical in Docker (tools/ros/run.sh; `make test-ros`): ignored by default.

use autonomousim_ros::msgs::{builtin_interfaces::Time, geometry_msgs::Twist, nav_msgs::Odometry};
use autonomousim_ros::node::{RosNode, qos};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// A DDS domain of its own, so other ROS traffic on the machine does not interfere.
const DOMAIN: u16 = 42;

/// Runs a shell command in the ROS container; returns its standard output.
pub fn ros(cmd: &str) -> String {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
    let out = Command::new(format!("{root}/tools/ros/run.sh"))
        .args(["bash", "-c", cmd])
        .env("ROS_DOMAIN_ID", DOMAIN.to_string())
        .output()
        .expect("docker (tools/ros/run.sh)");
    assert!(out.status.success(), "{cmd}: {}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout).into_owned()
}

#[test]
#[ignore = "needs Docker with ros:lyrical-ros-base (make test-ros)"]
fn ros_echoes_our_odometry_and_we_receive_its_twists() {
    let mut node = RosNode::new("/", "autonomousim_interop", DOMAIN).unwrap();
    let odom = node.publisher::<Odometry>("/autonomousim_test/odom", qos::RELIABLE).unwrap();
    let cmd = node.subscription::<Twist>("/autonomousim_test/cmd_vel", qos::RELIABLE).unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    let received = Arc::new(Mutex::new(Vec::new()));
    let worker = {
        let (stop, received) = (stop.clone(), received.clone());
        std::thread::spawn(move || {
            let mut k = 0u32;
            while !stop.load(Ordering::Relaxed) {
                let mut msg = Odometry::default();
                msg.header.stamp = Time::from_secs(k as f64 * 0.1);
                msg.header.frame_id = "map".into();
                msg.child_frame_id = "agent0/base_link".into();
                msg.pose.pose.position.x = 1.25;
                msg.pose.pose.orientation.z = 0.5;
                msg.twist.twist.angular.z = -0.75;
                odom.publish(msg).unwrap();
                while let Some((t, _)) = cmd.take().unwrap() {
                    received.lock().unwrap().push(t);
                }
                k += 1;
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
        })
    };
    let echo = ros("timeout 30 ros2 topic echo --once /autonomousim_test/odom nav_msgs/msg/Odometry");
    for line in ["frame_id: map", "child_frame_id: agent0/base_link", "x: 1.25", "z: 0.5", "z: -0.75"] {
        assert!(echo.contains(line), "{line:?} not in\n{echo}");
    }
    ros("timeout 30 ros2 topic pub --times 3 -r 5 -w 1 /autonomousim_test/cmd_vel geometry_msgs/msg/Twist \
         '{linear: {x: 1.5, y: -2.0}, angular: {z: -0.25}}'");
    std::thread::sleep(std::time::Duration::from_millis(300));
    stop.store(true, Ordering::Relaxed);
    worker.join().unwrap();
    let got = received.lock().unwrap();
    assert!(!got.is_empty(), "no Twist received");
    for t in got.iter() {
        assert_eq!((t.linear.x, t.linear.y, t.angular.z), (1.5, -2.0, -0.25));
    }
}
