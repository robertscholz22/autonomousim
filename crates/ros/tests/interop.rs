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

#[test]
#[ignore = "needs Docker with ros:lyrical-ros-base (make test-ros)"]
fn ros_sees_the_bridge_at_its_rates_with_sim_time_and_tf() {
    use autonomousim_ros::bridge::{Bridge, BridgeConfig, Pacing};
    let sc = autonomousim_sim::Scenario::from_toml(include_str!("bridge_scenario.toml")).unwrap();
    let config = BridgeConfig { domain_id: DOMAIN, pacing: Pacing::Realtime(1.0), ..Default::default() };
    let mut bridge = Bridge::new(Arc::new(sc.compile().unwrap()), config).unwrap();
    // The probe starts with the container (some seconds) and listens for 6 s.
    let probe = std::thread::spawn(|| {
        ros("python3 tools/ros/probe.py --seconds 6 --agent 0 \
             --sensors imu:Imu,gps:NavSatFix,baro:FluidPressure,mag:MagneticField,down:Range")
    });
    while !probe.is_finished() {
        bridge.step();
    }
    let out: serde_json::Value = serde_json::from_str(probe.join().unwrap().trim()).unwrap();
    eprintln!("{out:#}");
    let topic = |t: &str| &out["topics"][t];
    for (t, hz) in [
        ("/agent0/odom", 50.0),
        ("/agent0/imu", 250.0),
        ("/agent0/gps", 10.0),
        ("/agent0/baro", 50.0),
        ("/agent0/mag", 50.0),
        ("/agent0/down", 50.0),
    ] {
        let r = topic(t)["rate"].as_f64().unwrap_or_else(|| panic!("{t}: {}", topic(t)));
        assert!((r / hz - 1.0).abs() < 0.03, "{t}: {r} Hz, expected {hz}");
    }
    // Reliable odometry arrives without gaps.
    assert!(topic("/agent0/odom")["max_gap"].as_f64().unwrap() < 0.021, "{}", topic("/agent0/odom"));
    // The node runs on sim time (the /clock), and tf2 has the odometry's pose at its stamp.
    let now = out["sim_time_now"].as_f64().unwrap();
    assert!((now - out["clock"]["last"].as_f64().unwrap()).abs() < 1e-6, "sim time {now}");
    let pos = &out["odom_last"]["position"];
    let tf = out["tf_lookup"].as_array().unwrap_or_else(|| panic!("tf2: {}", out["tf_lookup"]));
    for k in 0..3 {
        assert_eq!(tf[k].as_f64(), pos[k].as_f64());
    }
    assert_eq!(out["tf_static"].as_array().unwrap().len(), 5);
    assert_eq!(out["meta"]["agents"][0]["frame"], "agent0/base_link");
}
