//! Interop with ROS 2 Lyrical in Docker (tools/ros/run.sh; `make test-ros`): ignored by default.

use autonomousim_ros::msgs::{builtin_interfaces::Time, geometry_msgs::Twist, nav_msgs::Odometry};
use autonomousim_ros::node::{RosNode, qos};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// A DDS domain of its own, so other ROS traffic on the machine does not interfere. The tests
/// share it: run them one at a time (`make test-ros` does).
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
    // The node runs on sim time (the /clock; its time source and the probe's own /clock
    // subscription may be a tick apart), and tf2 has the odometry's pose at its stamp.
    let now = out["sim_time_now"].as_f64().unwrap();
    assert!((now - out["clock"]["last"].as_f64().unwrap()).abs() < 0.041, "sim time {now}");
    let pos = &out["odom_last"]["position"];
    let tf = out["tf_lookup"].as_array().unwrap_or_else(|| panic!("tf2: {}", out["tf_lookup"]));
    for k in 0..3 {
        assert_eq!(tf[k].as_f64(), pos[k].as_f64());
    }
    assert_eq!(out["tf_static"].as_array().unwrap().len(), 5);
    assert_eq!(out["meta"]["agents"][0]["frame"], "agent0/base_link");
}

/// The bridge on the test scenario, in lockstep (at most 5 s per step) or as fast as it goes.
fn test_bridge(lockstep: bool) -> autonomousim_ros::bridge::Bridge {
    use autonomousim_ros::bridge::{Bridge, BridgeConfig, Pacing};
    let sc = autonomousim_sim::Scenario::from_toml(include_str!("bridge_scenario.toml")).unwrap();
    let lockstep = lockstep.then_some(5.0);
    let config = BridgeConfig { domain_id: DOMAIN, pacing: Pacing::Fast, lockstep, ..Default::default() };
    Bridge::new(Arc::new(sc.compile().unwrap()), config).unwrap()
}

/// `pattern()` of tools/ros/drive.py: (linear x, y, z, angular z) at step `k`.
fn pattern(k: i64, car: bool) -> Twist {
    let c = |x: i64| x as f64;
    let (x, y, z, w) = if car {
        (0.5 * c((k / 25) % 4), 0.0, 0.0, 0.25 * c((k / 40) % 3 - 1))
    } else {
        (0.25 * c((k / 30) % 3), 0.125 * c((k / 20) % 3 - 1), 0.25 * c((k / 50) % 2), 0.125 * c((k / 35) % 2))
    };
    let mut t = Twist::default();
    (t.linear.x, t.linear.y, t.linear.z, t.angular.z) = (x, y, z, w);
    t
}

#[test]
#[ignore = "needs Docker with ros:lyrical-ros-base (make test-ros)"]
fn an_rclpy_driver_steers_the_bridge_in_lockstep() {
    use autonomousim_ros::bridge::AgentCommand;
    // Closed loop: the car drives a 12 m square, the drone flies four waypoints.
    let mut bridge = test_bridge(true);
    let driver = std::thread::spawn(|| ros("python3 tools/ros/drive.py square --car 1 --drone 0 --timeout 90"));
    while !driver.is_finished() {
        bridge.step();
    }
    let out: serde_json::Value = serde_json::from_str(driver.join().unwrap().trim()).unwrap();
    eprintln!("square: {out}, {:.1} s simulated", bridge.time());
    assert_eq!(out["car"]["reached"], 4, "{out}");
    assert_eq!(out["drone"]["reached"], 4, "{out}");
    drop(bridge);

    // Open loop: 300 steps of commands that depend on the step only, replayed in-process.
    let mut bridge = test_bridge(true);
    let driver = std::thread::spawn(|| ros("python3 tools/ros/drive.py pattern --car 1 --drone 0 --steps 300"));
    while bridge.steps() < 300 {
        bridge.step();
    }
    let hash = bridge.world().state_hash();
    driver.join().unwrap();
    drop(bridge);
    let mut replay = test_bridge(false);
    for k in 0..300 {
        replay.command(1, AgentCommand::Twist(pattern(k, true))).unwrap();
        replay.command(0, AgentCommand::Twist(pattern(k, false))).unwrap();
        replay.step();
    }
    assert_eq!(replay.world().state_hash(), hash, "the lockstep run differs from its replay");
}

#[test]
#[ignore = "needs Docker with ros:lyrical-ros-base (make test-ros)"]
fn ros_pauses_and_resets_the_bridge() {
    let mut bridge = test_bridge(false);
    let calls = std::thread::spawn(|| {
        ros("ros2 service call /autonomousim/pause std_srvs/srv/SetBool '{data: true}' \
             && ros2 service call /autonomousim/reset std_srvs/srv/Trigger \
             && ros2 service call /autonomousim/pause std_srvs/srv/SetBool '{data: false}'")
    });
    let mut paused_at = None;
    while !calls.is_finished() {
        bridge.step();
        if bridge.paused() && paused_at.is_none() {
            paused_at = Some(bridge.time());
        }
    }
    let out = calls.join().unwrap();
    assert_eq!(out.matches("success=True").count(), 3, "{out}");
    assert!(out.contains("message='paused'") && out.contains("message='running'"), "{out}");
    let t = paused_at.expect("never paused");
    assert_eq!(bridge.episodes(), 1);
    // The reset came while paused: the clock went on from where it stood.
    assert!(bridge.time() >= t && bridge.world().time() <= bridge.time() - t + 1e-9);
}

#[test]
#[ignore = "needs Docker with ros:lyrical-ros-base (make test-ros)"]
fn a_street_car_with_lidar_and_camera_bridges_in_real_time() {
    use autonomousim_ros::bridge::{Bridge, BridgeConfig, Pacing};
    // The scenario of tests/bridge.rs: a 16×900 LiDAR and a 640×480 camera at 10 Hz on the
    // machine's GPU, with 15 NPCs, 20 pedestrians and signals.
    let sc = autonomousim_sim::Scenario::from_toml(include_str!("street_scenario.toml")).unwrap();
    let config = BridgeConfig { domain_id: DOMAIN, pacing: Pacing::Realtime(1.0), ..Default::default() };
    let mut bridge = Bridge::new(Arc::new(sc.compile().unwrap()), config).unwrap();
    let probe = std::thread::spawn(|| ros("python3 tools/ros/street_probe.py --seconds 6 --agent 0"));
    let (start, t0) = (std::time::Instant::now(), bridge.time());
    while !probe.is_finished() {
        bridge.step();
    }
    let (wall, sim) = (start.elapsed().as_secs_f64(), bridge.time() - t0);
    let out: serde_json::Value = serde_json::from_str(probe.join().unwrap().trim()).unwrap();
    eprintln!("{out:#}\nsim {sim:.2} s in {wall:.2} s");
    // Real time: the bridge kept up, and ROS had every `/clock` (50 Hz).
    assert!((sim / wall - 1.0).abs() < 0.05, "sim {sim} s in {wall} s");
    let clock = &out["clock"];
    let hz = (clock["count"].as_f64().unwrap() - 1.0) / clock["sim"].as_f64().unwrap();
    assert!((hz / 50.0 - 1.0).abs() < 0.03, "clock: {clock}");
    let topic = |t: &str| &out["topics"][t];
    for t in ["/agent0/lidar", "/agent0/front/image", "/agent0/front/depth", "/agent0/front/semantic"] {
        let r = topic(t)["rate"].as_f64().unwrap_or_else(|| panic!("{t}: {}", topic(t)));
        assert!((r / 10.0 - 1.0).abs() < 0.05, "{t}: {r} Hz");
    }
    let cloud = &topic("/agent0/lidar")["last"];
    let width = cloud["width"].as_u64().unwrap();
    assert!(width > 1000 && width <= 16 * 900 && cloud["bytes"].as_u64() == Some(16 * width), "{cloud}");
    assert_eq!(cloud["fields"], serde_json::json!(["x", "y", "z", "kind"]));
    for (t, enc, bpp) in [("image", "rgb8", 3), ("depth", "32FC1", 4), ("semantic", "mono8", 1)] {
        let i = &topic(&format!("/agent0/front/{t}"))["last"];
        assert_eq!(
            (i["width"].as_u64(), i["height"].as_u64(), i["encoding"].as_str()),
            (Some(640), Some(480), Some(enc))
        );
        assert_eq!(i["bytes"].as_u64(), Some(640 * 480 * bpp));
        assert_eq!(i["frame"], "agent0/front_optical");
    }
    assert_eq!(topic("/agent0/front/camera_info")["last"]["k"][2], 320.0);
    for (t, n) in [("npcs", Some(15)), ("pedestrians", Some(20)), ("signals", None)] {
        let m = topic(&format!("/autonomousim/{t}"));
        let r = m["rate"].as_f64().unwrap_or_else(|| panic!("{t}: {m}"));
        assert!((r / 10.0 - 1.0).abs() < 0.1, "{t}: {r} Hz");
        let count = m["last"]["markers"].as_u64().unwrap();
        assert!(n.map_or(count > 0, |n| count == n), "{t}: {m}");
    }
    let route = &topic("/agent0/route")["last"];
    assert!(route["poses"].as_u64().unwrap() > 5 && route["frame"] == "map", "{route}");
}

mod common;

/// A path under the repository as the container sees it (the repository is at `/ws`).
fn in_container(path: &std::path::Path) -> String {
    let root = std::fs::canonicalize(concat!(env!("CARGO_MANIFEST_DIR"), "/../..")).unwrap();
    format!("/ws/{}", std::fs::canonicalize(path).unwrap().strip_prefix(&root).unwrap().display())
}

#[test]
#[ignore = "needs Docker with ros:lyrical-ros-base (make test-ros)"]
fn ros2_bag_records_the_bridge_with_our_schemas() {
    use autonomousim_ros::bridge::{Bridge, BridgeConfig, Pacing};
    let tmp = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"));
    let dir = tmp.join("ros_recorded_bag");
    let _ = std::fs::remove_dir_all(&dir);
    let sc = autonomousim_sim::Scenario::from_toml(include_str!("street_scenario.toml")).unwrap();
    let config = BridgeConfig { domain_id: DOMAIN, pacing: Pacing::Realtime(1.0), ..Default::default() };
    let mut bridge = Bridge::new(Arc::new(sc.compile().unwrap()), config).unwrap();
    let target = format!("{}/ros_recorded_bag", in_container(tmp));
    let record = std::thread::spawn(move || {
        ros(&format!("timeout -s INT 8 ros2 bag record -s mcap -o {target} --all-topics > /dev/null 2>&1; true"))
    });
    while !record.is_finished() {
        bridge.step();
    }
    record.join().unwrap();
    // rosbag2 wrote the bridge's types with the definitions it has: ours are the same text.
    // (`--all-topics` leaves out `/clock`.)
    let file = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| p.extension().is_some_and(|e| e == "mcap"))
        .expect("rosbag2's MCAP file");
    let bytes = std::fs::read(file).unwrap();
    let summary = mcap::Summary::read(&bytes).unwrap().unwrap();
    let mut checked = Vec::new();
    for s in summary.schemas.values() {
        let ty = s.name.replace("/msg/", "/");
        let Ok(ours) = autonomousim_ros::definitions::schema(&ty) else { continue };
        assert_eq!(ours, String::from_utf8_lossy(&s.data), "{ty}");
        checked.push(ty);
    }
    checked.sort();
    eprintln!("schemas equal rosbag2's: {checked:?}");
    for ty in [
        "nav_msgs/Odometry",
        "sensor_msgs/PointCloud2",
        "sensor_msgs/Image",
        "sensor_msgs/CameraInfo",
        "visualization_msgs/MarkerArray",
        "nav_msgs/Path",
        "tf2_msgs/TFMessage",
    ] {
        assert!(checked.iter().any(|c| c == ty), "{ty} not recorded");
    }
}

#[test]
#[ignore = "needs Docker with ros:lyrical-ros-base (make test-ros)"]
fn ros2_bag_reads_and_plays_an_exported_recording() {
    use autonomousim_ros::bag::{ExportConfig, export};
    let tmp = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"));
    let (path, dir) = (tmp.join("interop_street.mcap"), tmp.join("interop_street_bag"));
    let _ = std::fs::remove_dir_all(&dir);
    let rec = common::record_street(&path);
    let bag = export(&rec, &path, &dir, &ExportConfig::default()).unwrap();
    let target = in_container(&dir);
    // `ros2 bag info`: every topic, its type and count.
    let info = ros(&format!("ros2 bag info {target}"));
    eprintln!("{info}");
    for (topic, ty, count) in &bag.topics {
        let line = info
            .lines()
            .find(|l| l.contains(&format!("Topic: {topic} |")))
            .unwrap_or_else(|| panic!("{topic} not in\n{info}"));
        assert!(line.contains(&format!("Type: {ty} |")) && line.contains(&format!("Count: {count} |")), "{line}");
    }
    // `ros2 bag play`: the first odometry `ros2 topic echo` sees is the first recorded state.
    let echo = ros(&format!(
        "ros2 topic echo --once /agent0/odom nav_msgs/msg/Odometry & E=$!; sleep 3; \
         ros2 bag play --delay 2 {target} > /dev/null 2>&1; wait $E"
    ));
    let value = |key: &str, after: &str| -> f64 {
        let rest = &echo[echo.find(after).unwrap_or_else(|| panic!("{after} not in\n{echo}"))..];
        let line = rest.lines().find(|l| l.trim_start().starts_with(key)).unwrap();
        line.split(':').nth(1).unwrap().trim().parse().unwrap()
    };
    let s = &rec.episodes[0].states[0][0];
    let echoed = [value("x:", "position:"), value("y:", "position:"), value("z:", "position:")];
    assert_eq!(echoed, s.position.to_array(), "{echo}");
    assert_eq!(value("sec:", "stamp:") + value("nanosec:", "stamp:") * 1e-9, s.time);
    assert!(echo.contains("child_frame_id: agent0/base_link"), "{echo}");
}
