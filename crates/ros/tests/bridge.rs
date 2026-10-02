//! The bridge over DDS, received in-process (RustDDS on both ends; no ROS needed): rates,
//! stamps, frames, values, latched topics and episode resets.

use autonomousim_ros::bridge::{AgentCommand, Bridge, BridgeConfig, Pacing};
use autonomousim_ros::msgs::RosMessage;
use autonomousim_ros::msgs::geometry_msgs::Twist;
use autonomousim_ros::msgs::nav_msgs::Odometry;
use autonomousim_ros::msgs::rosgraph_msgs::Clock;
use autonomousim_ros::msgs::sensor_msgs::{FluidPressure, Imu, JointState, MagneticField, NavSatFix, Range};
use autonomousim_ros::msgs::std_msgs::StringMsg;
use autonomousim_ros::msgs::tf2_msgs::TFMessage;
use autonomousim_ros::node::{RosNode, Subscription, qos};
use autonomousim_sim::Scenario;
use std::sync::Arc;

/// A domain of its own (other tests and ROS traffic on the machine stay out).
const DOMAIN: u16 = 43;

const SCENARIO: &str = include_str!("bridge_scenario.toml");

fn drain<M: RosMessage>(s: &Subscription<M>, out: &mut Vec<M>) {
    while let Some((m, _)) = s.take().unwrap() {
        out.push(m);
    }
}

#[test]
fn the_bridge_publishes_the_simulation() {
    let sc = Arc::new(Scenario::from_toml(SCENARIO).unwrap().compile().unwrap());
    let mut rx = RosNode::new("/", "bridge_test", DOMAIN).unwrap();
    let clock = rx.subscription::<Clock>("/clock", qos::RELIABLE).unwrap();
    let odom = rx.subscription::<Odometry>("/agent0/odom", qos::RELIABLE).unwrap();
    let imu = rx.subscription::<Imu>("/agent0/imu", qos::SENSOR.history(ros2_client::qos::History::KeepAll)).unwrap();
    let gps = rx.subscription::<NavSatFix>("/agent0/gps", qos::SENSOR).unwrap();
    let baro = rx.subscription::<FluidPressure>("/agent0/baro", qos::SENSOR).unwrap();
    let mag = rx.subscription::<MagneticField>("/agent0/mag", qos::SENSOR).unwrap();
    let range = rx.subscription::<Range>("/agent0/down", qos::SENSOR).unwrap();
    let joints = rx.subscription::<JointState>("/agent1/joint_states", qos::RELIABLE).unwrap();
    let tf = rx.subscription::<TFMessage>("/tf", qos::RELIABLE).unwrap();

    let config = BridgeConfig {
        domain_id: DOMAIN,
        pacing: Pacing::Realtime(1.0),
        episode_time: Some(2.0),
        ..Default::default()
    };
    let mut bridge = Bridge::new(sc.clone(), config).unwrap();
    // Latched topics reach subscriptions created after they were published.
    let tf_static = rx.subscription::<TFMessage>("/tf_static", qos::LATCHED).unwrap();
    let meta = rx.subscription::<StringMsg>("/autonomousim/meta", qos::LATCHED).unwrap();

    let (
        mut clocks,
        mut odoms,
        mut imus,
        mut fixes,
        mut baros,
        mut mags,
        mut ranges,
        mut js,
        mut tfs,
        mut statics,
        mut metas,
    ) = (vec![], vec![], vec![], vec![], vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
    // 3.5 s: an episode of 2 s and a reset (the clock runs on).
    while bridge.time() < 3.5 {
        bridge.step();
        drain(&clock, &mut clocks);
        drain(&odom, &mut odoms);
        drain(&imu, &mut imus);
        drain(&gps, &mut fixes);
        drain(&baro, &mut baros);
        drain(&mag, &mut mags);
        drain(&range, &mut ranges);
        drain(&joints, &mut js);
        drain(&tf, &mut tfs);
        drain(&tf_static, &mut statics);
        drain(&meta, &mut metas);
    }
    assert_eq!(bridge.episodes(), 1);

    // The clock ticks at the policy rate and never jumps back across the reset.
    let t: Vec<f64> = clocks.iter().map(|c| c.clock.as_secs()).collect();
    assert!(t.len() > 75, "{} clock messages", t.len());
    // (Our RustDDS reader loses a run of samples while discovery settles, up to about a dozen
    // in a row, depending on the endpoints matched; ROS's readers get every one, see the
    // Docker tests.)
    let mut lost = 0.0;
    for w in t.windows(2) {
        let k = (w[1] - w[0]) / 0.02; // policy steps (1, or more after lost samples)
        assert!(k > 0.5 && (k - k.round()).abs() < 1e-6, "clock step {} -> {}", w[0], w[1]);
        lost += k.round() - 1.0;
    }
    assert!(lost <= 25.0, "{lost} clock messages lost");
    assert!(*t.last().unwrap() > 3.4);

    // Sensors at their rates, stamped with their measurement times (after discovery: 1.5–3.5 s).
    let rate = |stamps: Vec<f64>| {
        let s: Vec<f64> = stamps.into_iter().filter(|&s| (1.5..3.4).contains(&s)).collect();
        (s.len() as f64 - 1.0) / (s.last().unwrap() - s.first().unwrap())
    };
    let r = rate(imus.iter().map(|m| m.header.stamp.as_secs()).collect());
    assert!((r - 250.0).abs() < 5.0, "imu {r} Hz");
    let r = rate(baros.iter().map(|m| m.header.stamp.as_secs()).collect());
    assert!((r - 50.0).abs() < 2.0, "baro {r} Hz");
    let r = rate(fixes.iter().map(|m| m.header.stamp.as_secs()).collect());
    assert!((r - 10.0).abs() < 1.0, "gps {r} Hz");
    assert!(!mags.is_empty() && !ranges.is_empty());

    // Values: hovering 2 m up in velocity mode, the IMU feels +g, the rangefinder sees the
    // ground about 2 m below, the GPS has a fix near the map's origin.
    let last = imus.last().unwrap();
    assert_eq!(last.header.frame_id, "agent0/imu");
    assert!((last.linear_acceleration.z - 9.81).abs() < 0.5, "{:?}", last.linear_acceleration);
    assert_eq!(last.orientation_covariance.0[0], -1.0);
    let d = ranges.last().unwrap();
    assert_eq!(d.header.frame_id, "agent0/down");
    assert!((d.range - 2.0).abs() < 0.5, "range {}", d.range);
    let fix = fixes.last().unwrap();
    assert!(fix.latitude.abs() <= 90.0 && fix.position_covariance.0[0] > 0.0);
    assert!(baros.last().unwrap().fluid_pressure > 80_000.0);

    // Odometry and /tf at the policy rate, matching the world (pose in map, twist in the body).
    // RustDDS readers (not ROS: tests/interop.rs) now and then lose a few reliable samples in a
    // row: the stamps step by 1/50 s, with at most a short gap.
    let os: Vec<f64> = odoms.iter().map(|m| m.header.stamp.as_secs()).collect();
    assert!(os.len() > 75, "{} odometry messages", os.len());
    let steps: Vec<f64> = os.windows(2).map(|w| w[1] - w[0]).collect();
    assert!(steps.iter().filter(|&&d| (d - 0.02).abs() < 1e-6).count() >= steps.len() - 2, "{steps:?}");
    let o = odoms.last().unwrap();
    assert_eq!((o.header.frame_id.as_str(), o.child_frame_id.as_str()), ("map", "agent0/base_link"));
    let v = &bridge.world().agent(0).vehicle;
    let p = v.position();
    assert_eq!((o.pose.pose.position.x, o.pose.pose.position.y, o.pose.pose.position.z), (p.x, p.y, p.z));
    let tf_last = tfs.last().unwrap();
    assert_eq!(tf_last.transforms.len(), 2);
    assert_eq!(tf_last.transforms[1].child_frame_id, "agent1/base_link");
    assert_eq!(tf_last.transforms[0].transform.translation.x, p.x);

    // Wheels of the car.
    let j = js.last().unwrap();
    assert_eq!(j.name.len(), 8);
    assert_eq!((j.name[0].as_str(), j.name[1].as_str()), ("wheel0", "steer0"));

    // Latched: the sensor mounts (the rangefinder's x axis along its beam, straight down) and meta.
    let st = &statics.last().expect("no /tf_static").transforms;
    assert_eq!(st.len(), 5);
    let down = st.iter().find(|t| t.child_frame_id == "agent0/down").unwrap();
    assert_eq!((down.header.frame_id.as_str(), down.transform.translation.x), ("agent0/base_link", 0.1));
    let q = down.transform.rotation;
    let x_axis_z = 2.0 * (q.x * q.z - q.w * q.y); // z of the rotated x axis
    assert!((x_axis_z + 1.0).abs() < 1e-9, "beam axis z {x_axis_z}");
    let meta: serde_json::Value = serde_json::from_str(&metas.last().expect("no meta").data).unwrap();
    assert_eq!(meta["policy_hz"], 50);
    assert_eq!(meta["agents"][1]["frame"], "agent1/base_link");
    assert_eq!(meta["map_hashes"][0].as_str().unwrap(), sc.map_hashes[0].hex());
}

#[test]
fn fast_pacing_beats_real_time() {
    let sc = Arc::new(Scenario::from_toml(SCENARIO).unwrap().compile().unwrap());
    let config = BridgeConfig { domain_id: DOMAIN + 1, pacing: Pacing::Fast, ..Default::default() };
    let mut bridge = Bridge::new(sc, config).unwrap();
    let start = std::time::Instant::now();
    bridge.run_until(5.0);
    let wall = start.elapsed().as_secs_f64();
    assert!(wall < 2.5, "5 s simulated in {wall} s");
    // Real time at ×4: 2 s simulated in about 0.5 s.
    bridge = Bridge::new(
        Arc::new(Scenario::from_toml(SCENARIO).unwrap().compile().unwrap()),
        BridgeConfig { domain_id: DOMAIN + 1, pacing: Pacing::Realtime(4.0), ..Default::default() },
    )
    .unwrap();
    let start = std::time::Instant::now();
    bridge.run_until(2.0);
    let wall = start.elapsed().as_secs_f64();
    assert!((wall - 0.5).abs() < 0.1, "2 s at ×4 took {wall} s");
}

fn twist(vx: f64, vz: f64, wz: f64) -> AgentCommand {
    let mut t = Twist::default();
    (t.linear.x, t.linear.z, t.angular.z) = (vx, vz, wz);
    AgentCommand::Twist(t)
}

#[test]
fn commands_are_held_and_time_out() {
    let sc = Arc::new(Scenario::from_toml(SCENARIO).unwrap().compile().unwrap());
    let config = BridgeConfig { domain_id: DOMAIN + 2, pacing: Pacing::Fast, ..Default::default() };
    let mut bridge = Bridge::new(sc, config).unwrap();
    let z0 = bridge.world().agent(0).vehicle.position().z;
    // The car at 3 m/s turning at 0.3 rad/s, the drone climbing at 1 m/s for 1 s.
    while bridge.time() < 4.0 {
        bridge.command(1, twist(3.0, 0.0, 0.3)).unwrap();
        if bridge.time() < 1.0 {
            bridge.command(0, twist(0.0, 1.0, 0.0)).unwrap();
        }
        bridge.step();
    }
    let car = &bridge.world().agent(1).vehicle;
    assert!((car.lin_vel_body().x - 3.0).abs() < 0.3, "car speed {}", car.lin_vel_body().x);
    assert!((car.ang_vel_body().z - 0.3).abs() < 0.05, "car yaw rate {}", car.ang_vel_body().z);
    // The drone's command timed out at 1.5 s: it holds where it was then, about 1 m up.
    let drone = &bridge.world().agent(0).vehicle;
    let dz = drone.position().z - z0;
    assert!((dz - 1.2).abs() < 0.4 && drone.lin_vel_body().length() < 0.1, "drone up {dz} m");
    // No more commands: the car brakes to a stop.
    bridge.run_until(8.0);
    let car = &bridge.world().agent(1).vehicle;
    assert!(car.lin_vel_body().length() < 0.1, "car still at {}", car.lin_vel_body());
    // Actions: the group's normalized action, of its length.
    let dim = bridge.world().scenario().groups[1].act_dim();
    assert!(bridge.command(1, AgentCommand::Action(vec![0.0; dim + 1])).is_err());
    bridge.command(1, AgentCommand::Action(vec![0.5; dim])).unwrap();
    assert!(bridge.command(7, twist(1.0, 0.0, 0.0)).is_err());
}

#[test]
fn pause_and_reset() {
    let sc = Arc::new(Scenario::from_toml(SCENARIO).unwrap().compile().unwrap());
    let config = BridgeConfig { domain_id: DOMAIN + 3, pacing: Pacing::Fast, ..Default::default() };
    let mut bridge = Bridge::new(sc, config).unwrap();
    bridge.run_until(1.0);
    let hash = bridge.world().state_hash();
    bridge.set_paused(true);
    for _ in 0..10 {
        bridge.step();
    }
    assert_eq!((bridge.time(), bridge.world().state_hash()), (1.0, hash));
    // A reset while paused: a new episode at once, the clock where it was.
    bridge.request_reset();
    bridge.step();
    assert_eq!((bridge.episodes(), bridge.world().time(), bridge.time()), (1, 0.0, 1.0));
    bridge.set_paused(false);
    bridge.step();
    assert!((bridge.time() - 1.02).abs() < 1e-9);
    // While running: after the step.
    bridge.request_reset();
    bridge.step();
    assert_eq!((bridge.episodes(), bridge.world().time()), (2, 0.0));
    assert!((bridge.time() - 1.04).abs() < 1e-9);
}

/// A car with LiDAR and camera in a street with traffic, pedestrians and signals (rendered
/// on lavapipe).
#[test]
fn the_street_scenario_bridges_lidar_camera_markers_and_route() {
    use autonomousim_ros::msgs::nav_msgs::Path;
    use autonomousim_ros::msgs::sensor_msgs::{CameraInfo, Image, PointCloud2};
    use autonomousim_ros::msgs::visualization_msgs::MarkerArray;
    autonomousim_sim::camera::use_adapter(&autonomousim_render::AdapterChoice::Software);
    // A 160×120 camera: RustDDS readers drop parts of the 640×480 stream (Fast DDS's, with large
    // enough socket buffers, do not: tests/interop.rs).
    let toml = include_str!("street_scenario.toml").replace("width = 640, height = 480", "width = 160, height = 120");
    let sc = Arc::new(Scenario::from_toml(&toml).unwrap().compile().unwrap());
    let mut rx = RosNode::new("/", "street_test", DOMAIN + 4).unwrap();
    const KEEP_ALL: ros2_client::QosProfile = qos::SENSOR.history(ros2_client::qos::History::KeepAll);
    let cloud = rx.subscription::<PointCloud2>("/agent0/lidar", KEEP_ALL).unwrap();
    let image = rx.subscription::<Image>("/agent0/front/image", KEEP_ALL).unwrap();
    let depth = rx.subscription::<Image>("/agent0/front/depth", KEEP_ALL).unwrap();
    let semantic = rx.subscription::<Image>("/agent0/front/semantic", KEEP_ALL).unwrap();
    let info = rx.subscription::<CameraInfo>("/agent0/front/camera_info", KEEP_ALL).unwrap();
    let npcs = rx.subscription::<MarkerArray>("/autonomousim/npcs", qos::RELIABLE).unwrap();
    let peds = rx.subscription::<MarkerArray>("/autonomousim/pedestrians", qos::RELIABLE).unwrap();
    let signals = rx.subscription::<MarkerArray>("/autonomousim/signals", qos::RELIABLE).unwrap();
    let config = BridgeConfig { domain_id: DOMAIN + 4, pacing: Pacing::Realtime(1.0), ..Default::default() };
    let mut bridge = Bridge::new(sc, config).unwrap();
    let route = rx.subscription::<Path>("/agent0/route", qos::LATCHED).unwrap();
    let tf_static = rx.subscription::<TFMessage>("/tf_static", qos::LATCHED).unwrap();
    let (mut clouds, mut images, mut depths, mut sems, mut infos) = (vec![], vec![], vec![], vec![], vec![]);
    let (mut npc, mut ped, mut sig, mut routes, mut statics) = (vec![], vec![], vec![], vec![], vec![]);
    // Until every topic has arrived (RustDDS discovery takes seconds) and 1 s more.
    let mut all_since = None;
    while all_since.is_none_or(|t0| bridge.time() < t0 + 1.0) {
        assert!(bridge.time() < 20.0, "topics missing after 20 s");
        bridge.step();
        drain(&cloud, &mut clouds);
        drain(&image, &mut images);
        drain(&depth, &mut depths);
        drain(&semantic, &mut sems);
        drain(&info, &mut infos);
        drain(&npcs, &mut npc);
        drain(&peds, &mut ped);
        drain(&signals, &mut sig);
        drain(&route, &mut routes);
        drain(&tf_static, &mut statics);
        let all = [clouds.len(), images.len(), depths.len(), sems.len(), infos.len(), npc.len(), ped.len(), sig.len()];
        if all_since.is_none() && all.iter().all(|&n| n > 0) {
            all_since = Some(bridge.time());
        }
    }
    // LiDAR: returns only, 16 bytes a point, at most the beam count, in the sensor frame.
    let c = clouds.last().expect("no point cloud");
    assert_eq!((c.header.frame_id.as_str(), c.height, c.point_step), ("agent0/lidar", 1, 16));
    assert!(c.width > 1000 && c.width <= 16 * 900 && c.data.len() == 16 * c.width as usize, "{} points", c.width);
    let fields: Vec<&str> = c.fields.iter().map(|f| f.name.as_str()).collect();
    assert_eq!(fields, ["x", "y", "z", "kind"]);
    let max = (0..c.width as usize)
        .map(|k| {
            let f = |o: usize| f32::from_le_bytes(c.data[16 * k + o..16 * k + o + 4].try_into().unwrap());
            (f(0).powi(2) + f(4).powi(2) + f(8).powi(2)).sqrt()
        })
        .fold(0.0f32, f32::max);
    assert!(max <= 40.01, "a point {max} m away");
    // Camera: at 10 Hz, in the optical frame, with its calibration.
    for (imgs, enc, bpp) in [(&images, "rgb8", 3), (&depths, "32FC1", 4), (&sems, "mono8", 1)] {
        let i = imgs.last().unwrap_or_else(|| panic!("no {enc} image"));
        assert_eq!((i.width, i.height, i.encoding.as_str(), i.step), (160, 120, enc, 160 * bpp));
        assert_eq!(i.data.len(), 160 * 120 * bpp as usize);
        assert_eq!(i.header.frame_id, "agent0/front_optical");
    }
    // Frames at the camera's 10 Hz (a loaded machine may drop some: best effort).
    let stamps: Vec<f64> = images.iter().map(|i| i.header.stamp.as_secs()).collect();
    assert!(stamps.windows(2).all(|w| w[1] > w[0]), "{stamps:?}");
    assert!(stamps.iter().all(|t| (t * 10.0 - (t * 10.0).round()).abs() < 1e-6), "{stamps:?}");
    let k = &infos.last().unwrap().k.0;
    assert!((k[0] - 80.0).abs() < 1e-9 && k[2] == 80.0 && k[5] == 60.0, "{k:?}");
    // The optical frame: z along the camera's x (forward), x along its −y (right).
    let st = &statics.last().expect("no /tf_static").transforms;
    let opt = st.iter().find(|t| t.child_frame_id == "agent0/front_optical").expect("no optical frame");
    assert_eq!(opt.header.frame_id, "agent0/front");
    let q = opt.transform.rotation;
    let q = glam::DQuat::from_xyzw(q.x, q.y, q.z, q.w);
    assert!((q * glam::DVec3::Z - glam::DVec3::X).length() < 1e-12);
    assert!((q * glam::DVec3::X + glam::DVec3::Y).length() < 1e-12);
    // Markers: 15 NPC boxes the size of a sedan, 20 pedestrians, signal heads.
    let n = npc.last().expect("no NPC markers");
    assert_eq!(n.markers.len(), 15);
    let s = &n.markers[0].scale;
    assert!((3.5..6.0).contains(&s.x) && (1.4..2.5).contains(&s.y), "NPC box {s:?}");
    assert_eq!(ped.last().expect("no pedestrian markers").markers.len(), 20);
    assert!(!sig.last().expect("no signal markers").markers.is_empty());
    // The route of the ego car (latched).
    let r = routes.last().expect("no route");
    assert!(r.poses.len() > 5 && r.header.frame_id == "map", "{} route points", r.poses.len());
}
