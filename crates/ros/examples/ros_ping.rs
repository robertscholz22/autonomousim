//! Interop check: publishes `nav_msgs/Odometry` on `/autonomousim_test/odom` at 10 Hz and prints
//! the `geometry_msgs/Twist` messages received on `/autonomousim_test/cmd_vel`, for `seconds`.
//!   cargo run -p autonomousim-ros --example ros_ping -- <domain> <seconds>

use autonomousim_ros::msgs::{builtin_interfaces::Time, geometry_msgs::Twist, nav_msgs::Odometry};
use autonomousim_ros::node::{RosNode, qos};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let domain = args.get(1).map_or(Ok(0), |s| s.parse())?;
    let seconds: f64 = args.get(2).map_or(Ok(10.0), |s| s.parse())?;
    let mut node = RosNode::new("/", "autonomousim_ping", domain)?;
    let odom = node.publisher::<Odometry>("/autonomousim_test/odom", qos::RELIABLE)?;
    let cmd = node.subscription::<Twist>("/autonomousim_test/cmd_vel", qos::RELIABLE)?;
    let start = std::time::Instant::now();
    let mut k = 0u32;
    while start.elapsed().as_secs_f64() < seconds {
        let mut msg = Odometry::default();
        msg.header.stamp = Time::from_secs(k as f64 * 0.1);
        msg.header.frame_id = "map".into();
        msg.child_frame_id = "agent0/base_link".into();
        msg.pose.pose.position.x = 1.25 + k as f64;
        msg.twist.twist.linear.x = 2.5;
        odom.publish(msg).map_err(|e| anyhow::anyhow!("publish: {e:?}"))?;
        while let Some((t, _)) = cmd.take().map_err(|e| anyhow::anyhow!("take: {e:?}"))? {
            println!("cmd_vel linear.x {} angular.z {}", t.linear.x, t.angular.z);
        }
        k += 1;
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    println!("published {k}");
    Ok(())
}
