//! ROS 2 bridge for autonomousim: DDS through `ros2-client` (RustDDS), standard message types
//! only, so stock ROS 2 tools (`ros2 topic echo`, rosbag2) work without custom packages.
//!
//! - [`bridge`]: runs a simulation as a ROS 2 node (clock, transforms, odometry, sensors).
//! - [`bag`]: exports recordings as rosbag2 bags.
//! - [`markers`]: NPCs, pedestrians, signals and routes as markers and paths.
//! - [`definitions`]: the `.msg` definitions (rosbag2 schemas).
//! - [`msgs`]: the messages as serde structs with ROS's field order (CDR as rmw writes it).
//! - [`node`]: a DDS node with typed publishers and subscriptions, QoS profiles.

pub mod bag;
pub mod bridge;
pub mod definitions;
pub mod markers;
pub mod msgs;
pub mod node;
