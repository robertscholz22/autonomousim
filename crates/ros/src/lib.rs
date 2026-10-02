//! ROS 2 bridge for autonomousim: DDS through `ros2-client` (RustDDS), standard message types
//! only, so stock ROS 2 tools (`ros2 topic echo`, rosbag2) work without custom packages.
//!
//! - [`msgs`]: the messages as serde structs with ROS's field order (CDR as rmw writes it).
//! - [`node`]: a DDS node with typed publishers and subscriptions, QoS profiles.

pub mod msgs;
pub mod node;
