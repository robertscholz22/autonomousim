#!/usr/bin/env bash
# Copies the ROS 2 message definitions the bridge uses from ROS 2 Lyrical into crates/ros/msg/
# (the rosbag2 export embeds them as `ros2msg` schemas). Apache-2.0, from the ROS 2 packages.
set -euo pipefail
cd "$(dirname "$0")/../.."
MSGS="builtin_interfaces/msg/Time builtin_interfaces/msg/Duration
std_msgs/msg/Header std_msgs/msg/String std_msgs/msg/UInt32 std_msgs/msg/ColorRGBA
std_msgs/msg/Float32MultiArray std_msgs/msg/MultiArrayLayout std_msgs/msg/MultiArrayDimension
geometry_msgs/msg/Vector3 geometry_msgs/msg/Point geometry_msgs/msg/Quaternion geometry_msgs/msg/Pose
geometry_msgs/msg/PoseStamped geometry_msgs/msg/PoseWithCovariance geometry_msgs/msg/Twist
geometry_msgs/msg/TwistWithCovariance geometry_msgs/msg/Transform geometry_msgs/msg/TransformStamped
nav_msgs/msg/Odometry nav_msgs/msg/Path
sensor_msgs/msg/Imu sensor_msgs/msg/NavSatFix sensor_msgs/msg/NavSatStatus sensor_msgs/msg/FluidPressure
sensor_msgs/msg/MagneticField sensor_msgs/msg/Range sensor_msgs/msg/PointCloud2 sensor_msgs/msg/PointField
sensor_msgs/msg/Image sensor_msgs/msg/CompressedImage sensor_msgs/msg/JointState
tf2_msgs/msg/TFMessage rosgraph_msgs/msg/Clock
visualization_msgs/msg/Marker visualization_msgs/msg/MarkerArray visualization_msgs/msg/MeshFile
visualization_msgs/msg/UVCoordinate
std_srvs/srv/Trigger std_srvs/srv/SetBool"
rm -rf crates/ros/msg && mkdir -p crates/ros/msg
for m in $MSGS; do
    mkdir -p "crates/ros/msg/${m%%/*}"
done
tools/ros/run.sh bash -c 'for m in '"$(echo $MSGS)"'; do pkg=${m%%/*}; name=${m##*/}; kind=${m#*/}; kind=${kind%%/*}; cp /opt/ros/lyrical/share/$pkg/$kind/$name.$kind crates/ros/msg/$pkg/; done'
