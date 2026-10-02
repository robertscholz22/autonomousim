//! The `.msg` definitions of the bridge's messages (`crates/ros/msg/`, copied from ROS 2
//! Lyrical by `tools/ros/fetch_msgs.sh`) and the `ros2msg` schemas rosbag2 writes for them.

use std::collections::BTreeSet;

const DEFINITIONS: &[(&str, &str)] = &[
    ("builtin_interfaces/Duration", include_str!("../msg/builtin_interfaces/Duration.msg")),
    ("builtin_interfaces/Time", include_str!("../msg/builtin_interfaces/Time.msg")),
    ("geometry_msgs/Point", include_str!("../msg/geometry_msgs/Point.msg")),
    ("geometry_msgs/Pose", include_str!("../msg/geometry_msgs/Pose.msg")),
    ("geometry_msgs/PoseStamped", include_str!("../msg/geometry_msgs/PoseStamped.msg")),
    ("geometry_msgs/PoseWithCovariance", include_str!("../msg/geometry_msgs/PoseWithCovariance.msg")),
    ("geometry_msgs/Quaternion", include_str!("../msg/geometry_msgs/Quaternion.msg")),
    ("geometry_msgs/Transform", include_str!("../msg/geometry_msgs/Transform.msg")),
    ("geometry_msgs/TransformStamped", include_str!("../msg/geometry_msgs/TransformStamped.msg")),
    ("geometry_msgs/Twist", include_str!("../msg/geometry_msgs/Twist.msg")),
    ("geometry_msgs/TwistWithCovariance", include_str!("../msg/geometry_msgs/TwistWithCovariance.msg")),
    ("geometry_msgs/Vector3", include_str!("../msg/geometry_msgs/Vector3.msg")),
    ("nav_msgs/Odometry", include_str!("../msg/nav_msgs/Odometry.msg")),
    ("nav_msgs/Path", include_str!("../msg/nav_msgs/Path.msg")),
    ("rosgraph_msgs/Clock", include_str!("../msg/rosgraph_msgs/Clock.msg")),
    ("sensor_msgs/CameraInfo", include_str!("../msg/sensor_msgs/CameraInfo.msg")),
    ("sensor_msgs/CompressedImage", include_str!("../msg/sensor_msgs/CompressedImage.msg")),
    ("sensor_msgs/FluidPressure", include_str!("../msg/sensor_msgs/FluidPressure.msg")),
    ("sensor_msgs/Image", include_str!("../msg/sensor_msgs/Image.msg")),
    ("sensor_msgs/Imu", include_str!("../msg/sensor_msgs/Imu.msg")),
    ("sensor_msgs/JointState", include_str!("../msg/sensor_msgs/JointState.msg")),
    ("sensor_msgs/MagneticField", include_str!("../msg/sensor_msgs/MagneticField.msg")),
    ("sensor_msgs/NavSatFix", include_str!("../msg/sensor_msgs/NavSatFix.msg")),
    ("sensor_msgs/NavSatStatus", include_str!("../msg/sensor_msgs/NavSatStatus.msg")),
    ("sensor_msgs/PointCloud2", include_str!("../msg/sensor_msgs/PointCloud2.msg")),
    ("sensor_msgs/PointField", include_str!("../msg/sensor_msgs/PointField.msg")),
    ("sensor_msgs/Range", include_str!("../msg/sensor_msgs/Range.msg")),
    ("sensor_msgs/RegionOfInterest", include_str!("../msg/sensor_msgs/RegionOfInterest.msg")),
    ("std_msgs/ColorRGBA", include_str!("../msg/std_msgs/ColorRGBA.msg")),
    ("std_msgs/Float32MultiArray", include_str!("../msg/std_msgs/Float32MultiArray.msg")),
    ("std_msgs/Header", include_str!("../msg/std_msgs/Header.msg")),
    ("std_msgs/MultiArrayDimension", include_str!("../msg/std_msgs/MultiArrayDimension.msg")),
    ("std_msgs/MultiArrayLayout", include_str!("../msg/std_msgs/MultiArrayLayout.msg")),
    ("std_msgs/String", include_str!("../msg/std_msgs/String.msg")),
    ("std_msgs/UInt32", include_str!("../msg/std_msgs/UInt32.msg")),
    ("tf2_msgs/TFMessage", include_str!("../msg/tf2_msgs/TFMessage.msg")),
    ("visualization_msgs/Marker", include_str!("../msg/visualization_msgs/Marker.msg")),
    ("visualization_msgs/MarkerArray", include_str!("../msg/visualization_msgs/MarkerArray.msg")),
    ("visualization_msgs/MeshFile", include_str!("../msg/visualization_msgs/MeshFile.msg")),
    ("visualization_msgs/UVCoordinate", include_str!("../msg/visualization_msgs/UVCoordinate.msg")),
];

const PRIMITIVES: &[&str] = &[
    "bool", "byte", "char", "float32", "float64", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64",
    "uint64", "string", "wstring",
];

/// The definition (`.msg` text) of a type (`package/Name`), if the crate has it.
pub fn definition(ty: &str) -> Option<&'static str> {
    DEFINITIONS.iter().find(|(t, _)| *t == ty).map(|(_, d)| *d)
}

/// The message types a definition of package `package` uses (`package/Name`), sorted.
fn dependencies(package: &str, text: &str) -> BTreeSet<String> {
    text.lines()
        .filter_map(|line| line.split('#').next()?.split_whitespace().next())
        .filter_map(|ty| {
            let base = ty.split(['[', '<']).next()?;
            if PRIMITIVES.contains(&base) {
                return None;
            }
            Some(match base.split('/').collect::<Vec<_>>()[..] {
                [p, "msg", n] | [p, n] => format!("{p}/{n}"),
                ["Header"] => "std_msgs/Header".to_string(),
                _ => format!("{package}/{base}"),
            })
        })
        .collect()
}

/// The `ros2msg` schema of a type (`package/Name`), as rosbag2 writes it: its definition,
/// then each type it uses, depth first (each type's dependencies in name order) and once,
/// after a line of 80 `=` and `MSG: package/Name`.
pub fn schema(ty: &str) -> Result<String, String> {
    fn add(ty: &str, out: &mut String, seen: &mut BTreeSet<String>) -> Result<(), String> {
        let text = definition(ty).ok_or_else(|| format!("no definition of {ty}"))?;
        let package = ty.split('/').next().unwrap_or_default();
        for dep in dependencies(package, text) {
            if seen.insert(dep.clone()) {
                out.push_str(&format!(
                    "
{}
MSG: {dep}
",
                    "=".repeat(80)
                ));
                out.push_str(definition(&dep).ok_or_else(|| format!("no definition of {dep} (used by {ty})"))?);
                add(&dep, out, seen)?;
            }
        }
        Ok(())
    }
    let mut out = definition(ty).ok_or_else(|| format!("no definition of {ty}"))?.to_string();
    add(ty, &mut out, &mut BTreeSet::from([ty.to_string()]))?;
    Ok(out)
}
