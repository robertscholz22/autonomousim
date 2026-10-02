//! The standard ROS 2 messages the bridge uses, as serde structs whose field order matches the
//! `.msg` definitions (`crates/ros/msg/`, from ROS 2 Lyrical), so their CDR encoding is ROS's.
//! Field names are ROS's too (the JSON fixtures of `fixtures/ros/cdr.json` decode into them).

use serde::de::{DeserializeOwned, SeqAccess, Visitor};
use serde::ser::SerializeTuple;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// A message type: its ROS name, `package/Name` (service parts: `package/Name_Request`).
pub trait RosMessage: Serialize + DeserializeOwned + Send + 'static {
    const TYPE: &'static str;
    /// A fixed-size type that is not a multiple of 4 bytes long: rmw serializes it with the
    /// trailing padding of its C struct, to a multiple of 4 ([`to_cdr`]).
    const PADDED: bool = false;
}

macro_rules! ros_message {
    ($($t:ty => $name:literal $(, $padded:ident)?);* $(;)?) => {$(
        impl RosMessage for $t {
            const TYPE: &'static str = $name;
            $(const PADDED: bool = ros_message!(@$padded);)?
        }
        impl ros2_client::Message for $t {}
    )*};
    (@padded) => { true };
}

/// A fixed-size `float64[N]` array (CDR: no length prefix). serde's own arrays stop at 32.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Array<const N: usize>(pub [f64; N]);

impl<const N: usize> Default for Array<N> {
    fn default() -> Self {
        Self([0.0; N])
    }
}

impl<const N: usize> Serialize for Array<N> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut t = s.serialize_tuple(N)?;
        for x in &self.0 {
            t.serialize_element(x)?;
        }
        t.end()
    }
}

impl<'de, const N: usize> Deserialize<'de> for Array<N> {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V<const N: usize>;
        impl<'de, const N: usize> Visitor<'de> for V<N> {
            type Value = Array<N>;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(f, "{N} float64 values")
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Array<N>, A::Error> {
                let mut a = [0.0; N];
                for (i, x) in a.iter_mut().enumerate() {
                    *x = seq.next_element()?.ok_or_else(|| serde::de::Error::invalid_length(i, &self))?;
                }
                Ok(Array(a))
            }
        }
        d.deserialize_tuple(N, V::<N>)
    }
}

pub mod builtin_interfaces {
    use super::*;

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
    pub struct Time {
        pub sec: i32,
        pub nanosec: u32,
    }

    impl Time {
        /// From seconds (sim time); negative times clamp to zero.
        pub fn from_secs(t: f64) -> Self {
            let ns = (t.max(0.0) * 1e9).round() as u64;
            Self { sec: (ns / 1_000_000_000) as i32, nanosec: (ns % 1_000_000_000) as u32 }
        }

        pub fn as_secs(&self) -> f64 {
            self.sec as f64 + self.nanosec as f64 * 1e-9
        }
    }

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
    pub struct Duration {
        pub sec: i32,
        pub nanosec: u32,
    }
}

pub mod std_msgs {
    use super::builtin_interfaces::Time;
    use super::*;

    #[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
    pub struct Header {
        pub stamp: Time,
        pub frame_id: String,
    }

    #[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
    pub struct StringMsg {
        pub data: String,
    }

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
    pub struct UInt32 {
        pub data: u32,
    }

    #[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
    pub struct ColorRGBA {
        pub r: f32,
        pub g: f32,
        pub b: f32,
        pub a: f32,
    }

    #[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
    pub struct MultiArrayDimension {
        pub label: String,
        pub size: u32,
        pub stride: u32,
    }

    #[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
    pub struct MultiArrayLayout {
        pub dim: Vec<MultiArrayDimension>,
        pub data_offset: u32,
    }

    #[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
    pub struct Float32MultiArray {
        pub layout: MultiArrayLayout,
        pub data: Vec<f32>,
    }
}

pub mod geometry_msgs {
    use super::std_msgs::Header;
    use super::*;

    #[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
    pub struct Vector3 {
        pub x: f64,
        pub y: f64,
        pub z: f64,
    }

    #[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
    pub struct Point {
        pub x: f64,
        pub y: f64,
        pub z: f64,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    pub struct Quaternion {
        pub x: f64,
        pub y: f64,
        pub z: f64,
        pub w: f64,
    }

    impl Default for Quaternion {
        fn default() -> Self {
            Self { x: 0.0, y: 0.0, z: 0.0, w: 1.0 }
        }
    }

    #[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
    pub struct Pose {
        pub position: Point,
        pub orientation: Quaternion,
    }

    #[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
    pub struct PoseStamped {
        pub header: Header,
        pub pose: Pose,
    }

    #[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
    pub struct PoseWithCovariance {
        pub pose: Pose,
        pub covariance: Array<36>,
    }

    #[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
    pub struct Twist {
        pub linear: Vector3,
        pub angular: Vector3,
    }

    #[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
    pub struct TwistWithCovariance {
        pub twist: Twist,
        pub covariance: Array<36>,
    }

    #[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
    pub struct Transform {
        pub translation: Vector3,
        pub rotation: Quaternion,
    }

    #[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
    pub struct TransformStamped {
        pub header: Header,
        pub child_frame_id: String,
        pub transform: Transform,
    }
}

pub mod nav_msgs {
    use super::geometry_msgs::{PoseStamped, PoseWithCovariance, TwistWithCovariance};
    use super::std_msgs::Header;
    use super::*;

    #[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
    pub struct Odometry {
        pub header: Header,
        pub child_frame_id: String,
        pub pose: PoseWithCovariance,
        pub twist: TwistWithCovariance,
    }

    #[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
    pub struct Path {
        pub header: Header,
        pub poses: Vec<PoseStamped>,
    }
}

pub mod sensor_msgs {
    use super::geometry_msgs::{Quaternion, Vector3};
    use super::std_msgs::Header;
    use super::*;

    #[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
    pub struct Imu {
        pub header: Header,
        pub orientation: Quaternion,
        pub orientation_covariance: Array<9>,
        pub angular_velocity: Vector3,
        pub angular_velocity_covariance: Array<9>,
        pub linear_acceleration: Vector3,
        pub linear_acceleration_covariance: Array<9>,
    }

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
    pub struct NavSatStatus {
        pub status: i8,
        pub service: u16,
    }

    impl NavSatStatus {
        pub const STATUS_NO_FIX: i8 = -1;
        pub const STATUS_FIX: i8 = 0;
        pub const SERVICE_GPS: u16 = 1;
    }

    #[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
    pub struct NavSatFix {
        pub header: Header,
        pub status: NavSatStatus,
        pub latitude: f64,
        pub longitude: f64,
        pub altitude: f64,
        pub position_covariance: Array<9>,
        pub position_covariance_type: u8,
    }

    impl NavSatFix {
        pub const COVARIANCE_TYPE_DIAGONAL_KNOWN: u8 = 2;
    }

    #[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
    pub struct FluidPressure {
        pub header: Header,
        pub fluid_pressure: f64,
        pub variance: f64,
    }

    #[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
    pub struct MagneticField {
        pub header: Header,
        pub magnetic_field: Vector3,
        pub magnetic_field_covariance: Array<9>,
    }

    #[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
    pub struct Range {
        pub header: Header,
        pub radiation_type: u8,
        pub field_of_view: f32,
        pub min_range: f32,
        pub max_range: f32,
        pub range: f32,
        pub variance: f32,
    }

    impl Range {
        pub const INFRARED: u8 = 1;
    }

    #[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
    pub struct PointField {
        pub name: String,
        pub offset: u32,
        pub datatype: u8,
        pub count: u32,
    }

    impl PointField {
        pub const UINT8: u8 = 2;
        pub const UINT16: u8 = 4;
        pub const FLOAT32: u8 = 7;
    }

    #[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
    pub struct PointCloud2 {
        pub header: Header,
        pub height: u32,
        pub width: u32,
        pub fields: Vec<PointField>,
        pub is_bigendian: bool,
        pub point_step: u32,
        pub row_step: u32,
        #[serde(with = "serde_bytes")]
        pub data: Vec<u8>,
        pub is_dense: bool,
    }

    #[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
    pub struct Image {
        pub header: Header,
        pub height: u32,
        pub width: u32,
        pub encoding: String,
        pub is_bigendian: u8,
        pub step: u32,
        #[serde(with = "serde_bytes")]
        pub data: Vec<u8>,
    }

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
    pub struct RegionOfInterest {
        pub x_offset: u32,
        pub y_offset: u32,
        pub height: u32,
        pub width: u32,
        pub do_rectify: bool,
    }

    /// Pinhole calibration (`k`: intrinsics, `r`: rectification, `p`: projection; row-major).
    #[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
    pub struct CameraInfo {
        pub header: Header,
        pub height: u32,
        pub width: u32,
        pub distortion_model: String,
        pub d: Vec<f64>,
        pub k: Array<9>,
        pub r: Array<9>,
        pub p: Array<12>,
        pub binning_x: u32,
        pub binning_y: u32,
        pub roi: RegionOfInterest,
    }

    #[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
    pub struct CompressedImage {
        pub header: Header,
        pub format: String,
        #[serde(with = "serde_bytes")]
        pub data: Vec<u8>,
    }

    #[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
    pub struct JointState {
        pub header: Header,
        pub name: Vec<String>,
        pub position: Vec<f64>,
        pub velocity: Vec<f64>,
        pub effort: Vec<f64>,
    }
}

pub mod tf2_msgs {
    use super::geometry_msgs::TransformStamped;
    use super::*;

    #[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
    pub struct TFMessage {
        pub transforms: Vec<TransformStamped>,
    }
}

pub mod rosgraph_msgs {
    use super::builtin_interfaces::Time;
    use super::*;

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
    pub struct Clock {
        pub clock: Time,
    }
}

pub mod visualization_msgs {
    use super::builtin_interfaces::Duration;
    use super::geometry_msgs::{Point, Pose, Vector3};
    use super::sensor_msgs::CompressedImage;
    use super::std_msgs::{ColorRGBA, Header};
    use super::*;

    #[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
    pub struct UVCoordinate {
        pub u: f32,
        pub v: f32,
    }

    #[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
    pub struct MeshFile {
        pub filename: String,
        #[serde(with = "serde_bytes")]
        pub data: Vec<u8>,
    }

    #[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
    pub struct Marker {
        pub header: Header,
        pub ns: String,
        pub id: i32,
        #[serde(rename = "type")]
        pub kind: i32,
        pub action: i32,
        pub pose: Pose,
        pub scale: Vector3,
        pub color: ColorRGBA,
        pub lifetime: Duration,
        pub frame_locked: bool,
        pub points: Vec<Point>,
        pub colors: Vec<ColorRGBA>,
        pub texture_resource: String,
        pub texture: CompressedImage,
        pub uv_coordinates: Vec<UVCoordinate>,
        pub text: String,
        pub mesh_resource: String,
        pub mesh_file: MeshFile,
        pub mesh_use_embedded_materials: bool,
    }

    impl Marker {
        pub const ARROW: i32 = 0;
        pub const CUBE: i32 = 1;
        pub const SPHERE: i32 = 2;
        pub const CYLINDER: i32 = 3;
        pub const LINE_STRIP: i32 = 4;
        pub const CUBE_LIST: i32 = 6;
        pub const SPHERE_LIST: i32 = 7;
        pub const TEXT_VIEW_FACING: i32 = 9;
        pub const ADD: i32 = 0;
        pub const DELETE: i32 = 2;
        pub const DELETEALL: i32 = 3;
    }

    #[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
    pub struct MarkerArray {
        pub markers: Vec<Marker>,
    }
}

pub mod std_srvs {
    use super::*;

    /// An empty request (IDL structs need a member: ROS adds a dummy byte).
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
    pub struct TriggerRequest {
        #[serde(default)]
        pub structure_needs_at_least_one_member: u8,
    }

    #[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
    pub struct TriggerResponse {
        pub success: bool,
        pub message: String,
    }

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
    pub struct SetBoolRequest {
        pub data: bool,
    }

    #[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
    pub struct SetBoolResponse {
        pub success: bool,
        pub message: String,
    }
}

ros_message! {
    builtin_interfaces::Time => "builtin_interfaces/Time";
    builtin_interfaces::Duration => "builtin_interfaces/Duration";
    std_msgs::Header => "std_msgs/Header";
    std_msgs::StringMsg => "std_msgs/String";
    std_msgs::UInt32 => "std_msgs/UInt32";
    std_msgs::ColorRGBA => "std_msgs/ColorRGBA";
    std_msgs::Float32MultiArray => "std_msgs/Float32MultiArray";
    geometry_msgs::Vector3 => "geometry_msgs/Vector3";
    geometry_msgs::Point => "geometry_msgs/Point";
    geometry_msgs::Quaternion => "geometry_msgs/Quaternion";
    geometry_msgs::Pose => "geometry_msgs/Pose";
    geometry_msgs::PoseStamped => "geometry_msgs/PoseStamped";
    geometry_msgs::PoseWithCovariance => "geometry_msgs/PoseWithCovariance";
    geometry_msgs::Twist => "geometry_msgs/Twist";
    geometry_msgs::TwistWithCovariance => "geometry_msgs/TwistWithCovariance";
    geometry_msgs::Transform => "geometry_msgs/Transform";
    geometry_msgs::TransformStamped => "geometry_msgs/TransformStamped";
    nav_msgs::Odometry => "nav_msgs/Odometry";
    nav_msgs::Path => "nav_msgs/Path";
    sensor_msgs::Imu => "sensor_msgs/Imu";
    sensor_msgs::NavSatFix => "sensor_msgs/NavSatFix";
    sensor_msgs::FluidPressure => "sensor_msgs/FluidPressure";
    sensor_msgs::MagneticField => "sensor_msgs/MagneticField";
    sensor_msgs::Range => "sensor_msgs/Range";
    sensor_msgs::PointCloud2 => "sensor_msgs/PointCloud2";
    sensor_msgs::Image => "sensor_msgs/Image";
    sensor_msgs::RegionOfInterest => "sensor_msgs/RegionOfInterest", padded;
    sensor_msgs::CameraInfo => "sensor_msgs/CameraInfo";
    sensor_msgs::CompressedImage => "sensor_msgs/CompressedImage";
    sensor_msgs::JointState => "sensor_msgs/JointState";
    tf2_msgs::TFMessage => "tf2_msgs/TFMessage";
    rosgraph_msgs::Clock => "rosgraph_msgs/Clock";
    visualization_msgs::Marker => "visualization_msgs/Marker";
    visualization_msgs::MarkerArray => "visualization_msgs/MarkerArray";
    std_srvs::TriggerRequest => "std_srvs/Trigger_Request";
    std_srvs::TriggerResponse => "std_srvs/Trigger_Response";
    std_srvs::SetBoolRequest => "std_srvs/SetBool_Request";
    std_srvs::SetBoolResponse => "std_srvs/SetBool_Response";
}

/// The CDR encapsulation header for little-endian plain CDR (as rmw serializes messages).
pub const CDR_LE: [u8; 4] = [0x00, 0x01, 0x00, 0x00];

/// A message serialized as rmw does (`rclpy.serialization.serialize_message`, rosbag2): the
/// encapsulation header and little-endian CDR, padded to at least 4 bytes of payload.
pub fn to_cdr<M: RosMessage>(msg: &M) -> Vec<u8> {
    let mut out = CDR_LE.to_vec();
    cdr_encoding::to_writer::<_, byteorder::LittleEndian, _>(&mut out, msg).expect("CDR encoding into a Vec");
    if out.len() < 8 {
        out.resize(8, 0);
    }
    if M::PADDED {
        out.resize(out.len().next_multiple_of(4), 0);
    }
    out
}

/// The inverse of [`to_cdr`] (little-endian payloads only).
pub fn from_cdr<M: DeserializeOwned>(bytes: &[u8]) -> Result<M, String> {
    let (head, payload) = bytes.split_at_checked(4).ok_or("shorter than the encapsulation header")?;
    if head != CDR_LE {
        return Err(format!("unsupported encapsulation {head:02x?}"));
    }
    cdr_encoding::from_bytes::<M, byteorder::LittleEndian>(payload).map(|(m, _)| m).map_err(|e| e.to_string())
}
