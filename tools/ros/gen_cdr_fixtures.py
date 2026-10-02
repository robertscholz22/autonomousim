"""CDR reference encodings of the ROS 2 messages the bridge uses (fixtures/ros/cdr.json).

Runs inside ROS 2 Lyrical (tools/ros/run.sh python3 tools/ros/gen_cdr_fixtures.py): fills one
message of each type with distinct values, serializes it with rclpy (the encapsulation header
included) and writes the field values (JSON, ROS field names) with the bytes as hex. The Rust
tests decode the values into their own structs, encode them and compare byte for byte.
"""

import array
import importlib
import json
import pathlib

from rclpy.serialization import serialize_message

TYPES = [
    "builtin_interfaces/msg/Time",
    "builtin_interfaces/msg/Duration",
    "std_msgs/msg/Header",
    "std_msgs/msg/String",
    "std_msgs/msg/UInt32",
    "std_msgs/msg/ColorRGBA",
    "std_msgs/msg/Float32MultiArray",
    "geometry_msgs/msg/Vector3",
    "geometry_msgs/msg/Point",
    "geometry_msgs/msg/Quaternion",
    "geometry_msgs/msg/Pose",
    "geometry_msgs/msg/PoseStamped",
    "geometry_msgs/msg/PoseWithCovariance",
    "geometry_msgs/msg/Twist",
    "geometry_msgs/msg/TwistWithCovariance",
    "geometry_msgs/msg/Transform",
    "geometry_msgs/msg/TransformStamped",
    "nav_msgs/msg/Odometry",
    "nav_msgs/msg/Path",
    "sensor_msgs/msg/Imu",
    "sensor_msgs/msg/NavSatFix",
    "sensor_msgs/msg/FluidPressure",
    "sensor_msgs/msg/MagneticField",
    "sensor_msgs/msg/Range",
    "sensor_msgs/msg/PointCloud2",
    "sensor_msgs/msg/Image",
    "sensor_msgs/msg/RegionOfInterest",
    "sensor_msgs/msg/CameraInfo",
    "sensor_msgs/msg/CompressedImage",
    "sensor_msgs/msg/JointState",
    "tf2_msgs/msg/TFMessage",
    "rosgraph_msgs/msg/Clock",
    "visualization_msgs/msg/Marker",
    "visualization_msgs/msg/MarkerArray",
    "std_srvs/srv/Trigger_Request",
    "std_srvs/srv/Trigger_Response",
    "std_srvs/srv/SetBool_Request",
    "std_srvs/srv/SetBool_Response",
]


def message_class(name):
    pkg, kind, typ = name.split("/")
    return getattr(importlib.import_module(f"{pkg}.{kind}"), typ)


class Counter:
    """Distinct, exactly representable values (k/8), so float32 fields round-trip."""

    def __init__(self):
        self.k = 0

    def next(self):
        self.k += 1
        return self.k


def fill(msg, c):
    for field, typ in msg.get_fields_and_field_types().items():
        setattr(msg, field, value(typ, c, getattr(msg, field)))
    return msg


def scalar(typ, c):
    k = c.next()
    if typ in ("float", "double"):
        return k / 8.0 - 3.0
    if typ == "boolean":
        return k % 2 == 1
    if typ == "string":
        return f"s{k}" * (k % 3 + 1)
    if typ in ("octet", "uint8", "char"):
        return k % 256
    if typ == "int8":
        return k % 100 - 50
    if typ.startswith("int"):
        return k * 7 - 100
    if typ.startswith("uint"):
        return k * 13
    raise ValueError(typ)


def value(typ, c, current):
    if typ.startswith("sequence<"):
        inner = typ[len("sequence<") : -1].split(",")[0]
        n = 2 + c.next() % 3
        if "/" in inner:
            cls = message_class(inner.replace("/", "/msg/"))
            return [fill(cls(), c) for _ in range(n)]
        return [scalar(inner, c) for _ in range(n)]
    if "[" in typ and typ.endswith("]"):
        inner, n = typ[:-1].split("[")
        return [scalar(inner, c) for _ in range(int(n))]
    if "/" in typ:
        return fill(type(current)(), c)
    return scalar(typ, c)


def plain(v):
    """JSON form: nested dicts with ROS field names, lists for arrays and sequences."""
    if hasattr(v, "get_fields_and_field_types"):
        return {f: plain(getattr(v, f)) for f in v.get_fields_and_field_types()}
    if isinstance(v, (list, tuple, array.array)) or type(v).__name__ == "ndarray":
        return [plain(x) for x in v]
    if isinstance(v, bytes):
        return list(v)
    if hasattr(v, "item"):
        return v.item()
    return v


def main():
    out = {}
    for name in TYPES:
        msg = fill(message_class(name)(), Counter())
        out[name.replace("/msg/", "/").replace("/srv/", "/")] = {
            "value": plain(msg),
            "cdr": serialize_message(msg).hex(),
        }
    path = pathlib.Path("fixtures/ros/cdr.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {path}: {len(out)} message types")


if __name__ == "__main__":
    main()
