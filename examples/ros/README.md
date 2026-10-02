# ROS 2 example

A drone flown by a ROS 2 node. The bridge (`autonomousim-ros`) runs the simulation on the
host and talks DDS to ROS 2 Lyrical in a Docker container. In the container, an rclpy node
flies the drone through waypoints while `ros2 bag record` records the run.

| File | What it is |
|---|---|
| `drone.toml` | The scenario: an `iris_like` quadrotor with IMU, GPS, barometer and a 16-ring LiDAR, over a 5 × 5 field of 5 m pillars, flown in `velocity` mode |
| `waypoints.py` | The controller (rclpy): reads `/agent0/odom` and publishes `/agent0/cmd_vel` toward the next waypoint. It climbs to 8 m, flies a 24 m square over the pillars, comes back and descends to 1 m |
| `demo.sh` | Runs all of it: builds the bridge, starts the recorder, the bridge (60 s of simulated time in real time) and the controller, then prints `ros2 bag info` |

## Requirements

- Docker, with your user allowed to run it, and the image: `docker pull ros:lyrical-ros-base`.
  [`tools/ros/run.sh`](../../tools/ros/run.sh) runs commands in that image with host networking,
  your user id and the repository mounted at `/ws`.
- A receive buffer of at least 4 MiB for Fast DDS, which matters for camera images. Check it
  with `sysctl net.core.rmem_max`, and raise it with `sudo sysctl -w net.core.rmem_max=4194304`
  if needed.
- The Rust toolchain (see the main README). Python is not needed on the host.

## Run it

```bash
examples/ros/demo.sh            # writes runs/ros_example/
```

The first run builds the bridge (about 2 minutes). Expected output: one line per waypoint,
about 8.5 s of simulated time apart, then the bag.

```text
[INFO] [...] [waypoint_follower]: start at (-31.0, -31.0, 1.0)
[INFO] [...] [waypoint_follower]: waypoint 0 reached at t = 5.94 s
...
[INFO] [...] [waypoint_follower]: waypoint 5 reached at t = 45.02 s
{"reached": [[0, 5.94], ...], "waypoints": 6, ...}

Files:             <date>.mcap
Duration:          58.660000000s
Topic information: Topic: /agent0/baro | Type: sensor_msgs/msg/FluidPressure | Count: 2904 | ...
                   Topic: /agent0/lidar | Type: sensor_msgs/msg/PointCloud2 | Count: 580 | ...
                   Topic: /agent0/odom | Type: nav_msgs/msg/Odometry | Count: 2894 | ...
                   Topic: /clock | Type: rosgraph_msgs/msg/Clock | Count: 2933 | ...
                   ...
```

The bag is in `runs/ros_example/bag/`. The logs and the controller's report are next to it.
The bag's times are simulated time: the recorder runs with `--use-sim-time`, so `ros2 bag info`
dates the run in 1970. Set `ROS_DOMAIN_ID` to keep the demo apart from other ROS traffic on
your network.

## Step by step

The same as `demo.sh`, in three terminals:

```bash
# 1. The bridge on the host: real time, publishes /clock, /tf, odometry and sensors.
cargo run -p autonomousim-ros --release -- run --scenario examples/ros/drone.toml

# 2. The controller in the container (exits when the last waypoint is reached).
tools/ros/run.sh python3 examples/ros/waypoints.py

# 3. Anything else ROS 2 offers, for example:
tools/ros/run.sh ros2 topic hz /agent0/odom
tools/ros/run.sh ros2 run tf2_ros tf2_echo map agent0/base_link
tools/ros/run.sh ros2 bag record -s mcap --use-sim-time --all-topics -o runs/my_bag
tools/ros/run.sh ros2 bag play runs/my_bag        # with the bridge stopped
```

Useful bridge options (`autonomousim-ros run --help` lists them all):

- `--policy runs/<run>/policy.json`: a trained policy flies its agents instead.
- `--lockstep 1.0`: the simulation waits for every commanded agent's command after each
  `/clock` tick. A slow controller then cannot fall behind.
- `--fast`: run faster than real time.
- `--duration 60`: stop after this much simulated time.

`/autonomousim/meta` (a latched JSON string) lists every agent with its frames and topics.
The services `/autonomousim/reset` (`std_srvs/Trigger`) and `/autonomousim/pause`
(`std_srvs/SetBool`) control the run.

## Recordings as bags

A recording from the Python API (`examples/eval_record.py`, or `attach_recorder`) can be
exported as a rosbag2 bag. The bag has the bridge's topic names and types:

```bash
cargo run -p autonomousim-ros --release -- bag recordings/<run>.mcap -o recordings/<run>_bag
tools/ros/run.sh ros2 bag info recordings/<run>_bag
```

`make test-ros` runs the interop tests in the container, this example included.
