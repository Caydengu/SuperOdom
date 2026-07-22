# Minimal ROS 2 Humble SuperOdometry container

This is the TML offboard SuperOdometry path for the Unitree G1 Mid-360. It
builds a pinned, CPU-only ROS 2 Humble image on Oslo while leaving the robot's
ROS 2 Foxy installation unchanged.

Current status: **offline replay candidate validated on 2026-07-22**. The image
builds, all required packages resolve, all shared libraries load, and the
transferred known-good bag produces `/state_estimation` with close parity to
the legacy offboard pipeline. The Foxy-to-Humble live DDS boundary and live
latency have not yet been validated.

## Operating boundary

The intended first deployment is:

```text
G1 (Foxy)                                  Oslo (Humble container)
Mid-360 acquisition and timestamps  --->  SuperOdometry
hardware watchdogs and safety             live diagnostics and recording
```

The image does not require a GPU, NVIDIA Container Runtime, X11, a desktop ROS
install, `--privileged`, or host IPC. The host wrapper uses host networking,
drops all Linux capabilities, enables `no-new-privileges`, runs as the caller,
mounts inputs read-only, and writes only to the selected output directory.

The legacy drop at `/move/u/caydengu/superodom_legacy_drop` is reference
evidence only. Do not edit it or write replay outputs into it. Its transferred
files are covered by `MANIFEST.sha256`.

## Pinned inputs

| Component | Revision |
| --- | --- |
| ROS base | `ros:humble-ros-base-jammy@sha256:afb40d6be65331c20a114d4e229a7ef099fed1b17bf6370daee193514b32aa16` |
| SuperOdometry upstream base | `57a6e233c348372485b7c64ae8238d53c1c7c2ad` |
| Livox ROS Driver 2 | `6b9356cadf77084619ba406e6a0eb41163b08039` |
| Livox SDK2 | `6a940156dd7151c3ab6a52442d86bc83613bd11b` |
| GTSAM | `4abef9248edc4c49943d8fd8a84c028deb486f4c` |
| Sophus | `97e71617749a32b83cdd411591ebb2ede9d330f0` |

The machine-readable source is
[`dependency-lock.env`](dependency-lock.env). The old Dockerfile was not reused
because it contained laptop-specific mounts, GUI/GPU assumptions, and unpinned
dependency installs.

## Build and verify

From this repository:

```bash
cd /move/u/caydengu/cayden/SuperOdom_humanoid_mid360
bash docker/humble-minimal/build_image.sh
bash scripts/test_minimal_container.sh
```

The default tag is `tml/superodom-humble:57a6e23-minimal`. The validated image
is amd64, approximately 1.025 GB, and had image ID
`sha256:068f6e9e32ebc5e9f2706bce9b43edfbf68f88e96f7b5a9be07df027f8243816`
on 2026-07-22. The tag is convenient but mutable; record the image ID for each
live experiment.

Run package and shared-library checks with:

```bash
mkdir -p /move/u/caydengu/superodom_outputs

docker/humble-minimal/run.sh \
  --data-dir /move/u/caydengu/superodom_legacy_drop/bags \
  --output-dir /move/u/caydengu/superodom_outputs \
  -- bash -c '
    ros2 pkg prefix livox_ros_driver2
    ros2 pkg prefix super_odometry_msgs
    ros2 pkg prefix super_odometry
    ldd /opt/superodom_ws/install/lib/super_odometry/laser_mapping_node
  '
```

All three package prefixes should be `/opt/superodom_ws/install`, and `ldd`
must not report `not found`.

## Reproduce the known-good offline replay

The default live configuration keeps `min_range: 0.5`. The transferred bag was
recorded while the G1 was on the gantry, so its regression uses the named
`livox_mid360_gantry.yaml` profile with `min_range: 2.0`. That profile filters
nearby gantry geometry and must not be used to construct the policy heightmap.

```bash
cd /move/u/caydengu/cayden/SuperOdom_humanoid_mid360

replay_output=/move/u/caydengu/superodom_outputs
mkdir -p "$replay_output"

docker/humble-minimal/run.sh \
  --data-dir /move/u/caydengu/superodom_legacy_drop/bags/known_good \
  --output-dir "$replay_output" \
  -- superodom-replay-smoke \
     --bag /data/g1_walk_20260413_080726 \
     --output /output/g1_walk_20260413_080726_superodom_humble
```

The replay driver:

- refuses to overwrite an existing output;
- launches all three estimator nodes with `use_sim_time:=true`;
- checks the generated node parameter files rather than relying on a blocked
  parameter service in `laser_mapping_node`;
- waits for the rosbag recorder's `/clock` endpoint;
- records `/state_estimation`; and
- shuts down the recorder and launch tree cleanly on failure or interruption.

CycloneDDS is the default. To run a middleware separator, add
`--rmw fastrtps` before `--` in the host wrapper. Do not change middleware and
algorithm parameters in the same comparison.

## Offline validation result

The immutable input contains 8,550 `/livox/imu` messages and 427
`/livox/lidar` messages over 42.74 seconds. The final CycloneDDS replay produced
8,314 `/state_estimation` messages over 41.56 seconds of header time.

Compared with the legacy v3 output:

| Metric | Result |
| --- | ---: |
| Shared header timestamps | 8,314 |
| Current-only header timestamps | 0 |
| Legacy-only initialization timestamps | 15 (75.75 ms) |
| Position RMSE on shared timestamps | 5.58 mm |
| Maximum position difference | 36.84 mm |
| Mean orientation difference | 0.0568 degrees |
| Maximum orientation difference | 0.5788 degrees |

Two unprimed current runs emitted exactly the same set of header timestamps,
but their multithreaded estimates were not bitwise identical: position RMSE was
2.22 mm and maximum position difference was 27.03 mm. A synthetic `/clock`
prime was rejected because it changed estimator initialization timing instead
of improving evidence quality.

This is **pipeline and timestamp parity**, not proof of odometry accuracy. The
legacy output is generated by nearly the same estimator and is not independent
ground truth. Its roughly -296 degree unwrapped yaw change also does not match
the informal 180 degree U-turn description closely enough to promote a motion
accuracy claim. Use surveyed static geometry or motion capture for that gate.

The complete logs and comparison JSON are in:

```text
/move/u/caydengu/cayden/research/perceptive-humanoid-diffusion/runs/
  2026-07-22_superodom-minimal-humble-container/
```

## Configuration contracts

- Live launch defaults to `use_sim_time:=false`.
- Offline replay always passes `use_sim_time:=true` explicitly.
- Input topics are `/livox/imu` and `/livox/lidar`.
- Output is `nav_msgs/msg/Odometry` on `/state_estimation`.
- The Mid-360's IMU does not provide a usable orientation quaternion here, so
  `use_imu_roll_pitch` remains false.
- The copied Mid-360 extrinsic is a legacy starting point, not a validated
  calibration for the current robot/sensor mount.
- SuperOdometry estimates the sensor/IMU pose. Converting that to the policy's
  pelvis/root pose is a separate synchronized-kinematics task.

Installed configurations are under:

```text
/opt/superodom_ws/install/share/super_odometry/config/livox_mid360.yaml
/opt/superodom_ws/install/share/super_odometry/config/livox_mid360_gantry.yaml
/opt/superodom_ws/install/share/super_odometry/config/livox/livox_mid360_calibration.yaml
```

## Next gate: live, non-actuating shadow on Oslo

Create empty host directories for the wrapper, then first inspect the Foxy
topics from the Humble container without launching SuperOdometry:

```bash
mkdir -p /move/u/caydengu/superodom_live_input
mkdir -p /move/u/caydengu/superodom_live_outputs

docker/humble-minimal/run.sh \
  --data-dir /move/u/caydengu/superodom_live_input \
  --output-dir /move/u/caydengu/superodom_live_outputs \
  -- bash -c '
    ros2 topic info /livox/imu --verbose
    ros2 topic info /livox/lidar --verbose
  '
```

Only after topic type, QoS, rate, timestamps, and cross-host clock health pass,
launch the estimator without actuation:

```bash
docker/humble-minimal/run.sh \
  --data-dir /move/u/caydengu/superodom_live_input \
  --output-dir /move/u/caydengu/superodom_live_outputs \
  -- ros2 launch super_odometry livox_humanoid.launch.py \
     use_sim_time:=false
```

The live gate must measure, rather than assume:

- Foxy-to-Humble discovery and QoS compatibility;
- sensor header monotonicity and clock offset between the G1 and Oslo;
- G1 capture to Oslo receipt latency;
- capture to `/state_estimation` receipt latency, with p50, p95, and maximum;
- output rate, drops, CPU load, and queue growth;
- behavior during disconnect, timestamp jump, and stale input; and
- static and moved-fixture trajectory quality against independent evidence.

Do not connect `/state_estimation` to root FK, the elevation mapper, the 24 by
16 policy adapter, or the policy until this live boundary passes.
