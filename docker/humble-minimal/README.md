# Minimal ROS 2 Humble SuperOdometry container

This is the TML offboard SuperOdometry path for the Unitree G1 Mid-360. It
builds a pinned, CPU-only ROS 2 Humble image on Oslo while leaving the robot's
ROS 2 Foxy installation unchanged.

Current status: **offline replay and stationary live shadow validated on
2026-07-22**. The image builds, all required packages resolve, all shared
libraries load, and the transferred known-good bag produces
`/state_estimation` with close parity to the legacy offboard pipeline. A live
Foxy Mid-360 on the gantry-supported G1 also drove coherent, stationary-stable
output through an isolated Humble container on Oslo, subject to the one-raw-
cloud-consumer constraint below.

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

## Live, non-actuating shadow on Oslo

Live DDS requires all three of these settings:

- an isolated `ROS_DOMAIN_ID` shared by the G1 driver and Oslo container;
- explicit CycloneDDS selection of Oslo's robot-facing NIC; and
- exactly one reliable offboard subscriber to the raw `CustomMsg` cloud.

Do not omit `--network-interface`. With host networking, CycloneDDS initially
selected Oslo's campus NIC (`eno2`) and discovered no G1 publishers. Selecting
`enxc8a362a1dc13` immediately discovered the Foxy topics. Resolve the current
robot NIC with `ip route get 192.168.123.164` rather than copying that interface
name across machines or boots.

The current G1 `msg_MID360_launch.py` must also be replaced by a driver-only
launch before production use. It starts a delayed Python
`ros2 topic hz /livox/lidar -w 5` process. That second raw-cloud reader reduced
receipt throughput below 10 Hz and caused sustained message-age growth. Killing
only the monitor restored a stable 10.0002 Hz reliable stream with 85.7 ms p95
newest-point age and no age growth. A permanent driver-only launch is still a
required G1-side package change.

First record a bounded input probe without launching SuperOdometry:

```bash
cd /move/u/caydengu/cayden/SuperOdom_humanoid_mid360

docker/humble-minimal/live_input_probe.sh \
  --ros-domain-id 42 \
  --network-interface enxc8a362a1dc13 \
  --output-dir /move/u/caydengu/superodom_live_input \
  --duration-sec 30
```

Only after type, QoS, rate, monotonicity, point offsets, and queue-age growth
pass, run the bounded output-only shadow:

```bash
docker/humble-minimal/live_shadow.sh \
  --ros-domain-id 42 \
  --network-interface enxc8a362a1dc13 \
  --output-dir /move/u/caydengu/superodom_live_shadow \
  --duration-sec 90
```

The default deliberately records only IMU and compact estimator diagnostics.
`--record-lidar` is a transport stress treatment, not the production topology:
the estimator plus a second 400 kB raw-cloud recorder reproduced backlog at
9.50 Hz, 1.80 s p95 newest-point age, and +14.55 ms/s age growth. Record the
raw cloud upstream in a separate calibration/data-collection run, or introduce
an explicitly designed single-reader transport, rather than casually adding
raw subscribers.

### Stationary live result

The production-topology 90-second run produced:

| Metric | Result |
| --- | ---: |
| `/state_estimation` rate | 200.00 Hz |
| Output timestamps matching recorded IMU timestamps | 18,161 / 18,161 |
| Output receipt age p95 | 0.892 ms |
| First output after launch | 1.354 s |
| Post-2-second maximum translation from warmup pose | 2.80 cm |
| Post-2-second terminal translation from warmup pose | 2.02 cm |
| Post-2-second maximum one-update translation | 2.39 mm |
| Post-2-second maximum yaw from warmup pose | 1.31 degrees |

An independent 45-second output-only diagnostic run reported all 8,830 health
flags true and all 439 prediction-source samples as laser-inertial odometry.
Whole-frame processing was 15.20 ms median, 18.02 ms p95, and 22.27 ms maximum,
well below the 100 ms LiDAR period. Small diagnostic topics did not recreate
raw-cloud backlog. Occasional roughly 200 ms interarrival gaps remain visible
in the 10 Hz correction/statistics stream and should be revisited during
dynamic validation.

The complete bags, logs, metrics, and decision dashboard are in:

```text
/move/u/caydengu/cayden/research/perceptive-humanoid-diffusion/runs/
  2026-07-22_superodometry-live-shadow/
```

### What this result does and does not establish

This gate establishes live Foxy-to-Humble transport, timestamp correspondence,
stationary relative stability, CPU processing headroom, and the viable
single-reader topology. It does **not** establish absolute pose accuracy,
dynamic motion quality, calibrated sensor extrinsics, or a pelvis/root pose.
`/state_estimation` remains a sensor/IMU trajectory.

The next gate must measure, rather than assume:

- dynamic trajectory and orientation against an independent pose reference;
- timestamped sensor-to-pelvis conversion using synchronized joint state and
  forward kinematics;
- current Mid-360 extrinsic calibration and time alignment;
- controlled disconnect/stale-input behavior; and
- whether the single-reader architecture should terminate in SuperOdometry or
  a dedicated acquisition/recording relay.

Do not connect `/state_estimation` to root FK, the elevation mapper, the 24 by
16 policy adapter, or the policy until the dynamic pose and root-FK gates pass.

## Recorder-free policy-state service

For policy co-load and fixed-rate shadow work, use the dedicated foreground
service instead of `root_state_shadow.sh`:

```bash
docker/humble-minimal/policy_state_service.sh \
  --network-interface <robot-facing-interface> \
  --ros-domain-id 42 \
  --output-dir /move/u/caydengu/superodom_outputs/<new-service-run> \
  --image tml/superodom-humble:h12-shadow-minimal
```

This path starts exactly one SuperOdometry launch tree and one
`g1_root_state_bridge` process. It does not start a ROS bag, raw-cloud reader,
full-packet JSONL trace, or policy. Readiness is based on two advancing,
strictly valid, same-epoch `HSPOLI01` packets—not merely an open TCP port.

The bridge's default empty replay path disables per-packet array and payload
serialization. It publishes a compact one-hertz heartbeat containing only
counters, the latest sequence/epoch, pending depth, and calibration readiness.
The service itself appends a bounded five-second process heartbeat to
`service_status.jsonl`, writes exact source/image/config identities, and writes
`stop_receipt.json` after process-group cleanup. The older bounded diagnostic
shadow retains full packet evidence by supplying a non-empty replay path; that
diagnostic mode is not the co-load topology.
