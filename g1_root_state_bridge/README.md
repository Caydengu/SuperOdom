# G1 KISS live localization

This ROS 2 package implements the selected G1-4123 localization producer:

1. subscribe to the deployed Livox PointCloud2 and IMU topics;
2. preserve the PointCloud2 `time` field as seconds relative to the scan header;
3. admit, but never replace, source clocks through bounded affine checks;
4. deskew every published scan with the co-mounted Livox IMU;
5. register scans with pinned KISS-ICP 1.3.0;
6. transform the Mid-360 trajectory to the pelvis using timestamped waist joints;
7. keep the KISS/dynamic-FK pelvis position and replace only orientation with
   bias/sign-corrected Livox torso/navigation yaw;
8. emit ROS odometry, the fixed 176-byte `HSROOT02` packet, and the same
   deskewed scan transformed into the initial-pelvis `kiss_local` frame for the
   slow structural-map lane.

The node is structurally command-incapable. LowState arrives through the
subscriber-only `g1-dynamic-capture-relay`; no Unitree command topic or channel
exists in the localization process.

## Runtime inputs

| Input | Default | Contract |
|---|---|---|
| Point cloud | `/utlidar/cloud_livox_mid360` | `sensor_msgs/PointCloud2`, float32 `x/y/z/time`; native G1 `time` is nanoseconds relative to the scan header and is normalized internally |
| Livox IMU | `/utlidar/imu_livox_mid360` | `sensor_msgs/Imu`, source header time, 3-axis angular velocity |
| LowState | UDP 5589 | CRC-protected `HSDYN001`, 29 measured joints, source epoch and robot callback time |

Outputs are `/g1/localization/pelvis_odom`,
`/g1/localization/cloud_registered`, and ZMQ PUB `tcp://*:5575`. The odometry
and registered cloud share the exact `kiss_local` frame and physical estimate
timestamp. Map ICP must consume this registered cloud; Gio's legacy
`/cloud_registered` belongs to a different estimator frame and must not be
composed with `HSROOT02`.
The heading is the mixed 2D robot-vlm convention: it does not subtract waist
yaw, and it does not claim to be the full rigid pelvis orientation. Reheading
KISS translation increments and subtracting waist yaw were both rejected by
the frozen G1-4123 Motive comparisons.

## Clock behavior

Livox header time and robot callback time remain the physical source times.
Callback receipt observations only fit and validate affine source-to-host clock
maps. The node publishes nothing until both maps have enough samples and meet
rate/residual limits. Each mapper uses a four-second, 4096-sample window after
a 0.5-second startup admission and fits the lower-delay half to a causal lower
envelope. A source regression, epoch change, or unhealthy fit resets the local
lane. A transient sample over the 10 ms transport-delay bound is dropped without
resetting KISS. Missing deskew coverage, stale correction, or joint-sync failure
still prevents output. Receipt time is never substituted for source time.

## Offline replay

```bash
g1-kiss-localization-replay \
  --scans <livox-scan-archive.npz> \
  --lowstate <packets.bin> \
  --output-dir <new-output-dir> \
  --gyro-bias-radps 0.025702817208593076 -0.02178237836035201 -0.01574406003550275
```

The first capture-boundary scan may lack complete IMU coverage. To reproduce
the frozen research treatment it may seed KISS once, but is explicitly
unpublished and never marked deskew-valid.

## Live node

Inside the pinned ROS Humble image:

```bash
g1-kiss-live-localization \
  --gyro-bias-radps 0.025702817208593076 -0.02178237836035201 -0.01574406003550275
```

Live execution is a Goal 2 admission gate. Offline replay and package tests do
not establish live clock, QoS, calibration, map, or physical-robot readiness.

The bounded passive two-host launcher is intentionally separate from AMO:

```bash
scripts/run_g1_kiss_live_stack.sh \
  --network-interface <offboard-G1-NIC> \
  --duration-sec 300
```

It stages only `g1_dynamic_capture_relay` into the G1's existing
`egonav-deploy` Python, where the relay creates a `rt/lowstate` subscriber and
forwards CRC-protected joint samples. The KISS producer remains in the pinned
offboard ROS image. The launcher is bounded by `--duration-sec`, contains no
policy command, and is not to be executed until the Goal 2 passive-shadow gate.

## Structural-map correction shadow

Run the slow map lane in a second terminal only after the local producer is
healthy. The launcher verifies the exact processed Polycam-map digest, mounts
the map read-only, consumes the registered cloud and pelvis odometry in the
shared `kiss_local` frame, and publishes fixed 176-byte `RVMAP001` corrections.
It has no robot command channel and does not launch locomotion:

```bash
scripts/run_g1_structural_map_shadow.sh \
  --network-interface <offboard-G1-NIC> \
  --map <polycam-fieldbay-structural.npz> \
  --map-sha256 8a4aa14ddf10e575459a528f1a213b895c025cefa1388e10e3c2c8f7811c5bb7 \
  --duration-sec 300
```

The selected map key is `map_xy_all_5cm`. Global initialization accumulates a
10-second query; accepted tracking corrections are attempted every two seconds
over a five-second window. `HSROOT02` remains the independent fast local lane
on port 5575, while this slow lane publishes `RVMAP001` on port 5577. Motive is
never an online input.
