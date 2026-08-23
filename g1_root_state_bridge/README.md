# G1 root-state bridge

This package produces the timestamped local lane consumed by robot-vlm without
treating the moving torso LiDAR and pelvis as one rigid body. The selected
G1-4123 path is:

1. preserve each Mid-360 point's acquisition time;
2. remove scan rotation with the co-mounted Livox IMU and a frozen stationary
   bias (no accelerometer double integration);
3. run pinned KISS-ICP 1.3.0 for local sensor translation;
4. apply source-time G1 waist FK to move the pose origin to the pelvis;
5. use the bias-corrected, sign-corrected Livox gyro as the torso-aligned
   navigation heading; and
6. emit the fixed 176-byte `HSROOT02` packet with independent health evidence.

`KissPelvisPacketBuilder` is the replay/live-shared producer contract. Missing
deskew, heading, joint synchronization, calibration, registration, or clock
evidence remains an explicit health bit and makes robot-vlm `loc_quality` fail
closed. The packet remains byte-compatible with the original bridge; bits 6/7
are now named `INERTIAL_DESKEW_VALID` and `HEADING_VALID`, with the original
root-IMU names retained as aliases.

The legacy SuperOdometry bridge remains available as the exact Gio baseline.
Its data path is:

The data path is:

1. normalize the exact SuperOdometry sensor-pose and IMU-origin twist semantics;
2. interpolate the typed 29-DOF joint stream at the estimator source timestamp;
3. use waist forward kinematics to recover pelvis pose and twist;
4. optionally interpolate the root IMU at that same timestamp, use its gravity
   roll/pitch and yaw rate, and retain LIO yaw as the complementary-filter anchor;
5. apply the source-time plausibility and health gates; and
6. publish `HSROOT02` and ROS pelvis odometry.

## G1-4123 replay result

With the Motive front-plane rigid body shifted to the CAD pelvis origin by
`[-0.061431244, 0, 0]` m in body coordinates, the selected local lane obtained:

| run | planar RMSE | 1 s translation RPE | yaw RMSE | availability |
|---|---:|---:|---:|---:|
| walk-02 discovery | 0.055 m | 0.033 m | 3.23 deg | 99.66% |
| walk-03 frozen confirmation | 0.051 m | 0.029 m | 2.24 deg | 99.72% |

Gio's matched SuperOdometry baseline was 1.318/1.186 m planar RMSE and
98.34/74.94 deg yaw RMSE. KISS-ICP registration itself measured 8.3 ms mean and
12.7 ms p95 at the frozen 0.15 m voxel size. These are offline Motive-referenced
results, not live hardware qualification.

The slow map lane lives in robot-vlm. It uses the Polycam structural scan for
one global x/y placement and later heading-only corrections. Continuous map
translation is intentionally disabled: on walk-02 it improved only 14.5% of
updates and increased update-time RMSE from 0.051 m to 0.196 m.

The read-only G1 relay emits two CRC-protected datagrams from each accepted
`rt/lowstate` callback: `HSJNT001` for joints and `HSIMU001` for the root IMU.
Both carry the same callback timestamp, sequence, source epoch, and vendor tick.
The bridge drops a root-fused estimate unless two IMU samples bracket its LIO
source time within `max_root_imu_sync_gap_ms`.

The legacy root-fusion mode is deliberately default-off:

```text
enable_root_imu_fusion=false
root_imu_yaw_anchor_tau_s=0.75
root_imu_maximum_step_ms=50.0
root_imu_max_yaw_innovation_deg=45.0
root_imu_gyro_z_sign=1.0
max_root_imu_transport_age_ms=10.0
max_root_imu_sync_gap_ms=5.0
```

For the selected path, `INERTIAL_DESKEW_VALID` and `HEADING_VALID` are required
instead. A gap, epoch change, non-monotonic timestamp, or unavailable joint
bracket removes health and cannot silently perturb a healthy local trajectory.

## Deployment boundary

The estimator algorithms, packet builder, map composition, and robot-vlm
adapter are implemented and replay-qualified. A live KISS ROS producer is not
yet qualified because the captured G1 LiDAR/IMU headers and the offboard host
use different clocks; the offline pipeline applied a measured affine clock
mapping. Do not substitute receipt time silently. The next gate is a passive
live shadow producer that publishes no commands, proves the online clock map,
and compares its `HSROOT02` bytes with the recorded replay before an actuating
test.
