# G1 root-state bridge

This package turns SuperOdometry's physical MID-360 state into a timestamped
`local_odometry <- pelvis` state without treating the moving head and pelvis as
one rigid body.

The data path is:

1. normalize the exact SuperOdometry sensor-pose and IMU-origin twist semantics;
2. interpolate the typed 29-DOF joint stream at the estimator source timestamp;
3. use waist forward kinematics to recover pelvis pose and twist;
4. optionally interpolate the root IMU at that same timestamp, use its gravity
   roll/pitch and yaw rate, and retain LIO yaw as the complementary-filter anchor;
5. apply the source-time plausibility and health gates; and
6. publish the fixed 176-byte `HSROOT02` packet and ROS pelvis odometry.

The read-only G1 relay emits two CRC-protected datagrams from each accepted
`rt/lowstate` callback: `HSJNT001` for joints and `HSIMU001` for the root IMU.
Both carry the same callback timestamp, sequence, source epoch, and vendor tick.
The bridge drops a root-fused estimate unless two IMU samples bracket its LIO
source time within `max_root_imu_sync_gap_ms`.

Root fusion is deliberately default-off until a live, passive calibration run
confirms the Unitree quaternion convention and `gyro_z_sign` on the actual G1:

```text
enable_root_imu_fusion=false
root_imu_yaw_anchor_tau_s=0.75
root_imu_maximum_step_ms=50.0
root_imu_max_yaw_innovation_deg=45.0
root_imu_gyro_z_sign=1.0
max_root_imu_transport_age_ms=10.0
max_root_imu_sync_gap_ms=5.0
```

When enabled, `ROOT_IMU_SYNC_VALID` and `ROOT_ORIENTATION_FUSED` are explicit
health bits. A gap, epoch change, non-monotonic timestamp, or excessive yaw
innovation removes fusion health and cannot silently perturb a healthy local
trajectory. Live walking accuracy remains unqualified until Motive comparison.
