#!/usr/bin/env python3
"""Generate causal G1 stance-foot odometry with Livox heading treatments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from g1_root_state_bridge.amo_dataset import load_lowstate
from g1_root_state_bridge.joint_contract import CANONICAL_G1_JOINT_NAMES
from g1_root_state_bridge.leg_contact_odometry import UrdfLegKinematics, stance_foot_odometry
from g1_root_state_bridge.livox_pelvis_yaw import recover_pelvis_yaw, yaw_quaternion_xyzw


def _read_imu(bag: Path, topic_name: str) -> tuple[np.ndarray, np.ndarray]:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Imu

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    time_ns: list[int] = []
    gyro_z: list[float] = []
    while reader.has_next():
        topic, serialized, _receipt = reader.read_next()
        if topic != topic_name:
            continue
        message = deserialize_message(serialized, Imu)
        time_ns.append(int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec))
        gyro_z.append(float(message.angular_velocity.z))
    if not time_ns:
        raise ValueError(f"no {topic_name} messages in {bag}")
    return np.asarray(time_ns, dtype=np.int64), np.asarray(gyro_z, dtype=np.float64)


def _downsample_indices(time_ns: np.ndarray, target_rate_hz: float) -> np.ndarray:
    minimum_step_ns = round(1e9 / target_rate_hz)
    selected = [0]
    last = int(time_ns[0])
    for index in range(1, time_ns.size):
        current = int(time_ns[index])
        if current - last >= minimum_step_ns:
            selected.append(index)
            last = current
    return np.asarray(selected, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lowstate", type=Path, required=True)
    parser.add_argument("--normalized-bag", type=Path, required=True)
    parser.add_argument("--calibration-report", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-rate-hz", type=float, default=200.0)
    parser.add_argument("--imu-topic", default="/utlidar/imu_livox_mid360")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    lowstate = load_lowstate(args.lowstate)
    imu_time, gyro_z = _read_imu(args.normalized_bag, args.imu_topic)
    coverage = (lowstate.oslo_event_ns >= imu_time[0]) & (lowstate.oslo_event_ns <= imu_time[-1])
    candidate = np.flatnonzero(coverage)
    if candidate.size < 2:
        raise ValueError("fewer than two lowstate samples overlap normalized IMU")
    relative = _downsample_indices(lowstate.oslo_event_ns[candidate], args.target_rate_hz)
    indices = candidate[relative]
    event_time = lowstate.oslo_event_ns[indices]
    model = UrdfLegKinematics(args.urdf)
    missing = sorted(set(model.required_joint_names) - set(CANONICAL_G1_JOINT_NAMES))
    if missing:
        raise ValueError(f"URDF requires unavailable joints: {missing}")
    foot_position = np.empty((indices.size, 2, 3), dtype=np.float64)
    for output_index, low_index in enumerate(indices):
        by_name = dict(zip(CANONICAL_G1_JOINT_NAMES, lowstate.joint_position[low_index], strict=True))
        foot_position[output_index] = model.foot_positions(by_name)

    calibration = json.loads(args.calibration_report.read_text(encoding="utf-8"))
    bias_z = float(calibration["stationary_first_10s"]["livox_gyro_radps"]["mean"][2])
    common = dict(
        output_time_ns=event_time,
        imu_time_ns=imu_time,
        sensor_gyro_z_radps=gyro_z,
        sensor_gyro_z_bias_radps=bias_z,
        waist_time_ns=lowstate.oslo_event_ns,
        waist_yaw_rad=lowstate.joint_position[:, 12],
    )
    headings = {
        "torso": recover_pelvis_yaw(**common, compensate_waist=False),
        "pelvis": recover_pelvis_yaw(**common, compensate_waist=True),
    }
    thresholds = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05)
    treatments: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for heading_name, yaw in headings.items():
        for threshold in thresholds:
            position, stance = stance_foot_odometry(
                foot_position,
                yaw,
                switch_height_margin_m=threshold,
            )
            threshold_mm = round(threshold * 1000.0)
            name = f"contact_kinematic_{heading_name}_yaw_switch_{threshold_mm:02d}mm"
            treatments[name] = (position, yaw_quaternion_xyzw(yaw), stance)
    alignment_name = "contact_kinematic_torso_yaw_switch_10mm"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({
            "schema": "g1_amo_localization_treatment_track_v1",
            "kind": "metadata",
            "motive_online_input": False,
            "alignment_reference_treatment": alignment_name,
            "lowstate": str(args.lowstate),
            "normalized_bag": str(args.normalized_bag),
            "urdf": str(args.urdf),
            "target_rate_hz": args.target_rate_hz,
            "gyro_bias_source": str(args.calibration_report),
            "gyro_bias_z_radps": bias_z,
            "contact_rule": "lower_sole_center_with_sticky_height_margin",
        }, sort_keys=True) + "\n")
        for name, (position, quaternion, stance) in treatments.items():
            for sample, low_index in enumerate(indices):
                stream.write(json.dumps({
                    "schema": "g1_amo_localization_treatment_track_v1",
                    "kind": "pose",
                    "treatment": name,
                    "source_time_ns": int(lowstate.robot_stamp_ns[low_index]),
                    "event_realtime_ns": int(event_time[sample]),
                    "source_lowstate_sequence": int(lowstate.sequence[low_index]),
                    "source_lowstate_age_ns": 0,
                    "frame_id": "local_odom",
                    "child_frame_id": "pelvis",
                    "position_xyz_m": [float(value) for value in position[sample]],
                    "quaternion_xyzw": [float(value) for value in quaternion[sample]],
                    "orientation_fusion_healthy": True,
                    "orientation_fusion_reason": "livox_gyro_heading_and_sticky_lower_foot_kinematics",
                    "yaw_innovation_rad": 0.0,
                    "stance_foot": "left" if int(stance[sample]) == 0 else "right",
                }, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "sample_count": int(indices.size),
        "treatments": sorted(treatments),
        "alignment_reference_treatment": alignment_name,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
