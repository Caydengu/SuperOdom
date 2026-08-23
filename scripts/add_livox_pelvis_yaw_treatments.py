#!/usr/bin/env python3
"""Add matched Livox-gyro pelvis-heading treatments to an existing pose track."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from g1_root_state_bridge.amo_dataset import load_lowstate
from g1_root_state_bridge.livox_pelvis_yaw import (
    quaternion_yaw_z_up,
    recover_pelvis_yaw,
    rehead_planar_trajectory,
    yaw_quaternion_xyzw,
)


def _read_imu(bag: Path, topic_name: str) -> tuple[np.ndarray, np.ndarray]:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Imu

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    times: list[int] = []
    gyro_z: list[float] = []
    while reader.has_next():
        topic, serialized, _receipt = reader.read_next()
        if topic != topic_name:
            continue
        message = deserialize_message(serialized, Imu)
        times.append(int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec))
        gyro_z.append(float(message.angular_velocity.z))
    if not times:
        raise ValueError(f"no {topic_name} messages in {bag}")
    return np.asarray(times, dtype=np.int64), np.asarray(gyro_z, dtype=np.float64)


def _load_records(path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    metadata: dict[str, object] | None = None
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            payload = json.loads(line)
            if payload.get("kind") == "metadata":
                metadata = payload
            else:
                records.append(payload)
    if metadata is None or not records:
        raise ValueError("treatment track must contain metadata and pose records")
    return metadata, records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-treatments", type=Path, required=True)
    parser.add_argument("--normalized-bag", type=Path, required=True)
    parser.add_argument("--lowstate", type=Path, required=True)
    parser.add_argument("--calibration-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imu-topic", default="/utlidar/imu_livox_mid360")
    parser.add_argument("--source-treatment", default="superodom_dynamic_fk_pelvis")
    parser.add_argument(
        "--output-prefix",
        default="superodom",
        help="Prefix for derived treatment names (keeps alternative odometry arms distinct).",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    metadata, records = _load_records(args.input_treatments)
    source_all = [row for row in records if row.get("treatment") == args.source_treatment]
    if not source_all:
        raise ValueError(f"missing source treatment {args.source_treatment}")
    imu_time, gyro_z = _read_imu(args.normalized_bag, args.imu_topic)
    lowstate = load_lowstate(args.lowstate)
    lower = max(int(imu_time[0]), int(lowstate.oslo_event_ns[0]))
    upper = min(int(imu_time[-1]), int(lowstate.oslo_event_ns[-1]))
    source = [
        row
        for row in source_all
        if lower <= int(row["event_realtime_ns"]) <= upper
    ]
    if len(source) < 2:
        raise ValueError("fewer than two source records overlap IMU and waist coverage")
    output_time = np.asarray([row["event_realtime_ns"] for row in source], dtype=np.int64)
    calibration = json.loads(args.calibration_report.read_text(encoding="utf-8"))
    bias_z = float(calibration["stationary_first_10s"]["livox_gyro_radps"]["mean"][2])
    common = dict(
        output_time_ns=output_time,
        imu_time_ns=imu_time,
        sensor_gyro_z_radps=gyro_z,
        sensor_gyro_z_bias_radps=bias_z,
        waist_time_ns=lowstate.oslo_event_ns,
        waist_yaw_rad=lowstate.joint_position[:, 12],
    )
    yaw_torso = recover_pelvis_yaw(**common, compensate_waist=False)
    yaw_pelvis = recover_pelvis_yaw(**common, compensate_waist=True)
    quaternions = {
        f"{args.output_prefix}_position_livox_gyro_torso_yaw": yaw_quaternion_xyzw(yaw_torso),
        f"{args.output_prefix}_position_livox_gyro_pelvis_yaw": yaw_quaternion_xyzw(yaw_pelvis),
    }
    source_position = np.asarray([row["position_xyz_m"] for row in source], dtype=np.float64)
    source_quaternion = np.asarray([row["quaternion_xyzw"] for row in source], dtype=np.float64)
    reheaded_positions = {
        f"{args.output_prefix}_reheaded_translation_livox_gyro_torso_yaw": rehead_planar_trajectory(
            source_position, quaternion_yaw_z_up(source_quaternion), yaw_torso
        ),
        f"{args.output_prefix}_reheaded_translation_livox_gyro_pelvis_yaw": rehead_planar_trajectory(
            source_position, quaternion_yaw_z_up(source_quaternion), yaw_pelvis
        ),
    }
    reheaded_quaternions = {
        f"{args.output_prefix}_reheaded_translation_livox_gyro_torso_yaw": yaw_quaternion_xyzw(yaw_torso),
        f"{args.output_prefix}_reheaded_translation_livox_gyro_pelvis_yaw": yaw_quaternion_xyzw(yaw_pelvis),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({
            **metadata,
            "livox_yaw_treatment": {
                "motive_online_input": False,
                "gyro_bias_source": str(args.calibration_report),
                "gyro_bias_z_radps": bias_z,
                "sensor_z_to_torso_yaw_sign": -1.0,
                "waist_joint_index": 12,
                "integration": "trapezoidal_causal",
            },
        }, sort_keys=True) + "\n")
        for row in records:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        for index, row in enumerate(source):
            for treatment, quaternion in quaternions.items():
                derived = dict(row)
                derived["treatment"] = treatment
                derived["quaternion_xyzw"] = [float(value) for value in quaternion[index]]
                derived["orientation_fusion_healthy"] = True
                derived["orientation_fusion_reason"] = "stationary_bias_livox_gyro_with_optional_waist_compensation"
                derived["yaw_innovation_rad"] = 0.0
                stream.write(json.dumps(derived, sort_keys=True, separators=(",", ":")) + "\n")
            for treatment, positions in reheaded_positions.items():
                derived = dict(row)
                derived["treatment"] = treatment
                derived["position_xyz_m"] = [float(value) for value in positions[index]]
                derived["quaternion_xyzw"] = [
                    float(value) for value in reheaded_quaternions[treatment][index]
                ]
                derived["orientation_fusion_healthy"] = True
                derived["orientation_fusion_reason"] = "source_local_translation_reintegrated_under_livox_gyro_heading"
                derived["yaw_innovation_rad"] = 0.0
                stream.write(json.dumps(derived, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "source_records": len(source),
        "source_records_before_coverage_filter": len(source_all),
        "bias_z_radps": bias_z,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
