#!/usr/bin/env python3
"""Export compact SuperOdometry diagnostic streams from a ROS 2 bag."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, TextIO


TOPICS = {
    "/livox/imu": "sensor_msgs/msg/Imu",
    "/laser_odometry": "nav_msgs/msg/Odometry",
    "/super_odometry_stats": "super_odometry_msgs/msg/OptimizationStats",
    "/state_estimation_correction": "super_odometry_msgs/msg/StateEstimationCorrection",
}


def stamp_ns(stamp: Any) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def record(topic: str, message: Any, receipt_time_ns: int) -> dict[str, Any]:
    base = {
        "schema": "g1_superodom_diagnostics_v1",
        "kind": "diagnostic",
        "topic": topic,
        "receipt_time_ns": int(receipt_time_ns),
    }
    if topic == "/livox/imu":
        return {
            **base,
            "source_time_ns": stamp_ns(message.header.stamp),
            "angular_velocity_xyz_radps": [
                float(message.angular_velocity.x),
                float(message.angular_velocity.y),
                float(message.angular_velocity.z),
            ],
            "linear_acceleration_xyz_mps2": [
                float(message.linear_acceleration.x),
                float(message.linear_acceleration.y),
                float(message.linear_acceleration.z),
            ],
        }
    if topic == "/laser_odometry":
        pose = message.pose.pose
        return {
            **base,
            "source_time_ns": stamp_ns(message.header.stamp),
            "position_xyz_m": [
                float(pose.position.x),
                float(pose.position.y),
                float(pose.position.z),
            ],
            "quaternion_xyzw": [
                float(pose.orientation.x),
                float(pose.orientation.y),
                float(pose.orientation.z),
                float(pose.orientation.w),
            ],
        }
    if topic == "/super_odometry_stats":
        return {
            **base,
            "source_time_ns": stamp_ns(message.header.stamp),
            "total_translation_m": float(message.total_translation),
            "total_rotation_rad": float(message.total_rotation),
            "translation_from_last_m": float(message.translation_from_last),
            "rotation_from_last_rad": float(message.rotation_from_last),
            "optimization_ms": float(message.time_elapsed),
            "frame_processing_ms": float(message.latency),
            "iterations": int(message.n_iterations),
            "average_distance_m": float(message.average_distance),
            "uncertainty_xyz_rpy": [
                float(message.uncertainty_x),
                float(message.uncertainty_y),
                float(message.uncertainty_z),
                float(message.uncertainty_roll),
                float(message.uncertainty_pitch),
                float(message.uncertainty_yaw),
            ],
            "plane_match_success": int(message.plane_match_success),
            "plane_failure_counts": {
                "not_enough_neighbor": int(message.plane_no_enough_neighbor),
                "neighbor_too_far": int(message.plane_neighbor_too_far),
                "bad_pca_structure": int(message.plane_badpca_structure),
                "invalid_numerical": int(message.plane_invalid_numerical),
                "mse_too_large": int(message.plane_mse_too_large),
                "unknown": int(message.plane_unknown),
            },
            "prediction_source": int(message.prediction_source),
        }
    if topic == "/state_estimation_correction":
        return {
            **base,
            "source_time_ns": stamp_ns(message.header.stamp),
            "newest_observation_time_ns": stamp_ns(message.newest_observation_stamp),
            "mapping_output_time_ns": stamp_ns(message.mapping_output_stamp),
            "application_time_ns": stamp_ns(message.application_stamp),
            "sequence": int(message.sequence),
            "reset_id": int(message.reset_id),
            "valid": bool(message.valid),
            "semantics_version": str(message.semantics_version),
        }
    raise ValueError(f"unsupported topic: {topic}")


def export_bag(bag_path: Path, output: TextIO) -> dict[str, int]:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_path), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )
    available = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
    missing = sorted(set(TOPICS) - available.keys())
    if missing:
        raise ValueError(f"bag is missing diagnostic topics: {', '.join(missing)}")
    wrong_type = {
        topic: available[topic]
        for topic, expected in TOPICS.items()
        if available[topic] != expected
    }
    if wrong_type:
        raise ValueError(f"diagnostic topic types differ: {wrong_type}")
    classes = {topic: get_message(type_name) for topic, type_name in TOPICS.items()}
    counts = {topic: 0 for topic in TOPICS}
    output.write(
        json.dumps(
            {
                "schema": "g1_superodom_diagnostics_v1",
                "kind": "metadata",
                "bag_path": str(bag_path),
                "topics": TOPICS,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    while reader.has_next():
        topic, serialized, receipt_time_ns = reader.read_next()
        if topic not in TOPICS:
            continue
        message = deserialize_message(serialized, classes[topic])
        output.write(
            json.dumps(
                record(topic, message, int(receipt_time_ns)),
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        counts[topic] += 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        counts = export_bag(args.bag, stream)
    print(json.dumps({"output": str(args.output), "counts": counts}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
