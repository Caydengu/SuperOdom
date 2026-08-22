#!/usr/bin/env python3
"""Export timestamped odometry topics from a ROS 2 bag as JSON Lines.

The exporter intentionally keeps source/header time and bag receipt time
separate.  Its output is small enough to analyze outside ROS while retaining
the frame and child-frame semantics needed for sensor-versus-pelvis studies.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, TextIO


DEFAULT_TOPICS = ("/state_estimation", "/pelvis_state_estimation")


def _stamp_ns(stamp: Any) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def odometry_record(topic: str, message: Any, receipt_time_ns: int) -> dict[str, Any]:
    """Convert a nav_msgs/Odometry-like object into a stable JSON record."""
    pose = message.pose.pose
    twist = message.twist.twist
    return {
        "schema": "g1_odometry_track_v1",
        "kind": "odometry",
        "topic": topic,
        "source_time_ns": _stamp_ns(message.header.stamp),
        "receipt_time_ns": int(receipt_time_ns),
        "frame_id": str(message.header.frame_id),
        "child_frame_id": str(message.child_frame_id),
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
        "linear_velocity_xyz_mps": [
            float(twist.linear.x),
            float(twist.linear.y),
            float(twist.linear.z),
        ],
        "angular_velocity_xyz_radps": [
            float(twist.angular.x),
            float(twist.angular.y),
            float(twist.angular.z),
        ],
    }


def export_bag(bag_path: Path, topics: Iterable[str], output: TextIO) -> dict[str, int]:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    requested = tuple(dict.fromkeys(topics))
    requested_set = set(requested)
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_path), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )
    available_types = {
        topic.name: topic.type for topic in reader.get_all_topics_and_types()
    }
    missing = sorted(requested_set - available_types.keys())
    if missing:
        raise ValueError(f"bag is missing requested topics: {', '.join(missing)}")
    wrong_type = {
        name: available_types[name]
        for name in requested
        if available_types[name] != "nav_msgs/msg/Odometry"
    }
    if wrong_type:
        raise ValueError(f"requested topics are not Odometry: {wrong_type}")

    message_type = get_message("nav_msgs/msg/Odometry")
    counts = {topic: 0 for topic in requested}
    output.write(
        json.dumps(
            {
                "schema": "g1_odometry_track_v1",
                "kind": "metadata",
                "bag_path": str(bag_path),
                "topics": list(requested),
                "time_contract": {
                    "source_time_ns": "ROS message header capture/estimate time",
                    "receipt_time_ns": "rosbag record receipt time",
                },
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )
    while reader.has_next():
        topic, serialized, receipt_time_ns = reader.read_next()
        if topic not in requested_set:
            continue
        message = deserialize_message(serialized, message_type)
        output.write(
            json.dumps(
                odometry_record(topic, message, int(receipt_time_ns)),
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        )
        counts[topic] += 1
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--topic", action="append", dest="topics")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    topics = args.topics or list(DEFAULT_TOPICS)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    with args.output.open("x", encoding="utf-8") as stream:
        counts = export_bag(args.bag, topics, stream)
    print(json.dumps({"output": str(args.output), "counts": counts}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
