#!/usr/bin/env python3
"""Export a sensor_msgs/Imu topic with source and bag receipt timestamps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Imu

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--topic", default="/utlidar/imu_livox_mid360")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(args.bag), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    available = {entry.name: entry.type for entry in reader.get_all_topics_and_types()}
    if available.get(args.topic) != "sensor_msgs/msg/Imu":
        raise ValueError(f"{args.topic} is not a sensor_msgs/Imu topic")
    count = 0
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "schema": "g1_imu_track_v1",
                    "kind": "metadata",
                    "bag_path": str(args.bag),
                    "topic": args.topic,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        while reader.has_next():
            topic, serialized, receipt_ns = reader.read_next()
            if topic != args.topic:
                continue
            message = deserialize_message(serialized, Imu)
            source_ns = (
                int(message.header.stamp.sec) * 1_000_000_000
                + int(message.header.stamp.nanosec)
            )
            stream.write(
                json.dumps(
                    {
                        "schema": "g1_imu_track_v1",
                        "kind": "sample",
                        "source_time_ns": source_ns,
                        "receipt_time_ns": int(receipt_ns),
                        "angular_velocity_radps": [
                            float(message.angular_velocity.x),
                            float(message.angular_velocity.y),
                            float(message.angular_velocity.z),
                        ],
                        "linear_acceleration": [
                            float(message.linear_acceleration.x),
                            float(message.linear_acceleration.y),
                            float(message.linear_acceleration.z),
                        ],
                    },
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            count += 1
    print(json.dumps({"count": count, "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
