#!/usr/bin/env python3
"""Export deployed G1 PointCloud2 and IMU messages into replay_cli's archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from g1_root_state_bridge.pointcloud2_adapter import (
    decode_livox_pointcloud2,
    header_time_ns,
)


def main() -> None:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Imu, PointCloud2

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--topic", default="/utlidar/cloud_livox_mid360")
    parser.add_argument("--imu-topic", default="/utlidar/imu_livox_mid360")
    parser.add_argument("--minimum-range-m", type=float, default=0.5)
    parser.add_argument("--maximum-range-m", type=float, default=15.0)
    parser.add_argument("--maximum-points-per-scan", type=int, default=5_000)
    parser.add_argument(
        "--point-time-unit",
        choices=("nanoseconds", "seconds"),
        default="nanoseconds",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(args.bag), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )
    available = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
    for topic in (args.topic, args.imu_topic):
        expected = "sensor_msgs/msg/PointCloud2" if topic == args.topic else "sensor_msgs/msg/Imu"
        if available.get(topic) != expected:
            raise ValueError(f"{topic} must be {expected}; found {available.get(topic)!r}")

    points: list[np.ndarray] = []
    relative_times: list[np.ndarray] = []
    offsets = [0]
    source_time_ns: list[int] = []
    receipt_time_ns: list[int] = []
    imu_source_time_ns: list[int] = []
    imu_receipt_time_ns: list[int] = []
    imu_angular: list[tuple[float, float, float]] = []
    imu_linear: list[tuple[float, float, float]] = []
    while reader.has_next():
        topic, serialized, receipt = reader.read_next()
        if topic == args.imu_topic:
            message = deserialize_message(serialized, Imu)
            imu_source_time_ns.append(header_time_ns(message))
            imu_receipt_time_ns.append(int(receipt))
            imu_angular.append(
                tuple(float(getattr(message.angular_velocity, axis)) for axis in "xyz")
            )
            imu_linear.append(
                tuple(float(getattr(message.linear_acceleration, axis)) for axis in "xyz")
            )
        elif topic == args.topic:
            message = deserialize_message(serialized, PointCloud2)
            xyz, relative = decode_livox_pointcloud2(
                message,
                minimum_range_m=args.minimum_range_m,
                maximum_range_m=args.maximum_range_m,
                maximum_points=args.maximum_points_per_scan,
                time_unit=args.point_time_unit,
            )
            points.append(xyz)
            relative_times.append(relative)
            offsets.append(offsets[-1] + xyz.shape[0])
            source_time_ns.append(header_time_ns(message))
            receipt_time_ns.append(int(receipt))

    if not points or not imu_source_time_ns:
        raise ValueError("bag is missing deployed LiDAR or IMU messages")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema": "g1_livox_pointcloud2_replay_archive_v1",
        "topic": args.topic,
        "imu_topic": args.imu_topic,
        "point_time_unit": args.point_time_unit,
        "minimum_range_m": args.minimum_range_m,
        "maximum_range_m": args.maximum_range_m,
        "maximum_points_per_scan": args.maximum_points_per_scan,
        "selection": "deployed_decoder_range_filter_then_even_cap",
    }
    np.savez_compressed(
        args.output,
        points_xyz_m=np.concatenate(points),
        point_relative_time_s=np.concatenate(relative_times),
        scan_offsets=np.asarray(offsets, dtype=np.int64),
        source_time_ns=np.asarray(source_time_ns, dtype=np.int64),
        receipt_time_ns=np.asarray(receipt_time_ns, dtype=np.int64),
        imu_source_time_ns=np.asarray(imu_source_time_ns, dtype=np.int64),
        imu_receipt_time_ns=np.asarray(imu_receipt_time_ns, dtype=np.int64),
        imu_angular_velocity_radps=np.asarray(imu_angular, dtype=np.float32),
        imu_linear_acceleration_mps2=np.asarray(imu_linear, dtype=np.float32),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "scan_count": len(points),
                "point_count": offsets[-1],
                "imu_count": len(imu_source_time_ns),
                "duration_s": (source_time_ns[-1] - source_time_ns[0]) * 1e-9,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
