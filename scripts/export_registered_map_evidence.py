#!/usr/bin/env python3
"""Export KISS-local registered clouds from a ROS 2 bag for exact replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from g1_root_state_bridge.pointcloud2_adapter import (
    decode_xyz_pointcloud2,
    header_time_ns,
)


def main() -> int:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import PointCloud2

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--topic",
        default="/g1/localization/cloud_registered",
    )
    parser.add_argument("--expected-frame", default="kiss_local")
    parser.add_argument("--maximum-points-per-cloud", type=int, default=5_000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(args.bag), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )
    available = {row.name: row.type for row in reader.get_all_topics_and_types()}
    if available.get(args.topic) != "sensor_msgs/msg/PointCloud2":
        raise ValueError(
            f"{args.topic} must be sensor_msgs/msg/PointCloud2; found {available.get(args.topic)!r}"
        )
    points: list[np.ndarray] = []
    offsets = [0]
    source_time_ns: list[int] = []
    receipt_time_ns: list[int] = []
    frame_id: list[str] = []
    while reader.has_next():
        topic, serialized, receipt_ns = reader.read_next()
        if topic != args.topic:
            continue
        message = deserialize_message(serialized, PointCloud2)
        if str(message.header.frame_id) != args.expected_frame:
            raise ValueError(
                "registered cloud frame must be "
                f"{args.expected_frame!r}, got {message.header.frame_id!r}"
            )
        cloud = decode_xyz_pointcloud2(
            message,
            maximum_points=args.maximum_points_per_cloud,
        )
        points.append(cloud)
        offsets.append(offsets[-1] + cloud.shape[0])
        source_time_ns.append(header_time_ns(message))
        receipt_time_ns.append(int(receipt_ns))
        frame_id.append(str(message.header.frame_id))
    if len(points) < 3:
        raise ValueError(f"bag contains fewer than three {args.topic} messages")
    source = np.asarray(source_time_ns, dtype=np.int64)
    if not np.all(np.diff(source) > 0):
        raise ValueError("registered-cloud source time did not strictly increase")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        points_xyz_m=np.concatenate(points, axis=0),
        cloud_offsets=np.asarray(offsets, dtype=np.int64),
        source_time_ns=source,
        receipt_time_ns=np.asarray(receipt_time_ns, dtype=np.int64),
        frame_id=np.asarray(frame_id),
        metadata_json=np.asarray(
            json.dumps(
                {
                    "schema": "g1_registered_map_evidence_archive_v1",
                    "bag": str(args.bag),
                    "topic": args.topic,
                    "frame_id": args.expected_frame,
                    "maximum_points_per_cloud": args.maximum_points_per_cloud,
                    "time_contract": "ROS header source time retained separately from bag receipt time",
                },
                sort_keys=True,
            )
        ),
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "cloud_count": len(points),
                "point_count": offsets[-1],
                "duration_s": float((source[-1] - source[0]) * 1e-9),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
