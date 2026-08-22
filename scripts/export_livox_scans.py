#!/usr/bin/env python3
"""Export a deterministic, bounded Livox scan archive from a ROS 2 bag."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rosbag2_py
from livox_ros_driver2.msg import CustomMsg
from rclpy.serialization import deserialize_message


def bounded_points(
    message: CustomMsg,
    *,
    minimum_range_m: float,
    maximum_range_m: float,
    maximum_points: int,
) -> np.ndarray:
    xyz = np.asarray(
        [(point.x, point.y, point.z) for point in message.points], dtype=np.float32
    )
    finite = np.all(np.isfinite(xyz), axis=1)
    squared_range = np.sum(xyz.astype(np.float64) ** 2, axis=1)
    admitted = (
        finite
        & (squared_range >= minimum_range_m**2)
        & (squared_range <= maximum_range_m**2)
    )
    xyz = xyz[admitted]
    if xyz.shape[0] > maximum_points:
        # Evenly spaced deterministic selection preserves the rotating Livox
        # pattern better than taking one contiguous prefix.
        indices = np.linspace(0, xyz.shape[0] - 1, maximum_points, dtype=np.int64)
        xyz = xyz[indices]
    return xyz


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--topic", default="/livox/lidar")
    parser.add_argument("--minimum-range-m", type=float, default=0.5)
    parser.add_argument("--maximum-range-m", type=float, default=15.0)
    parser.add_argument("--maximum-points-per-scan", type=int, default=5000)
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
    points: list[np.ndarray] = []
    offsets = [0]
    source_time_ns: list[int] = []
    receipt_time_ns: list[int] = []
    while reader.has_next():
        topic, serialized, receipt = reader.read_next()
        if topic != args.topic:
            continue
        message = deserialize_message(serialized, CustomMsg)
        scan = bounded_points(
            message,
            minimum_range_m=args.minimum_range_m,
            maximum_range_m=args.maximum_range_m,
            maximum_points=args.maximum_points_per_scan,
        )
        points.append(scan)
        offsets.append(offsets[-1] + scan.shape[0])
        source_time_ns.append(
            int(message.header.stamp.sec) * 1_000_000_000
            + int(message.header.stamp.nanosec)
        )
        receipt_time_ns.append(int(receipt))
    if not points:
        raise ValueError(f"bag contains no {args.topic} messages")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        points_xyz_m=np.concatenate(points, axis=0),
        scan_offsets=np.asarray(offsets, dtype=np.int64),
        source_time_ns=np.asarray(source_time_ns, dtype=np.int64),
        receipt_time_ns=np.asarray(receipt_time_ns, dtype=np.int64),
        metadata_json=np.asarray(
            json.dumps(
                {
                    "schema": "g1_livox_scan_archive_v1",
                    "topic": args.topic,
                    "minimum_range_m": args.minimum_range_m,
                    "maximum_range_m": args.maximum_range_m,
                    "maximum_points_per_scan": args.maximum_points_per_scan,
                    "selection": "range_filter_then_evenly_spaced_deterministic_cap",
                },
                sort_keys=True,
            )
        ),
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "scan_count": len(points),
                "point_count": offsets[-1],
                "duration_s": (source_time_ns[-1] - source_time_ns[0]) * 1e-9,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
