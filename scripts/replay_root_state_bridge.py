#!/usr/bin/env python3
"""Replay historical SuperOdometry odometry through the pelvis bridge core.

This is explicitly a mechanics replay. Historical bags lack typed G1 joints,
typed estimator health/correction, and the new calibration message, so those
inputs are reconstructed or synthesized and identified in the metadata record.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

from g1_root_state_bridge.bridge_node import _observation_from_odometry
from g1_root_state_bridge.calibration_contract import load_installed_calibration_contract
from g1_root_state_bridge.replay import (
    HistoricalRootStateReplayer,
    gravity_alignment_from_accelerations,
)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bag_sources(bag: Path) -> list[dict[str, object]]:
    sources: list[dict[str, object]] = []
    for path in sorted(bag.iterdir()):
        if path.is_file() and (path.name == "metadata.yaml" or path.suffix in {".db3", ".mcap"}):
            sources.append(
                {
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
    return sources


def _read_bag(
    bag: Path,
    *,
    max_odometry_messages: int | None,
    gravity_sample_count: int,
) -> tuple[list[object], list[np.ndarray]]:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("", ""),
    )
    topic_types = {
        item.name: get_message(item.type) for item in reader.get_all_topics_and_types()
    }
    required = {"/state_estimation", "/livox/imu"}
    missing = required - topic_types.keys()
    if missing:
        raise RuntimeError(f"historical bag is missing required topics: {sorted(missing)}")

    odometry: list[object] = []
    accelerations: list[np.ndarray] = []
    while reader.has_next():
        topic, payload, _storage_time_ns = reader.read_next()
        if topic == "/state_estimation":
            if max_odometry_messages is None or len(odometry) < max_odometry_messages:
                odometry.append(deserialize_message(payload, topic_types[topic]))
        elif topic == "/livox/imu" and len(accelerations) < gravity_sample_count:
            message = deserialize_message(payload, topic_types[topic])
            accelerations.append(
                np.asarray(
                    (
                        message.linear_acceleration.x,
                        message.linear_acceleration.y,
                        message.linear_acceleration.z,
                    ),
                    dtype=np.float64,
                )
            )
        if (
            max_odometry_messages is not None
            and len(odometry) >= max_odometry_messages
            and len(accelerations) >= gravity_sample_count
        ):
            break
    if not odometry:
        raise RuntimeError("historical bag contains no /state_estimation messages")
    if not accelerations:
        raise RuntimeError("historical bag contains no usable /livox/imu messages")
    return odometry, accelerations


def _packet_record(result) -> dict[str, object]:
    packet = result.packet
    return {
        "schema": "g1-root-state-replay-v1",
        "kind": "packet",
        "original_estimate_time_ns": result.original_estimate_time_ns,
        "receipt_time_ns": result.consumer_receipt_time_ns,
        "payload_hex": result.payload.hex(),
        "sequence": packet.sequence,
        "source_epoch": packet.source_epoch,
        "estimate_time_ns": packet.estimate_time_ns,
        "publish_time_ns": packet.publish_time_ns,
        "correction_time_ns": packet.correction_time_ns,
        "joint_time_ns": packet.joint_time_ns,
        "joint_sync_gap_ns": packet.joint_sync_gap_ns,
        "health_flags": int(packet.health_flags),
        "strictly_valid_at_publish": packet.strictly_valid,
        "position": packet.position,
        "quaternion_wxyz": packet.quaternion_wxyz,
        "linear_velocity_world": packet.linear_velocity,
        "angular_velocity_world": packet.angular_velocity,
        "calibration_digest_sha256": packet.calibration_digest.hex(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--output-binary", required=True, type=Path)
    parser.add_argument("--metrics-json", required=True, type=Path)
    parser.add_argument("--max-odometry-messages", type=int)
    parser.add_argument("--gravity-sample-count", type=int, default=400)
    parser.add_argument("--source-revision", default="unknown")
    args = parser.parse_args()

    bag = args.bag.resolve()
    if not bag.is_dir():
        raise FileNotFoundError(f"bag directory does not exist: {bag}")
    if args.max_odometry_messages is not None and args.max_odometry_messages <= 0:
        parser.error("--max-odometry-messages must be positive")
    if args.gravity_sample_count <= 0:
        parser.error("--gravity-sample-count must be positive")
    for output in (args.output_jsonl, args.output_binary, args.metrics_json):
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f"refusing to overwrite replay artifact: {output}")

    odometry_messages, accelerations = _read_bag(
        bag,
        max_odometry_messages=args.max_odometry_messages,
        gravity_sample_count=args.gravity_sample_count,
    )
    observations = [
        _observation_from_odometry(
            message,
            receipt_time_ns=(
                int(message.header.stamp.sec) * 1_000_000_000
                + int(message.header.stamp.nanosec)
            ),
        )
        for message in odometry_messages
    ]
    contract = load_installed_calibration_contract()
    map_T_gravity = gravity_alignment_from_accelerations(
        accelerations,
        map_T_lidar=observations[0].map_T_lidar,
        imu_T_lidar=contract.imu_T_lidar,
    )
    replayer = HistoricalRootStateReplayer(
        map_T_gravity=map_T_gravity,
        imu_T_lidar=contract.imu_T_lidar,
        calibration_digest=contract.digest,
        map_frame=observations[0].frame_id,
        sensor_frame=observations[0].child_frame_id,
    )

    metadata = {
        "schema": "g1-root-state-replay-v1",
        "kind": "metadata",
        "source_revision": args.source_revision,
        "source_bag": str(bag),
        "source_files": _bag_sources(bag),
        "calibration_digest_sha256": contract.digest.hex(),
        "calibration_manifest": contract.manifest,
        "map_T_gravity": map_T_gravity.tolist(),
        "gravity_sample_count": len(accelerations),
        "mean_acceleration_imu": np.mean(np.stack(accelerations), axis=0).tolist(),
        "evidence_scope": "mechanics_only_not_dynamic_ground_truth",
        "synthetic_inputs": [
            "zero_29dof_joint_position_and_velocity",
            "healthy_typed_estimator_sidecar",
            "per-estimate_correction_sidecar",
            "shifted_timestamp_epoch",
        ],
        "derived_inputs": ["startup_gravity_alignment_from_bagged_imu"],
    }

    packet_count = 0
    strict_count = 0
    rejection_counts: dict[str, int] = {}
    positions: list[tuple[float, float, float]] = []
    linear_speeds: list[float] = []
    angular_speeds: list[float] = []
    with args.output_jsonl.open("x", encoding="utf-8") as jsonl, args.output_binary.open(
        "xb"
    ) as binary:
        jsonl.write(json.dumps(metadata, sort_keys=True, allow_nan=False) + "\n")
        for index, observation in enumerate(observations):
            try:
                result = replayer.replay(observation)
            except Exception as error:  # Preserve exact invalid evidence and continue.
                reason = f"{type(error).__name__}:{error}"
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                jsonl.write(
                    json.dumps(
                        {
                            "schema": "g1-root-state-replay-v1",
                            "kind": "rejection",
                            "index": index,
                            "original_estimate_time_ns": observation.estimate_time_ns,
                            "reason": reason,
                        },
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n"
                )
                continue
            record = _packet_record(result)
            jsonl.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            binary.write(result.payload)
            packet_count += 1
            strict_count += int(result.packet.strictly_valid)
            positions.append(result.packet.position)
            linear_speeds.append(float(np.linalg.norm(result.packet.linear_velocity)))
            angular_speeds.append(float(np.linalg.norm(result.packet.angular_velocity)))

    metrics = {
        "schema": "g1-root-state-replay-metrics-v1",
        "source_revision": args.source_revision,
        "source_bag": str(bag),
        "input_odometry_messages": len(observations),
        "output_packets": packet_count,
        "strictly_valid_at_publish": strict_count,
        "strict_publish_fraction": 0.0 if packet_count == 0 else strict_count / packet_count,
        "rejection_counts": rejection_counts,
        "calibration_digest_sha256": contract.digest.hex(),
        "original_start_time_ns": observations[0].estimate_time_ns,
        "original_end_time_ns": observations[-1].estimate_time_ns,
        "position_min": np.min(np.asarray(positions), axis=0).tolist() if positions else None,
        "position_max": np.max(np.asarray(positions), axis=0).tolist() if positions else None,
        "linear_speed_max_mps": max(linear_speeds, default=None),
        "angular_speed_max_radps": max(angular_speeds, default=None),
        "evidence_scope": "mechanics_only_not_dynamic_ground_truth",
    }
    with args.metrics_json.open("x", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False))
    return 0 if packet_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
