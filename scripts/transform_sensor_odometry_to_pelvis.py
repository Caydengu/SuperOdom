#!/usr/bin/env python3
"""Apply the frozen G1 waist FK and initial-frame normalization to odometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from g1_root_state_bridge.amo_dataset import load_lowstate
from g1_root_state_bridge.amo_treatments import rotation_from_xyzw, xyzw_from_rotation
from g1_root_state_bridge.waist_kinematics import (
    pelvis_T_mid360,
    sensor_local_pose_to_pelvis_local_pose,
)


def _nearest_indices(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    right = np.clip(np.searchsorted(reference, query, side="left"), 0, reference.size - 1)
    left = np.clip(right - 1, 0, reference.size - 1)
    return np.where(np.abs(query - reference[left]) <= np.abs(reference[right] - query), left, right)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-treatments", type=Path, required=True)
    parser.add_argument("--source-treatment", default="kiss_icp_sensor")
    parser.add_argument("--output-treatment", default="kiss_icp_dynamic_fk_pelvis")
    parser.add_argument("--lowstate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-lowstate-age-ms", type=float, default=10.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    metadata: dict[str, object] | None = None
    records: list[dict[str, object]] = []
    with args.input_treatments.open(encoding="utf-8") as stream:
        for line in stream:
            payload = json.loads(line)
            if payload.get("kind") == "metadata":
                metadata = payload
            else:
                records.append(payload)
    source = [row for row in records if row.get("treatment") == args.source_treatment]
    if metadata is None or not source:
        raise ValueError("input must contain metadata and the requested source treatment")
    lowstate = load_lowstate(args.lowstate)
    event_time = np.asarray([row["event_realtime_ns"] for row in source], dtype=np.int64)
    overlapping = (event_time >= lowstate.oslo_event_ns[0]) & (
        event_time <= lowstate.oslo_event_ns[-1]
    )
    if float(np.mean(overlapping)) < 0.95:
        raise ValueError(f"lowstate temporal overlap below 95%: {float(np.mean(overlapping)):.6f}")
    source = [row for row, keep in zip(source, overlapping) if keep]
    event_time = event_time[overlapping]
    indices = _nearest_indices(lowstate.oslo_event_ns, event_time)
    age_ns = np.abs(event_time - lowstate.oslo_event_ns[indices])
    admitted = age_ns <= args.maximum_lowstate_age_ms * 1e6
    if float(np.mean(admitted)) < 0.99:
        raise ValueError(f"lowstate coverage below 99%: {float(np.mean(admitted)):.6f}")

    source = [row for row, keep in zip(source, admitted) if keep]
    indices = indices[admitted]
    age_ns = age_ns[admitted]
    initial_joint = lowstate.joint_position[indices[0]]
    pelvis0_T_sensor0 = pelvis_T_mid360(*initial_joint[12:15])
    derived: list[dict[str, object]] = []
    for row, low_index, lowstate_age_ns in zip(source, indices, age_ns):
        position = np.asarray(row["position_xyz_m"], dtype=np.float64)
        sensor0_T_sensor_t = np.eye(4)
        sensor0_T_sensor_t[:3, :3] = rotation_from_xyzw(
            np.asarray(row["quaternion_xyzw"], dtype=np.float64)
        )
        sensor0_T_sensor_t[:3, 3] = position
        joint = lowstate.joint_position[low_index]
        pelvis_t_T_sensor_t = pelvis_T_mid360(*joint[12:15])
        pelvis0_T_pelvis_t = sensor_local_pose_to_pelvis_local_pose(
            sensor0_T_sensor_t,
            pelvis0_T_sensor0=pelvis0_T_sensor0,
            pelvis_t_T_sensor_t=pelvis_t_T_sensor_t,
        )
        transformed = dict(row)
        transformed.update(
            treatment=args.output_treatment,
            frame_id="initial_pelvis_local",
            child_frame_id="pelvis",
            position_xyz_m=[float(value) for value in pelvis0_T_pelvis_t[:3, 3]],
            quaternion_xyzw=list(xyzw_from_rotation(pelvis0_T_pelvis_t[:3, :3])),
            source_lowstate_sequence=int(lowstate.sequence[low_index]),
            source_lowstate_age_ns=int(lowstate_age_ns),
        )
        derived.append(transformed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({
            **metadata,
            "alignment_reference_treatment": args.output_treatment,
            "pelvis_fk": {
                "schema": "g1_mid360_pelvis_fk_v1",
                "urdf_contract": "g1_pelvis_mid360_kinematics.urdf",
                "initial_world_frame": "pelvis_at_first_admitted_odometry_sample",
                "motive_online_input": False,
                "lowstate": str(args.lowstate),
                "maximum_lowstate_age_ms": args.maximum_lowstate_age_ms,
                "coverage_fraction": float(np.mean(admitted)),
                "temporal_overlap_fraction": float(np.mean(overlapping)),
            },
        }, sort_keys=True) + "\n")
        for row in records:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        for row in derived:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "source_count": len(source),
        "derived_count": len(derived),
        "coverage_fraction": float(np.mean(admitted)),
        "temporal_overlap_fraction": float(np.mean(overlapping)),
        "alignment_reference_treatment": args.output_treatment,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
