#!/usr/bin/env python3
"""Replay registered KISS-local evidence through the production map engine."""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
from g1_root_state_bridge.map_protocol import serialize_map_correction_v1
from g1_root_state_bridge.structural_map_localization import (
    AutomaticMapCorrectionEngine,
    RegisteredMapEvidenceBuffer,
    StructuralMapConfig,
    StructuralMapLocalizationError,
    TimedLocalPose,
    TimedRegisteredCloud,
)
from g1_root_state_bridge.ui_initialization import load_ui_map_initialization


def _load_odometry(
    path: Path, topic: str, treatment: str | None
) -> list[dict[str, object]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            is_odometry = row.get("kind") == "odometry" and row.get("topic") == topic
            is_treatment = row.get("kind") == "pose" and row.get("treatment") == treatment
            if is_odometry:
                if row.get("frame_id") != "kiss_local":
                    raise ValueError("odometry frame must be kiss_local")
                rows.append(row)
            elif is_treatment:
                rows.append(
                    {
                        **row,
                        "source_time_ns": int(
                            row.get("source_time_ns", row["event_realtime_ns"])
                        ),
                        "frame_id": "kiss_local",
                    }
                )
    if len(rows) < 3:
        raise ValueError(f"missing odometry topic {topic!r}")
    time_ns = np.asarray([row["source_time_ns"] for row in rows], dtype=np.int64)
    if not np.all(np.diff(time_ns) > 0):
        raise ValueError("odometry source time did not strictly increase")
    return rows


def _json_safe(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--map-key", default="map_xy_all_5cm")
    parser.add_argument("--map-version", type=int, default=1)
    parser.add_argument("--map-epoch", type=int, default=1)
    parser.add_argument("--registered-clouds", type=Path, required=True)
    parser.add_argument("--odometry", type=Path, required=True)
    parser.add_argument("--odometry-topic", default="/g1/localization/pelvis_odom")
    parser.add_argument("--odometry-treatment")
    parser.add_argument("--local-source-epoch", type=int, required=True)
    parser.add_argument(
        "--initialization-receipt",
        type=Path,
        help="accepted Gio UI/headless bounded-Snap receipt; skips global search",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--packets-output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.packets_output.exists():
        raise FileExistsError("refusing to overwrite map-replay output")

    config = StructuralMapConfig()
    engine = AutomaticMapCorrectionEngine(
        map_path=args.map,
        map_key=args.map_key,
        map_version=args.map_version,
        map_epoch=args.map_epoch,
        config=config,
    )
    engine.bind_local_epoch(args.local_source_epoch)
    initialization = None
    if args.initialization_receipt is not None:
        initialization = load_ui_map_initialization(
            args.initialization_receipt,
            expected_structural_map_sha256=engine.map_digest.hex(),
            minimum_fitness=config.minimum_inlier_fraction,
            maximum_rmse_m=config.maximum_rmse_m,
            minimum_eigenvalue=config.minimum_observability_eigenvalue,
            maximum_condition_number=config.maximum_observability_condition_number,
        )
    buffer = RegisteredMapEvidenceBuffer(maximum_history_sec=15.0)
    odometry = _load_odometry(
        args.odometry, args.odometry_topic, args.odometry_treatment
    )
    odom_index = 0
    attempts: list[dict[str, object]] = []
    packets: list[dict[str, object]] = []
    last_attempt_ns = 0
    with np.load(args.registered_clouds, allow_pickle=False) as archive:
        points = np.asarray(archive["points_xyz_m"], dtype=np.float32)
        offsets = np.asarray(archive["cloud_offsets"], dtype=np.int64)
        cloud_time = np.asarray(archive["source_time_ns"], dtype=np.int64)
        frame = np.asarray(archive["frame_id"])
    for index, evidence_ns in enumerate(cloud_time):
        while odom_index < len(odometry) and int(
            odometry[odom_index]["source_time_ns"]
        ) <= int(evidence_ns):
            row = odometry[odom_index]
            buffer.append_pose(
                TimedLocalPose(
                    int(row["source_time_ns"]),
                    str(row["frame_id"]),
                    np.asarray(row["position_xyz_m"][:2], dtype=np.float64),
                )
            )
            odom_index += 1
        buffer.append_cloud(
            TimedRegisteredCloud(
                int(evidence_ns),
                str(frame[index]),
                points[offsets[index] : offsets[index + 1]],
            )
        )
        if (
            engine.rotation is None
            and initialization is not None
            and int(evidence_ns) < initialization.evidence_time_ns
        ):
            continue
        if engine.rotation is None:
            window_sec = config.global_window_sec
        else:
            if int(evidence_ns) - last_attempt_ns < round(
                config.tracking_period_sec * 1e9
            ):
                continue
            window_sec = config.tracking_window_sec
        end_to_end_start = time.perf_counter_ns()
        previous_rotation = None if engine.rotation is None else engine.rotation.copy()
        previous_translation = (
            None if engine.translation_m is None else engine.translation_m.copy()
        )
        previous_sequence = engine.sequence
        try:
            query, audit = buffer.structural_query(
                end_source_time_ns=int(evidence_ns),
                window_sec=window_sec,
                config=config,
            )
            if engine.rotation is None and initialization is not None:
                attempt = engine.initialize_from_ui(
                    map_T_local=initialization.map_T_local,
                    fitness=initialization.fitness,
                    rmse_m=initialization.rmse_m,
                    min_eig=initialization.min_eig,
                    cond_number=initialization.cond_number,
                    reference_time_ns=initialization.reference_time_ns,
                    evidence_time_ns=initialization.evidence_time_ns,
                    # Offline replay preserves the evidence ordering without
                    # pretending wall-clock latency from another session.
                    application_time_ns=initialization.evidence_time_ns + 1,
                )
            elif engine.rotation is None:
                attempt = engine.initialize(
                    query,
                    reference_time_ns=int(audit["start_source_time_ns"]),
                    evidence_time_ns=int(evidence_ns),
                    # The packet is replaced below with the measured replay age.
                    application_time_ns=int(evidence_ns) + 1,
                )
            else:
                pose = buffer.pose_at(
                    int(evidence_ns),
                    maximum_age_ns=config.maximum_root_evidence_gap_ns,
                )
                attempt = engine.track(
                    query,
                    local_position_xy_m=pose.position_xy_m,
                    reference_time_ns=int(audit["start_source_time_ns"]),
                    evidence_time_ns=int(evidence_ns),
                    application_time_ns=int(evidence_ns) + 1,
                )
        except StructuralMapLocalizationError:
            continue
        end_to_end_ms = (time.perf_counter_ns() - end_to_end_start) * 1e-6
        last_attempt_ns = int(evidence_ns)
        packet = attempt.packet
        rejection_reason = attempt.rejection_reason
        accepted = attempt.accepted
        if accepted and end_to_end_ms > config.maximum_input_age_ns * 1e-6:
            accepted = False
            packet = None
            rejection_reason = "map_evidence_stale_before_publication"
            engine.rotation = previous_rotation
            engine.translation_m = previous_translation
            engine.sequence = previous_sequence
        if packet is not None:
            application_ns = int(evidence_ns) + max(1, round(end_to_end_ms * 1e6))
            packet = dataclasses.replace(
                packet,
                application_time_ns=application_ns,
                publish_time_ns=application_ns,
            )
            payload = serialize_map_correction_v1(packet)
            packets.append(
                {
                    "schema": "g1_map_correction_replay_packet_v1",
                    "kind": attempt.kind,
                    "evidence_time_ns": int(evidence_ns),
                    "end_to_end_runtime_ms": end_to_end_ms,
                    "payload_hex": payload.hex(),
                }
            )
        icp = attempt.report.get("icp", attempt.report)
        attempts.append(
            {
                "kind": attempt.kind,
                "accepted": accepted,
                "rejection_reason": rejection_reason,
                "reference_time_ns": attempt.reference_time_ns,
                "evidence_time_ns": int(evidence_ns),
                "algorithm_runtime_ms": attempt.runtime_ms,
                "end_to_end_runtime_ms": end_to_end_ms,
                "query_audit": audit,
                "report": _json_safe(icp),
            }
        )
    if not attempts:
        raise ValueError("no map-correction attempt could be formed")
    runtimes = np.asarray([row["end_to_end_runtime_ms"] for row in attempts])
    report = {
        "schema": "g1_structural_map_correction_replay_v1",
        "status": "pass" if packets else "fail",
        "command_capability": "structurally_unavailable",
        "motive_online_input": False,
        "map_path": str(args.map),
        "map_key": args.map_key,
        "map_digest": engine.map_digest.hex(),
        "initialization_mode": (
            "ui_pin_locked" if initialization is not None else "automatic_global_search"
        ),
        "initialization_receipt": (
            None
            if args.initialization_receipt is None
            else str(args.initialization_receipt)
        ),
        "local_source_epoch": args.local_source_epoch,
        "attempt_count": len(attempts),
        "accepted_count": len(packets),
        "rejected_count": len(attempts) - len(packets),
        "end_to_end_runtime_ms": {
            "p50": float(np.quantile(runtimes, 0.50)),
            "p95": float(np.quantile(runtimes, 0.95)),
            "p99": float(np.quantile(runtimes, 0.99)),
            "maximum": float(np.max(runtimes)),
        },
        "maximum_input_age_ms": config.maximum_input_age_ns * 1e-6,
        "attempts": attempts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.packets_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.packets_output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in packets),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "packets_output": str(args.packets_output),
                "status": report["status"],
                "attempt_count": report["attempt_count"],
                "accepted_count": report["accepted_count"],
                "runtime_p95_ms": report["end_to_end_runtime_ms"]["p95"],
            },
            sort_keys=True,
        )
    )
    return 0 if packets else 1


if __name__ == "__main__":
    raise SystemExit(main())
