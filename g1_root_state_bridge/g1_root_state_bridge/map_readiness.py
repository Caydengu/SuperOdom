"""Wait for one fresh, healthy, identity-matched RVMAP001 correction."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from g1_root_state_bridge.map_protocol import (
    REQUIRED_MAP_CORRECTION_HEALTH,
    MapCorrectionPacketV1,
    deserialize_map_correction_v1,
)


def readiness_rejection_reason(
    packet: MapCorrectionPacketV1,
    *,
    expected_map_digest: bytes,
    expected_local_source_epoch: int,
    now_ns: int,
    maximum_age_ns: int,
) -> str | None:
    """Return the first fail-closed readiness violation, or ``None``."""

    if packet.map_digest != expected_map_digest:
        return "map_digest"
    if packet.local_source_epoch != expected_local_source_epoch:
        return "local_source_epoch"
    if (
        packet.health_flags & REQUIRED_MAP_CORRECTION_HEALTH
    ) != REQUIRED_MAP_CORRECTION_HEALTH:
        return "map_health"
    if packet.publish_time_ns > now_ns:
        return "publish_time_in_future"
    if now_ns - packet.publish_time_ns > maximum_age_ns:
        return "map_packet_stale"
    return None


def _digest(value: str) -> bytes:
    try:
        digest = bytes.fromhex(value)
    except ValueError as error:
        raise ValueError("map SHA-256 must be hexadecimal") from error
    if len(digest) != 32:
        raise ValueError("map SHA-256 must contain 32 bytes")
    return digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5577")
    parser.add_argument("--map-sha256", required=True)
    parser.add_argument("--local-source-epoch", type=int, required=True)
    parser.add_argument("--timeout-sec", type=float, default=40.0)
    parser.add_argument("--maximum-age-ms", type=float, default=500.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.local_source_epoch < 1:
        parser.error("local source epoch must be positive")
    if args.timeout_sec <= 0.0 or args.maximum_age_ms <= 0.0:
        parser.error("timeouts and maximum age must be positive")
    return args


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    expected_digest = _digest(args.map_sha256)
    maximum_age_ns = round(args.maximum_age_ms * 1e6)

    import zmq

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(args.endpoint)
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    deadline_ns = time.monotonic_ns() + round(args.timeout_sec * 1e9)
    rejections: Counter[str] = Counter()
    try:
        while time.monotonic_ns() < deadline_ns:
            remaining_ms = max(1, (deadline_ns - time.monotonic_ns()) // 1_000_000)
            if not dict(poller.poll(int(remaining_ms))).get(socket, 0) & zmq.POLLIN:
                continue
            try:
                packet = deserialize_map_correction_v1(socket.recv())
            except ValueError:
                rejections["wire_protocol"] += 1
                continue
            receipt_ns = time.time_ns()
            reason = readiness_rejection_reason(
                packet,
                expected_map_digest=expected_digest,
                expected_local_source_epoch=args.local_source_epoch,
                now_ns=receipt_ns,
                maximum_age_ns=maximum_age_ns,
            )
            if reason is not None:
                rejections[reason] += 1
                continue
            report = {
                "schema": "g1_structural_map_readiness_v1",
                "status": "ready",
                "endpoint": args.endpoint,
                "sequence": packet.sequence,
                "map_epoch": packet.map_epoch,
                "map_version": packet.map_version,
                "local_source_epoch": packet.local_source_epoch,
                "reference_time_ns": packet.reference_time_ns,
                "evidence_time_ns": packet.evidence_time_ns,
                "application_time_ns": packet.application_time_ns,
                "publish_time_ns": packet.publish_time_ns,
                "receipt_time_ns": receipt_ns,
                "packet_age_at_probe_ms": (receipt_ns - packet.publish_time_ns) * 1e-6,
                "health_flags": int(packet.health_flags),
                "fitness": packet.fitness,
                "rmse_m": packet.rmse_m,
                "minimum_eigenvalue": packet.min_eig,
                "condition_number": packet.cond_number,
                "map_digest": packet.map_digest.hex(),
                "rejections": dict(rejections),
                "command_capability": "structurally_unavailable",
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(report, sort_keys=True))
            return 0
    finally:
        socket.close()
        context.term()
    raise TimeoutError(
        f"no fresh, healthy, identity-matched RVMAP001 packet from "
        f"{args.endpoint} within {args.timeout_sec:.1f}s; "
        f"rejections={dict(rejections)}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
