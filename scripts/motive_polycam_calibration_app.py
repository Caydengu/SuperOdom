#!/usr/bin/env python3
"""Pair a pivot-calibrated Motive pointer with clicks on the exact Polycam map."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import signal
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import viser

from fit_motive_map_transform import _content_sha256, fit_control_points


def _atomic_json(path: Path, value: dict) -> None:
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


class MotiveBuffer:
    def __init__(self, rigid_body_id: int, rigid_body_name: str) -> None:
        self.rigid_body_id = int(rigid_body_id)
        self.rigid_body_name = rigid_body_name
        self.lock = threading.Lock()
        self.rows: deque[tuple[int, np.ndarray, np.ndarray, float]] = deque()
        self.frames = 0
        self.valid = 0

    def append(self, body, now_ns: int) -> None:
        with self.lock:
            self.frames += 1
            if body is None or not bool(body.tracking_valid):
                return
            self.rows.append(
                (
                    now_ns,
                    np.asarray(body.pos, dtype=np.float64),
                    np.asarray(body.rot, dtype=np.float64),
                    float(getattr(body, "error", 0.0)),
                )
            )
            cutoff = now_ns - 3_000_000_000
            while self.rows and self.rows[0][0] < cutoff:
                self.rows.popleft()
            self.valid += 1

    def _stable_rows(self, window_sec: float) -> tuple[list, int, int]:
        cutoff = time.monotonic_ns() - round(window_sec * 1e9)
        with self.lock:
            selected = list(self.rows)
            frames, valid = self.frames, self.valid
        selected = [row for row in selected if row[0] >= cutoff]
        minimum = max(20, round(window_sec * 60.0))
        if len(selected) < minimum:
            raise ValueError(
                f"only {len(selected)} valid {self.rigid_body_name} frames "
                f"in the last {window_sec:.1f}s"
            )
        return selected, frames, valid

    def stable_position(self, window_sec: float, max_std_m: float) -> tuple[np.ndarray, dict]:
        selected, frames, valid = self._stable_rows(window_sec)
        positions = np.asarray([row[1] for row in selected], dtype=np.float64)
        standard_deviation = np.std(positions, axis=0)
        if float(np.max(standard_deviation)) > max_std_m:
            raise ValueError(
                f"pointer is moving: maximum axis std={float(np.max(standard_deviation))*1000:.1f} mm"
            )
        return np.median(positions, axis=0), {
            "sample_count": len(positions),
            "axis_std_m": standard_deviation.tolist(),
            "marker_error_m_median": float(np.median([row[3] for row in selected])),
            "tracking_valid_frames": valid,
            "frames_seen": frames,
        }

    def stable_map_yaw(
        self,
        map_R_motive: np.ndarray,
        *,
        window_sec: float,
        max_position_std_m: float,
        max_yaw_std_deg: float,
    ) -> tuple[float, dict]:
        selected, frames, valid = self._stable_rows(window_sec)
        positions = np.asarray([row[1] for row in selected], dtype=np.float64)
        position_std = np.std(positions, axis=0)
        if float(np.max(position_std)) > max_position_std_m:
            raise ValueError(
                "G1 pelvis is moving: maximum position std="
                f"{float(np.max(position_std))*1000:.1f} mm"
            )
        yaws = []
        for row in selected:
            x, y, z, w = np.asarray(row[2], dtype=np.float64)
            norm = math.sqrt(x * x + y * y + z * z + w * w)
            if norm <= 1e-8:
                raise ValueError("Motive emitted a zero-norm pelvis quaternion")
            x, y, z, w = x / norm, y / norm, z / norm, w / norm
            rotation = np.asarray(
                (
                    (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
                    (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
                    (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
                ),
                dtype=np.float64,
            )
            forward_map = np.asarray(map_R_motive, dtype=np.float64) @ rotation[:, 0]
            yaws.append(math.atan2(float(forward_map[1]), float(forward_map[0])))
        yaws = np.asarray(yaws, dtype=np.float64)
        mean_yaw = math.atan2(float(np.mean(np.sin(yaws))), float(np.mean(np.cos(yaws))))
        residual = np.arctan2(np.sin(yaws - mean_yaw), np.cos(yaws - mean_yaw))
        yaw_std_deg = float(np.degrees(np.std(residual)))
        if yaw_std_deg > max_yaw_std_deg:
            raise ValueError(f"G1 pelvis yaw is moving: circular std={yaw_std_deg:.2f} deg")
        return mean_yaw, {
            "sample_count": len(selected),
            "position_axis_std_m": position_std.tolist(),
            "raw_map_yaw_std_deg": yaw_std_deg,
            "marker_error_m_median": float(np.median([row[3] for row in selected])),
            "tracking_valid_frames": valid,
            "frames_seen": frames,
        }


class MotiveStreams:
    def __init__(self, *buffers: MotiveBuffer) -> None:
        self.buffers = tuple(buffers)

    def callback(self, data) -> None:
        bodies = getattr(data.get("rigid_body_data"), "rigid_body_list", ())
        by_id = {int(body.id_num): body for body in bodies}
        now_ns = time.monotonic_ns()
        for buffer in self.buffers:
            buffer.append(by_id.get(buffer.rigid_body_id), now_ns)


class CalibrationApp:
    def __init__(
        self,
        args,
        client,
        pointer: MotiveBuffer,
        pelvis: MotiveBuffer,
    ) -> None:
        self.args = args
        self.client = client
        self.pointer = pointer
        self.pelvis = pelvis
        sys.path.insert(0, str(args.localize_dir.resolve()))
        from mesh_map import load_mesh_map

        self.mesh = load_mesh_map(
            str(args.glb), up="y", sample_count=args.sample_count,
            sample_seed=args.sample_seed, sample_cache=str(args.surface_cache), verbose=True,
        )
        if self.mesh.source_sha256.lower() != args.glb_sha256.lower():
            raise ValueError(
                "loaded GLB identity differs from --glb-sha256: "
                f"{self.mesh.source_sha256} != {args.glb_sha256.lower()}"
            )
        if self.mesh.surface_sha256.lower() != args.surface_sha256.lower():
            raise ValueError(
                "loaded surface identity differs from --surface-sha256: "
                f"{self.mesh.surface_sha256} != {args.surface_sha256.lower()}"
            )
        self.server = viser.ViserServer(host="0.0.0.0", port=args.port)
        self.mesh_handle = self.server.scene.add_mesh_trimesh("/polycam-map", self.mesh.mesh)
        self.mesh_handle.on_click(self._on_mesh_click)
        self.armed = False
        self.heading_clicks: list[np.ndarray] = []
        self.transform_result: dict | None = None
        self.points = []
        self.stop = threading.Event()
        self.status = self.server.gui.add_markdown("Waiting for the Motive pointer...")
        self.label = self.server.gui.add_text("Landmark label", initial_value="p1")
        self.role = self.server.gui.add_dropdown("Role", options=("fit", "holdout"), initial_value="fit")
        self.arm = self.server.gui.add_button("Arm next map click")
        self.finalize = self.server.gui.add_button("Fit + validate Motive → map")
        self.arm_heading = self.server.gui.add_button("Arm G1 forward heading (2 clicks)")
        self.arm.on_click(self._arm)
        self.finalize.on_click(self._finalize)
        self.arm_heading.on_click(self._arm_heading)
        self._load_existing()

    def _identity(self) -> dict:
        return {
            "glb_sha256": self.args.glb_sha256.lower(),
            "surface_sha256": self.args.surface_sha256.lower(),
            "structural_map_sha256": self.args.structural_map_sha256.lower(),
        }

    def _value(self) -> dict:
        return {
            "schema": "g1_motive_polycam_control_points_v1",
            "map_identity": self._identity(),
            "motive_contract": {
                "up_axis": "y", "horizontal_axes": ["x", "z"],
                "rigid_body_id": self.args.rigid_body_id,
                "rigid_body_name": self.args.rigid_body_name,
                "pointer_contract": "Motive rigid-body origin is pivot-calibrated to the physical tip",
            },
            "points": self.points,
        }

    def _load_existing(self) -> None:
        if not self.args.control_points.exists():
            return
        value = json.loads(self.args.control_points.read_text(encoding="utf-8"))
        if value.get("map_identity") != self._identity():
            raise ValueError("existing control points belong to another map identity")
        self.points = list(value.get("points", []))
        for row in self.points:
            self.server.scene.add_icosphere(
                f"/control-points/{row['label']}", radius=0.07, position=row["map_xyz_m"],
                color=(49, 208, 170) if row["role"] == "fit" else (251, 191, 36),
            )
        if self.args.transform_output.exists():
            result = json.loads(self.args.transform_output.read_text(encoding="utf-8"))
            if result.get("map_identity") != self._identity():
                raise ValueError("existing transform belongs to another map identity")
            recorded_digest = result.get("content_sha256")
            unsigned = {
                key: value for key, value in result.items() if key != "content_sha256"
            }
            if recorded_digest != _content_sha256(unsigned):
                raise ValueError("existing transform content digest is invalid")
            if result.get("control_points_sha256") != _content_sha256(value):
                raise ValueError("existing transform does not match the saved control points")
            self.transform_result = result

    def _arm(self, _event) -> None:
        label = self.label.value.strip()
        if not label:
            self.status.content = "### 🔴 Enter a nonempty landmark label."
            return
        if any(row["label"] == label for row in self.points):
            self.status.content = f"### 🔴 Label `{label}` already exists."
            return
        self.heading_clicks = []
        self.armed = True
        self.status.content = (
            f"### 🟡 Armed `{label}`\nHold the pivot-calibrated tip still on that floor landmark, "
            "then click the same location on the Polycam floor."
        )

    def _map_floor_click(self, event) -> np.ndarray:
        origin = np.asarray(event.ray_origin, dtype=np.float64)
        direction = np.asarray(event.ray_direction, dtype=np.float64)
        floor_z = float(self.mesh.meta.floor_z)
        if abs(direction[2]) < 1e-8:
            raise ValueError("click ray is parallel to the map floor")
        distance = (floor_z - origin[2]) / direction[2]
        if distance <= 0.0:
            raise ValueError("click ray does not intersect the map floor in front of the camera")
        return origin + distance * direction

    def _on_mesh_click(self, event) -> None:
        if not self.armed and not self.heading_clicks:
            return
        try:
            map_point = self._map_floor_click(event)
            if self.heading_clicks:
                self._heading_click(map_point)
                return
            motive_point, stability = self.pointer.stable_position(
                self.args.motive_window_sec, self.args.maximum_pointer_std_m
            )
            row = {
                "label": self.label.value.strip(), "role": self.role.value,
                "map_xyz_m": [float(map_point[0]), float(map_point[1]), float(map_point[2])],
                "motive_xyz_m": [float(value) for value in motive_point],
                "motive_stability": stability, "captured_realtime_ns": time.time_ns(),
            }
            self.points.append(row)
            _atomic_json(self.args.control_points, self._value())
            self.server.scene.add_icosphere(
                f"/control-points/{row['label']}", radius=0.07, position=row["map_xyz_m"],
                color=(49, 208, 170) if row["role"] == "fit" else (251, 191, 36),
            )
            self.armed = False
            self.label.value = f"p{len(self.points) + 1}"
            self.status.content = (
                f"### ✅ Saved `{row['label']}` ({row['role']})\n"
                f"Motive={np.round(motive_point, 3).tolist()}  map={np.round(map_point, 3).tolist()}  "
                f"samples={stability['sample_count']}"
            )
        except Exception as error:
            self.armed = False
            self.heading_clicks = []
            self.status.content = f"### 🔴 Pair rejected\n{type(error).__name__}: {error}"

    def _arm_heading(self, _event) -> None:
        if self.transform_result is None or self.transform_result.get("status") != "accepted":
            self.status.content = "### 🔴 Fit and accept the Motive→map transform first."
            return
        self.armed = False
        self.heading_clicks = [np.asarray((np.nan, np.nan, np.nan))]
        self.status.content = (
            "### 🟣 G1 heading armed\nKeep G1-4123 stationary. Click its pelvis position "
            "on the map, then click a point directly in front of it."
        )

    def _heading_click(self, map_point: np.ndarray) -> None:
        if len(self.heading_clicks) == 1 and np.isnan(self.heading_clicks[0]).all():
            self.heading_clicks = [map_point]
            self.status.content = "### 🟣 Pelvis point saved\nNow click a point directly in front of G1-4123."
            return
        origin = self.heading_clicks[0]
        direction = np.asarray(map_point[:2] - origin[:2], dtype=np.float64)
        if float(np.linalg.norm(direction)) < 0.25:
            raise ValueError("heading click must be at least 0.25 m from the pelvis click")
        clicked_yaw = math.atan2(float(direction[1]), float(direction[0]))
        transform = np.asarray(self.transform_result["T_map_from_motive"], dtype=np.float64)
        raw_yaw, stability = self.pelvis.stable_map_yaw(
            transform[:3, :3],
            window_sec=self.args.motive_window_sec,
            max_position_std_m=self.args.maximum_pelvis_std_m,
            max_yaw_std_deg=self.args.maximum_pelvis_yaw_std_deg,
        )
        yaw_offset = math.atan2(
            math.sin(clicked_yaw - raw_yaw), math.cos(clicked_yaw - raw_yaw)
        )
        result = dict(self.transform_result)
        result.pop("content_sha256", None)
        result["pelvis_heading"] = {
            "rigid_body_id": self.args.pelvis_rigid_body_id,
            "rigid_body_name": self.args.pelvis_rigid_body_name,
            "forward_axis": "+x",
            "clicked_map_pelvis_xyz_m": [float(value) for value in origin],
            "clicked_map_forward_xyz_m": [float(value) for value in map_point],
            "raw_map_yaw_rad": raw_yaw,
            "clicked_map_yaw_rad": clicked_yaw,
            "yaw_offset_rad": yaw_offset,
            "stability": stability,
            "calibration_role": "independent evaluator heading; no estimator input",
        }
        result["evaluator_ready"] = True
        result["content_sha256"] = _content_sha256(result)
        _atomic_json(self.args.transform_output, result)
        self.transform_result = result
        self.heading_clicks = []
        self.server.scene.add_icosphere(
            "/pelvis-heading/origin", radius=0.09, position=origin, color=(168, 85, 247)
        )
        self.status.content = (
            "### ✅ Motive evaluator transform ready\n"
            f"Pelvis asset yaw offset={math.degrees(yaw_offset):.2f}°; "
            f"raw stability={stability['raw_map_yaw_std_deg']:.2f}°."
        )

    def _finalize(self, _event) -> None:
        try:
            result = fit_control_points(self._value())
            result["evaluator_ready"] = False
            result["content_sha256"] = _content_sha256(
                {key: value for key, value in result.items() if key != "content_sha256"}
            )
            self.transform_result = result
            _atomic_json(self.args.transform_output, result)
            metrics = result["metrics"]
            icon = "✅" if result["status"] == "accepted" else "🔴"
            self.status.content = (
                f"### {icon} Transform {result['status']}\n"
                f"fit RMSE={metrics['fit_rmse_m']*100:.1f} cm; "
                f"holdout max={metrics['holdout_max_m']*100:.1f} cm; "
                f"LOO max={metrics['leave_one_out_max_m']*100:.1f} cm; "
                f"scale={metrics['pairwise_scale_median']:.4f}; failures={result['failures']}"
            )
            if result["status"] == "accepted":
                self.status.content += "\nNow arm and set the independent G1 forward heading."
        except Exception as error:
            self.status.content = f"### 🔴 Calibration incomplete\n{type(error).__name__}: {error}"

    def run(self) -> None:
        print(f"Motive/Polycam calibration UI: http://localhost:{self.args.port}", flush=True)
        while not self.stop.wait(0.25):
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--localize-dir", type=Path, required=True)
    parser.add_argument("--glb", type=Path, required=True)
    parser.add_argument("--surface-cache", type=Path, required=True)
    parser.add_argument("--glb-sha256", required=True)
    parser.add_argument("--surface-sha256", required=True)
    parser.add_argument("--structural-map-sha256", required=True)
    parser.add_argument("--sdk-root", type=Path, required=True)
    parser.add_argument("--server-address", required=True)
    parser.add_argument("--client-address", required=True)
    parser.add_argument("--rigid-body-id", type=int, required=True)
    parser.add_argument("--rigid-body-name", required=True)
    parser.add_argument("--pelvis-rigid-body-id", type=int, default=42)
    parser.add_argument("--pelvis-rigid-body-name", default="G1_PELVIS_F_4123")
    parser.add_argument("--control-points", type=Path, required=True)
    parser.add_argument("--transform-output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8083)
    parser.add_argument("--sample-count", type=int, default=500_000)
    parser.add_argument("--sample-seed", type=int, default=24)
    parser.add_argument("--motive-window-sec", type=float, default=1.5)
    parser.add_argument("--maximum-pointer-std-m", type=float, default=0.004)
    parser.add_argument("--maximum-pelvis-std-m", type=float, default=0.006)
    parser.add_argument("--maximum-pelvis-yaw-std-deg", type=float, default=0.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    package_root = args.sdk_root / "mocap_utils"
    if not package_root.is_dir():
        raise FileNotFoundError(f"NatNet SDK package is missing: {package_root}")
    sys.path[:0] = [str(package_root), str(args.sdk_root)]
    natnet = importlib.import_module("mocap_utils.natnet_client")
    client = natnet.NatNetClient()
    client.set_client_address(args.client_address)
    client.set_server_address(args.server_address)
    client.set_use_multicast(True)
    if hasattr(client, "set_print_level"):
        client.set_print_level(0)
    pointer = MotiveBuffer(args.rigid_body_id, args.rigid_body_name)
    pelvis = MotiveBuffer(args.pelvis_rigid_body_id, args.pelvis_rigid_body_name)
    streams = MotiveStreams(pointer, pelvis)
    client.new_frame_listener = streams.callback
    if not client.run():
        raise RuntimeError("NatNet client failed to start")
    client.data_socket.settimeout(0.1)
    client.command_socket.settimeout(0.1)
    app = CalibrationApp(args, client, pointer, pelvis)

    def stop(_signum, _frame):
        app.stop.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        app.run()
    finally:
        client.stop_threads = True
        for stream_socket in (client.command_socket, client.data_socket):
            stream_socket.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
