#!/usr/bin/env python3
"""Convert a metric Y-up Polycam GLB into static planar localization maps."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np


COMPONENT_DTYPES = {
    5120: np.dtype("<i1"),
    5121: np.dtype("<u1"),
    5122: np.dtype("<i2"),
    5123: np.dtype("<u2"),
    5125: np.dtype("<u4"),
    5126: np.dtype("<f4"),
}
TYPE_WIDTHS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def load_glb(path: Path) -> tuple[dict[str, object], bytes]:
    raw = path.read_bytes()
    magic, version, declared_length = struct.unpack_from("<4sII", raw, 0)
    if magic != b"glTF" or version != 2 or declared_length != len(raw):
        raise ValueError("invalid or unsupported GLB")
    chunks: dict[int, bytes] = {}
    offset = 12
    while offset < len(raw):
        length, kind = struct.unpack_from("<II", raw, offset)
        offset += 8
        chunks[kind] = raw[offset : offset + length]
        offset += length
    document = json.loads(chunks[0x4E4F534A].decode("utf-8").rstrip("\x00 \t\r\n"))
    return document, chunks[0x004E4942]


def read_accessor(document: dict[str, object], binary: bytes, index: int) -> np.ndarray:
    accessor = document["accessors"][index]
    view = document["bufferViews"][accessor["bufferView"]]
    dtype = COMPONENT_DTYPES[accessor["componentType"]]
    width = TYPE_WIDTHS[accessor["type"]]
    count = int(accessor["count"])
    offset = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    element_bytes = dtype.itemsize * width
    stride = int(view.get("byteStride", element_bytes))
    if stride == element_bytes:
        return np.frombuffer(
            binary, dtype=dtype, count=count * width, offset=offset
        ).reshape(count, width).copy()
    output = np.empty((count, width), dtype=dtype)
    for row in range(count):
        output[row] = np.frombuffer(
            binary, dtype=dtype, count=width, offset=offset + row * stride
        )
    return output


def voxelize_xy(points: np.ndarray, resolution_m: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not points.size:
        raise ValueError("voxelization requires a nonempty Nx2 cloud")
    cells = np.rint(points / float(resolution_m)).astype(np.int64)
    _, first = np.unique(cells, axis=0, return_index=True)
    return points[np.sort(first)].astype(np.float32)


def build_maps(
    glb: Path,
    *,
    floor_y_m: float,
    minimum_height_m: float,
    maximum_height_m: float,
    vertical_tolerance_deg: float,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    document, binary = load_glb(glb)
    vertices_all: list[np.ndarray] = []
    centroids_all: list[np.ndarray] = []
    normals_all: list[np.ndarray] = []
    for mesh in document.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            if primitive.get("mode", 4) != 4:
                continue
            vertices = read_accessor(
                document, binary, primitive["attributes"]["POSITION"]
            ).astype(np.float64)
            indices = read_accessor(document, binary, primitive["indices"]).reshape(-1, 3)
            triangle = vertices[indices]
            cross = np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])
            norm = np.linalg.norm(cross, axis=1)
            valid = norm > 1e-12
            normals = np.zeros_like(cross)
            normals[valid] = cross[valid] / norm[valid, None]
            vertices_all.append(vertices)
            centroids_all.append(triangle.mean(axis=1))
            normals_all.append(normals)
    vertices = np.concatenate(vertices_all)
    centroids = np.concatenate(centroids_all)
    normals = np.concatenate(normals_all)

    vertex_height = vertices[:, 1] - floor_y_m
    vertex_gate = (
        np.isfinite(vertices).all(axis=1)
        & (vertex_height >= minimum_height_m)
        & (vertex_height <= maximum_height_m)
    )
    centroid_height = centroids[:, 1] - floor_y_m
    vertical_gate = (
        np.isfinite(centroids).all(axis=1)
        & (centroid_height >= minimum_height_m)
        & (centroid_height <= maximum_height_m)
        & (np.abs(normals[:, 1]) <= math.sin(math.radians(vertical_tolerance_deg)))
    )
    # GLB is Y-up. Its X/Z coordinates become localization map X/Y; global
    # handedness is resolved by the explicit yaw search rather than assumed.
    all_xy = vertices[vertex_gate][:, (0, 2)]
    structural_xy = centroids[vertical_gate][:, (0, 2)]
    arrays = {
        "map_xy_structural_2cm": voxelize_xy(structural_xy, 0.02),
        "map_xy_structural_5cm": voxelize_xy(structural_xy, 0.05),
        "map_xy_all_5cm": voxelize_xy(all_xy, 0.05),
    }
    digest = hashlib.sha256(glb.read_bytes()).hexdigest()
    metadata = {
        "schema": "polycam_structural_planar_map_v1",
        "source_glb": str(glb),
        "source_sha256": digest,
        "map_id": f"src-fieldbay-polycam-{digest[:12]}",
        "coordinate_contract": "source X/Z -> map X/Y; source Y-floor -> height",
        "floor_y_m": floor_y_m,
        "height_range_m": [minimum_height_m, maximum_height_m],
        "vertical_tolerance_deg": vertical_tolerance_deg,
        "source_vertex_count": int(vertices.shape[0]),
        "source_triangle_count": int(centroids.shape[0]),
        "admitted_vertex_count": int(np.sum(vertex_gate)),
        "admitted_vertical_triangle_count": int(np.sum(vertical_gate)),
        "output_counts": {name: int(value.shape[0]) for name, value in arrays.items()},
        "map_role": "candidate prior; not ground truth",
    }
    return arrays, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glb", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--floor-y-m", type=float, default=-1.4825164413452143)
    parser.add_argument("--minimum-height-m", type=float, default=0.15)
    parser.add_argument("--maximum-height-m", type=float, default=2.85)
    parser.add_argument("--vertical-tolerance-deg", type=float, default=20.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    arrays, metadata = build_maps(
        args.glb,
        floor_y_m=args.floor_y_m,
        minimum_height_m=args.minimum_height_m,
        maximum_height_m=args.maximum_height_m,
        vertical_tolerance_deg=args.vertical_tolerance_deg,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    rendered = json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
