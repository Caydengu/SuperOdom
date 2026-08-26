"""Versioned slow-lane map-correction wire contract.

The root-state and map-correction lanes are deliberately independent.  This
module owns only the command-incapable ``RVMAP001`` payload emitted by the map
localizer and consumed by robot-vlm.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from enum import IntFlag

import numpy as np

MAP_CORRECTION_V1_MAGIC = b"RVMAP001"
MAP_CORRECTION_V1_STRUCT = struct.Struct("<8sQQQQqqqqI17f32s")
MAP_CORRECTION_V1_NUM_BYTES = MAP_CORRECTION_V1_STRUCT.size


class MapCorrectionProtocolError(ValueError):
    """The map-correction payload violates the frozen wire contract."""


class MapCorrectionHealth(IntFlag):
    ICP_CONVERGED = 1 << 0
    INLIERS_VALID = 1 << 1
    OBSERVABILITY_VALID = 1 << 2
    MAP_IDENTITY_VALID = 1 << 3
    CLOCK_VALID = 1 << 4
    # The existing UI receipt has independently passed the operator pin,
    # bounded-ICP, identity, freshness, and observability gates. Consumers may
    # replace their map anchor for this packet; background tracking packets do
    # not carry this bit.
    OPERATOR_ANCHOR = 1 << 5


REQUIRED_MAP_CORRECTION_HEALTH = (
    MapCorrectionHealth.ICP_CONVERGED
    | MapCorrectionHealth.INLIERS_VALID
    | MapCorrectionHealth.OBSERVABILITY_VALID
    | MapCorrectionHealth.MAP_IDENTITY_VALID
    | MapCorrectionHealth.CLOCK_VALID
)


def _rotation_from_wxyz(quaternion: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = quaternion
    norm = math.sqrt(sum(value * value for value in quaternion))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-3):
        raise MapCorrectionProtocolError("quaternion_wxyz must be normalized")
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _wxyz_from_rotation(rotation: np.ndarray) -> tuple[float, float, float, float]:
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        values = (
            0.25 * scale,
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(1 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
            values = (
                (matrix[2, 1] - matrix[1, 2]) / scale,
                0.25 * scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
            )
        elif index == 1:
            scale = math.sqrt(1 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
            values = (
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                0.25 * scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
            )
        else:
            scale = math.sqrt(1 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
            values = (
                (matrix[1, 0] - matrix[0, 1]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                0.25 * scale,
            )
    quaternion = np.asarray(values, dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0:
        quaternion *= -1
    return tuple(float(value) for value in quaternion)


def _validate_transform(value: np.ndarray) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise MapCorrectionProtocolError("map_T_local must be one finite 4x4 transform")
    if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9):
        raise MapCorrectionProtocolError("map_T_local must be homogeneous")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise MapCorrectionProtocolError("map_T_local rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise MapCorrectionProtocolError("map_T_local rotation must be proper")
    return transform.copy()


@dataclass(frozen=True)
class MapCorrectionPacketV1:
    sequence: int
    map_epoch: int
    local_source_epoch: int
    map_version: int
    reference_time_ns: int
    evidence_time_ns: int
    application_time_ns: int
    publish_time_ns: int
    health_flags: MapCorrectionHealth
    map_T_local: np.ndarray
    fitness: float
    rmse_m: float
    min_eig: float
    cond_number: float
    covariance_diagonal: tuple[float, float, float, float, float, float]
    map_digest: bytes

    def __post_init__(self) -> None:
        if (
            min(
                self.sequence, self.map_epoch, self.local_source_epoch, self.map_version
            )
            < 1
        ):
            raise MapCorrectionProtocolError(
                "sequence, epochs, and version must be positive"
            )
        ordered = (
            self.reference_time_ns,
            self.evidence_time_ns,
            self.application_time_ns,
            self.publish_time_ns,
        )
        if min(ordered) < 0 or tuple(sorted(ordered)) != ordered:
            raise MapCorrectionProtocolError(
                "map times must satisfy reference <= evidence <= application <= publish"
            )
        _validate_transform(self.map_T_local)
        if len(self.map_digest) != 32:
            raise MapCorrectionProtocolError("map digest must contain 32 bytes")
        metrics = (self.fitness, self.rmse_m, self.min_eig, self.cond_number)
        if not all(math.isfinite(value) for value in metrics):
            raise MapCorrectionProtocolError("map metrics must be finite")
        if len(self.covariance_diagonal) != 6 or not all(
            math.isfinite(value) and value >= 0 for value in self.covariance_diagonal
        ):
            raise MapCorrectionProtocolError(
                "map covariance must contain six finite variances"
            )


def serialize_map_correction_v1(packet: MapCorrectionPacketV1) -> bytes:
    transform = _validate_transform(packet.map_T_local)
    position = tuple(float(value) for value in transform[:3, 3])
    quaternion = _wxyz_from_rotation(transform[:3, :3])
    return MAP_CORRECTION_V1_STRUCT.pack(
        MAP_CORRECTION_V1_MAGIC,
        packet.sequence,
        packet.map_epoch,
        packet.local_source_epoch,
        packet.map_version,
        packet.reference_time_ns,
        packet.evidence_time_ns,
        packet.application_time_ns,
        packet.publish_time_ns,
        int(packet.health_flags),
        *position,
        *quaternion,
        packet.fitness,
        packet.rmse_m,
        packet.min_eig,
        packet.cond_number,
        *packet.covariance_diagonal,
        packet.map_digest,
    )


def deserialize_map_correction_v1(payload: bytes) -> MapCorrectionPacketV1:
    if len(payload) != MAP_CORRECTION_V1_NUM_BYTES:
        raise MapCorrectionProtocolError("map-correction V1 packet length mismatch")
    values = MAP_CORRECTION_V1_STRUCT.unpack(payload)
    if values[0] != MAP_CORRECTION_V1_MAGIC:
        raise MapCorrectionProtocolError("map-correction V1 magic mismatch")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _rotation_from_wxyz(
        tuple(float(value) for value in values[13:17])
    )
    transform[:3, 3] = values[10:13]
    return MapCorrectionPacketV1(
        sequence=values[1],
        map_epoch=values[2],
        local_source_epoch=values[3],
        map_version=values[4],
        reference_time_ns=values[5],
        evidence_time_ns=values[6],
        application_time_ns=values[7],
        publish_time_ns=values[8],
        health_flags=MapCorrectionHealth(values[9]),
        map_T_local=transform,
        fitness=values[17],
        rmse_m=values[18],
        min_eig=values[19],
        cond_number=values[20],
        covariance_diagonal=tuple(float(value) for value in values[21:27]),
        map_digest=values[27],
    )


def map_transform_from_row_se2(
    rotation: np.ndarray, translation_m: np.ndarray
) -> np.ndarray:
    """Convert ``xy_map = xy_local @ R + t`` into ``map_T_local``."""
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation_m, dtype=np.float64)
    if rotation.shape != (2, 2) or translation.shape != (2,):
        raise MapCorrectionProtocolError(
            "row SE(2) requires a 2x2 rotation and 2-vector"
        )
    transform = np.eye(4, dtype=np.float64)
    transform[:2, :2] = rotation.T
    transform[:2, 3] = translation
    return _validate_transform(transform)
