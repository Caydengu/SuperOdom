"""Deterministic identity for the fixed G1 root-state calibration contract."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re

import numpy as np

from g1_root_state_bridge.bridge_core import ESTIMATOR_SEMANTICS_VERSION
from g1_root_state_bridge.kinematics import KINEMATICS_URDF_PATH


CALIBRATION_CONTRACT_SCHEMA = "g1-root-state-calibration-v1"


@dataclass(frozen=True)
class CalibrationContract:
    digest: bytes
    manifest: dict[str, str]
    imu_T_lidar: np.ndarray


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _opencv_matrix(text: str, key: str, rows: int, columns: int) -> np.ndarray:
    block = re.search(
        rf"{re.escape(key)}\s*:\s*!!opencv-matrix(?P<body>.*?)(?=\n\S|\Z)",
        text,
        flags=re.DOTALL,
    )
    if block is None:
        raise ValueError(f"calibration has no {key!r} OpenCV matrix")
    values_match = re.search(r"data\s*:\s*\[(?P<values>.*?)\]", block.group("body"), re.DOTALL)
    if values_match is None:
        raise ValueError(f"calibration matrix {key!r} has no data array")
    values = [
        float(token)
        for token in re.findall(
            r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
            values_match.group("values"),
        )
    ]
    if len(values) != rows * columns:
        raise ValueError(
            f"calibration matrix {key!r} has {len(values)} values, expected {rows * columns}"
        )
    return np.asarray(values, dtype=np.float64).reshape(rows, columns)


def load_calibration_contract(
    *,
    estimator_config_path: str | Path,
    estimator_calibration_path: str | Path,
    kinematics_urdf_path: str | Path = KINEMATICS_URDF_PATH,
) -> CalibrationContract:
    estimator_config = Path(estimator_config_path).resolve()
    estimator_calibration = Path(estimator_calibration_path).resolve()
    kinematics_urdf = Path(kinematics_urdf_path).resolve()
    for name, path in (
        ("estimator config", estimator_config),
        ("estimator calibration", estimator_calibration),
        ("kinematics URDF", kinematics_urdf),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{name} is missing: {path}")

    calibration_text = estimator_calibration.read_text(encoding="utf-8")
    imu_R_lidar = _opencv_matrix(
        calibration_text, "extrinsicRotation_imu_laser", 3, 3
    )
    imu_p_lidar = _opencv_matrix(
        calibration_text, "extrinsicTranslation_imu_laser", 3, 1
    ).reshape(3)
    imu_T_lidar = np.eye(4, dtype=np.float64)
    imu_T_lidar[:3, :3] = imu_R_lidar
    imu_T_lidar[:3, 3] = imu_p_lidar

    manifest = {
        "schema": CALIBRATION_CONTRACT_SCHEMA,
        "estimator_semantics_version": ESTIMATOR_SEMANTICS_VERSION,
        "estimator_config_sha256": _file_sha256(estimator_config),
        "estimator_calibration_sha256": _file_sha256(estimator_calibration),
        "kinematics_urdf_sha256": _file_sha256(kinematics_urdf),
    }
    canonical = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return CalibrationContract(
        digest=sha256(canonical).digest(),
        manifest=manifest,
        imu_T_lidar=imu_T_lidar,
    )


def load_installed_calibration_contract() -> CalibrationContract:
    from ament_index_python.packages import get_package_share_directory

    share = Path(get_package_share_directory("super_odometry"))
    return load_calibration_contract(
        estimator_config_path=share / "config" / "livox_mid360.yaml",
        estimator_calibration_path=(
            share / "config" / "livox" / "livox_mid360_calibration.yaml"
        ),
    )
