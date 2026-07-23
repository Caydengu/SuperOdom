from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np

from g1_root_state_bridge.calibration_contract import load_calibration_contract
from g1_root_state_bridge.kinematics import KINEMATICS_URDF_PATH


ROOT = Path(__file__).resolve().parents[2]


def test_contract_hashes_exact_files_and_extracts_imu_to_lidar() -> None:
    config = ROOT / "super_odometry/config/livox_mid360.yaml"
    calibration = ROOT / "super_odometry/config/livox/livox_mid360_calibration.yaml"

    first = load_calibration_contract(
        estimator_config_path=config,
        estimator_calibration_path=calibration,
    )
    second = load_calibration_contract(
        estimator_config_path=config,
        estimator_calibration_path=calibration,
    )

    assert first.digest == second.digest
    assert len(first.digest) == 32
    assert first.manifest["estimator_config_sha256"] == sha256(config.read_bytes()).hexdigest()
    assert first.manifest["estimator_calibration_sha256"] == sha256(
        calibration.read_bytes()
    ).hexdigest()
    assert first.manifest["kinematics_urdf_sha256"] == sha256(
        Path(KINEMATICS_URDF_PATH).read_bytes()
    ).hexdigest()
    assert np.allclose(first.imu_T_lidar[:3, :3], np.eye(3))
    assert np.allclose(first.imu_T_lidar[:3, 3], (-0.011, -0.02329, 0.04412))
