from __future__ import annotations

import subprocess
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RECORDER_SOURCE = (
    REPOSITORY_ROOT / "tools" / "optitrack_reference_recorder.cpp"
)
BUILD_SCRIPT = REPOSITORY_ROOT / "scripts" / "build_optitrack_reference_recorder.sh"


def test_recorder_preserves_raw_capture_and_quality_fields() -> None:
    source = RECORDER_SOURCE.read_text(encoding="utf-8")

    for required_text in (
        "g1_optitrack_raw_v1",
        "motive_software_time_s",
        "camera_mid_exposure_ticks",
        "camera_data_received_ticks",
        "transmit_ticks",
        "precision_timestamp_seconds",
        "precision_timestamp_fractional_seconds",
        "receipt_realtime_ns",
        "receipt_monotonic_ns",
        "capture_realtime_estimate_ns",
        "seconds_since_host_mid_exposure",
        "high_resolution_clock_frequency_hz",
        "tracking_valid",
        "mean_marker_error_m",
        "position_motive_xyz_m",
        "quaternion_motive_xyzw",
    ):
        assert required_text in source

    for forbidden_transform in (
        "RECENTER_OFFSET",
        "tare",
        "y_up_to_z_up",
        "head_pos",
        "head_quat",
    ):
        assert forbidden_transform not in source


def test_recorder_binds_identity_to_motive_description_fail_closed() -> None:
    source = RECORDER_SOURCE.read_text(encoding="utf-8")

    for required_text in (
        "GetDataDescriptionList",
        "NatNet_FreeDescriptions",
        "Descriptor_RigidBody",
        "rigid_body_identity_validated",
        "server_rigid_body_name",
        "server_rigid_body_marker_count",
        "available direct rigid bodies",
        "does not match Motive description",
    ):
        assert required_text in source

    assert "description->ID == options.rigid_body_id" in source
    assert "server_name != options.rigid_body_name" in source
    assert "context.tracking_valid_count.load() > 0" in source


def test_build_script_is_syntax_valid_and_requires_pinned_sdk_root() -> None:
    script = BUILD_SCRIPT.read_text(encoding="utf-8")

    subprocess.run(["bash", "-n", str(BUILD_SCRIPT)], check=True)
    assert "NATNET_SDK_ROOT" in script
    assert "NatNet_SDK_4.4" in script
    assert "libNatNet.so" in script
    assert "optitrack_reference_recorder.cpp" in script
