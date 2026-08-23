from __future__ import annotations

import numpy as np

from g1_root_state_bridge.amo_dataset import AffineClockMapping, MotiveCapture
from scripts.score_segmented_amo_localization import score_segment


def test_segment_score_is_rate_agnostic_and_origin_resets() -> None:
    event = 1_000_000_000 + np.arange(151, dtype=np.int64) * 100_000_000
    distance = np.linspace(0.0, 3.0, event.size)
    quaternion = np.tile(np.asarray([0.0, 0.0, 0.0, 1.0]), (event.size, 1))
    clock = AffineClockMapping(0, 0, 1.0, 0.0, 0.0, 0.0, 0.0)
    motive = MotiveCapture(
        software_stamp_ns=event,
        oslo_event_ns=event,
        receipt_realtime_ns=event,
        frame_number=np.arange(event.size),
        position_xyz_m=np.column_stack((distance, np.full(event.size, 0.8), np.zeros(event.size))),
        quaternion_xyzw=quaternion,
        mean_marker_error_m=np.zeros(event.size),
        clock=clock,
        total_frames=event.size,
        rejected_tracking_frames=0,
        rejected_marker_error_frames=0,
    )
    track = {
        "event_realtime_ns": event,
        "position_xyz_m": np.column_stack((distance + 10.0, np.zeros(event.size), np.full(event.size, 0.8))),
        "quaternion_xyzw": quaternion,
        "source_lowstate_age_ns": np.zeros(event.size, dtype=np.int64),
        "fusion_healthy": np.ones(event.size, dtype=bool),
    }
    report = score_segment(
        track,
        motive,
        start_realtime_ns=int(event[10]),
        end_realtime_ns=int(event[110]),
        front_plane_to_pelvis_x_m=0.0,
        estimator_to_reference_rotation=np.eye(2),
    )
    assert report["nominal_rate_hz"] == 10.0
    assert report["availability_fraction"] == 1.0
    assert report["planar_error_m"]["maximum"] < 1e-12
    assert report["yaw_error_deg"]["maximum"] < 1e-12
    assert report["rpe"]["5.0"]["translation_error_m"]["maximum"] < 1e-12
