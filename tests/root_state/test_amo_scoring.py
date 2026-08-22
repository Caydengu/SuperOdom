from __future__ import annotations

import numpy as np
from g1_root_state_bridge.amo_dataset import AffineClockMapping, MotiveCapture
from g1_root_state_bridge.amo_scoring import score_track


def test_identical_planar_motion_scores_zero_after_coordinate_adaptation() -> None:
    count = 201
    event = np.arange(count, dtype=np.int64) * 10_000_000 + 1_000_000_000
    distance = np.linspace(0.0, 2.0, count)
    motive_position = np.column_stack((distance, np.full(count, 0.8), np.zeros(count)))
    quaternion = np.tile(np.asarray([0.0, 0.0, 0.0, 1.0]), (count, 1))
    clock = AffineClockMapping(0, 0, 1.0, 0.0, 0.0, 0.0, 0.0)
    motive = MotiveCapture(
        software_stamp_ns=event,
        oslo_event_ns=event,
        receipt_realtime_ns=event,
        frame_number=np.arange(count),
        position_xyz_m=motive_position,
        quaternion_xyzw=quaternion,
        mean_marker_error_m=np.zeros(count),
        clock=clock,
        total_frames=count,
        rejected_tracking_frames=0,
        rejected_marker_error_frames=0,
    )
    track = {
        "event_realtime_ns": event,
        "position_xyz_m": np.column_stack(
            (distance, np.zeros(count), np.full(count, 0.8))
        ),
        "quaternion_xyzw": quaternion,
        "source_lowstate_age_ns": np.zeros(count, dtype=np.int64),
        "fusion_healthy": np.ones(count, dtype=bool),
    }
    report = score_track(
        track,
        motive,
        scoring_start_realtime_ns=int(event[0]),
        front_plane_to_pelvis_x_m=0.0,
        estimator_to_reference_rotation=np.eye(2),
    )
    assert report["planar_error_m"]["maximum"] < 1e-12
    assert report["yaw_error_deg"]["maximum"] < 1e-12
