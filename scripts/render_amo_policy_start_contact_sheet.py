#!/usr/bin/env python3
"""Render RGB evidence around a proposed AMO walking boundary.

The MP4 concatenates successfully decoded frames, while ``timestamps.npz``
retains each frame's Oslo realtime timestamp.  Seeking by wall-clock video
seconds is therefore wrong across capture gaps; this script selects the nearest
timestamp first and then seeks to that encoded frame index.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--boundary-realtime-ns", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--offset-sec",
        type=float,
        action="append",
        default=None,
        help="Frame offset from the proposed boundary; repeat as needed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    offsets = args.offset_sec or [-8.0, -4.0, -1.0, 0.0, 1.0, 4.0, 8.0]
    video_path = args.run_dir / "vision" / "capture" / "recording.mp4"
    timestamp_path = args.run_dir / "vision" / "capture" / "timestamps.npz"
    with np.load(timestamp_path, allow_pickle=False) as archive:
        frame_times_s = np.asarray(archive["frame_times"], dtype=np.float64)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {video_path}")
    encoded_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if encoded_count != frame_times_s.size:
        raise RuntimeError(
            f"video/timestamp count mismatch: {encoded_count} != {frame_times_s.size}"
        )

    boundary_s = args.boundary_realtime_ns * 1e-9
    panels: list[tuple[float, float, np.ndarray]] = []
    for offset_s in offsets:
        target_s = boundary_s + offset_s
        index = int(np.argmin(np.abs(frame_times_s - target_s)))
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, bgr = capture.read()
        if not ok or bgr is None:
            raise RuntimeError(f"failed to decode frame {index} from {video_path}")
        panels.append(
            (offset_s, float(frame_times_s[index] - boundary_s), bgr[:, :, ::-1])
        )
    capture.release()

    figure, axes = plt.subplots(2, 4, figsize=(16, 8.6), constrained_layout=True)
    flat = list(axes.flat)
    for axis, (requested, actual, rgb) in zip(flat, panels, strict=False):
        axis.imshow(rgb)
        axis.set_title(f"requested {requested:+.1f} s\nactual {actual:+.2f} s")
        axis.axis("off")
    for axis in flat[len(panels) :]:
        axis.axis("off")
    figure.suptitle(
        f"Onboard RGB around proposed AMO start — {args.run_dir.name}",
        fontsize=16,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
