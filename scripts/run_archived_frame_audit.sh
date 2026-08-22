#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 LEGACY_RUN_DIR OUTPUT_RUN_DIR" >&2
  exit 2
fi

legacy_run_dir=$(realpath "$1")
output_run_dir=$(realpath "$2")
source_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
image=${SUPERODOM_AUDIT_IMAGE:-tml/superodom-humble:98f2bcb-policy-state}
trials=(
  passive-controlled-01
  passive-controlled-02
  translation-only-04
)

mkdir -p "$output_run_dir/data" "$output_run_dir/logs"

for trial in "${trials[@]}"; do
  tracks_output="$output_run_dir/data/$trial-odometry-tracks.jsonl"
  metrics_output="$output_run_dir/data/$trial-frame-audit.json"
  samples_output="$output_run_dir/data/$trial-aligned-samples.csv"
  docker run --rm \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -v "$source_root:/workspace/SuperOdom:ro" \
    -v "$legacy_run_dir:/legacy:ro" \
    -v "$output_run_dir:/out:rw" \
    "$image" \
    bash -lc "
      source /opt/ros/humble/setup.bash
      source /opt/superodom_ws/install/setup.bash
      if [[ ! -e /out/data/$trial-odometry-tracks.jsonl ]]; then
        python3 /workspace/SuperOdom/scripts/export_odometry_tracks.py \
          --bag /legacy/data/live/$trial/data/root_state \
          --output /out/data/$trial-odometry-tracks.jsonl
      fi
      if [[ ! -e /out/data/$trial-frame-audit.json ]]; then
        python3 /workspace/SuperOdom/scripts/score_odometry_tracks.py \
          --motive /legacy/data/$trial-motive-raw.jsonl \
          --tracks /out/data/$trial-odometry-tracks.jsonl \
          --bridge-status /legacy/data/live/$trial/data/bridge_status.jsonl \
          --metrics /out/data/$trial-frame-audit.json \
          --samples /out/data/$trial-aligned-samples.csv
      fi
    " >"$output_run_dir/logs/$trial.log" 2>&1
done

python3 - "$output_run_dir" <<'PY'
import csv
import json
from pathlib import Path
import sys

run_dir = Path(sys.argv[1])
rows = []
for path in sorted((run_dir / "data").glob("*-frame-audit.json")):
    metrics = json.loads(path.read_text(encoding="utf-8"))
    trial = path.name.removesuffix("-frame-audit.json")
    sensor = metrics["topics"]["/state_estimation"]
    pelvis = metrics["topics"]["/pelvis_state_estimation"]
    rows.append(
        {
            "trial": trial,
            "overlap_duration_s": metrics["overlap_duration_s"],
            "source_epoch_count": metrics["bridge_events_in_overlap"]["source_epoch_count"],
            "sensor_position_rmse_m": sensor["planar_position_error_m"]["rmse"],
            "pelvis_position_rmse_m": pelvis["planar_position_error_m"]["rmse"],
            "pelvis_position_improvement_percent": metrics["pelvis_improvement_percent"]["planar_position_rmse"],
            "sensor_rpe_1s_rmse_m": sensor["relative_position_error_1s_m"]["rmse"],
            "pelvis_rpe_1s_rmse_m": pelvis["relative_position_error_1s_m"]["rmse"],
            "pelvis_rpe_improvement_percent": metrics["pelvis_improvement_percent"]["relative_position_1s_rmse"],
            "sensor_yaw_rmse_deg": sensor["relative_yaw_error_deg"]["rmse"],
            "pelvis_yaw_rmse_deg": pelvis["relative_yaw_error_deg"]["rmse"],
            "pelvis_yaw_improvement_percent": metrics["pelvis_improvement_percent"]["relative_yaw_rmse"],
            "joint_or_estimate_failure_count": metrics["bridge_events_in_overlap"]["joint_or_estimate_failure_count"],
            "sensor_to_pelvis_offset_change_p95_m": metrics["sensor_to_pelvis_offset_change_from_median_m"]["p95"],
        }
    )

metrics_path = run_dir / "metrics.csv"
temporary_path = run_dir / "metrics.csv.tmp"
with temporary_path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
temporary_path.replace(metrics_path)
print(json.dumps(rows, indent=2, sort_keys=True))
PY
