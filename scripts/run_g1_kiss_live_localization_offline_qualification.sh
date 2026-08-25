#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 --run-dir PATH --all" >&2
}

run_dir=""
run_all=false
while (( $# )); do
  case "$1" in
    --run-dir) run_dir=$2; shift 2 ;;
    --all) run_all=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
done
[[ -n "$run_dir" && "$run_all" == true ]] || { usage; exit 2; }
[[ -d "$run_dir" && -f "$run_dir/manifest.json" ]] || {
  echo "run directory must be initialized with an artifact manifest" >&2
  exit 2
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "$script_dir/.." && pwd -P)"
workspace_root=/move/u/caydengu/cayden
research_root="$workspace_root/research/perceptive-humanoid-diffusion"
reference_run="$research_root/runs/2026-08-23_g1-4123-localization-stack-search"
scorer_checkout="$workspace_root/.worktrees/superodom-g1-4123-localization-stack-search"
python="$reference_run/env/kiss-icp/bin/python"
qualification_dir="$run_dir/data/qualification"
[[ ! -e "$qualification_dir" ]] || {
  echo "refusing to overwrite qualification output: $qualification_dir" >&2
  exit 2
}
mkdir -p "$qualification_dir" "$run_dir/logs"

export PYTHONPATH="$repo_root/g1_root_state_bridge"
set -o pipefail
localization_image="${G1_LOCALIZATION_IMAGE:-tml/g1-kiss-localization:1.5.1-ui-init-humble}"
docker run --rm --entrypoint /bin/bash \
  --volume "$repo_root:/repo:ro" \
  "$localization_image" -lc \
  'source /opt/ros/humble/setup.bash; export PYTHONPATH=/repo/g1_root_state_bridge; export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1; cd /repo; python3 -m pytest -q tests/root_state -k "not g1_dynamic_capture_recorder and not g1_dynamic_capture_relay and not udp_receiver"' \
  2>&1 | tee "$run_dir/logs/qualification-tests.log"
"$python" -m compileall -q "$repo_root/g1_root_state_bridge/g1_root_state_bridge"
"$python" -m ruff check "$repo_root/g1_root_state_bridge" "$repo_root/tests/root_state" \
  2>&1 | tee "$run_dir/logs/qualification-ruff.log"
"$python" -m g1_root_state_bridge.fault_matrix_cli \
  --output "$qualification_dir/fault-matrix.json" \
  2>&1 | tee "$run_dir/logs/qualification-fault-matrix.log"

candidate_treatment=kiss_live_selected_pelvis_navigation
reference_treatment=kiss_icp_exact_gyro_deskew_fk_position_livox_gyro_torso_yaw
front_plane_to_pelvis_x_m=-0.061431244015693665
windows="$research_root/runs/2026-08-23_g1-4123-night-localization-data-readiness/data/evaluation_windows.json"

run_one() {
  local short_name=$1
  local capture_name=$2
  local replay_dir="$qualification_dir/replay/$short_name"
  local scans="$reference_run/data/scans/${short_name}-livox-scans-with-time-imu.npz"
  local capture="$research_root/runs/$capture_name"
  local reference_treatments="$reference_run/data/treatments/kiss-icp-${short_name}-v015-exact-gyro-deskew-fk-livox-yaw.jsonl"

  "$python" -m g1_root_state_bridge.clock_replay_cli \
    --lowstate "$capture/lowstate/packets.bin" \
    --output "$qualification_dir/${short_name}-clock-replay.json" \
    2>&1 | tee "$run_dir/logs/qualification-${short_name}-clock-replay.log"

  mkdir -p "$replay_dir"
  docker run --rm --entrypoint /bin/bash \
    --user "$(id -u):$(id -g)" \
    --network none \
    --cap-drop ALL \
    --security-opt no-new-privileges \
    --read-only \
    --tmpfs /tmp:rw,nosuid,size=512m \
    --env HOME=/tmp/home \
    --volume "$repo_root:/repo:ro" \
    --volume "$scans:/inputs/scans.npz:ro" \
    --volume "$capture/lowstate/packets.bin:/inputs/packets.bin:ro" \
    --volume "$replay_dir:/output:rw" \
    "$localization_image" -lc \
    'source /opt/ros/humble/setup.bash; export PYTHONPATH=/repo/g1_root_state_bridge; python3 -m g1_root_state_bridge.replay_cli --scans /inputs/scans.npz --lowstate /inputs/packets.bin --output-dir /output --gyro-bias-radps 0.025702817208593076 -0.02178237836035201 -0.01574406003550275 --treatment-name kiss_live_selected_pelvis_navigation' \
    2>&1 | tee "$run_dir/logs/qualification-${short_name}-replay.log"

  "$python" -m g1_root_state_bridge.compare_replay_cli \
    --candidate "$replay_dir/treatments.jsonl" \
    --candidate-treatment "$candidate_treatment" \
    --reference "$reference_treatments" \
    --reference-treatment "$reference_treatment" \
    --pose-comparison initial-relative \
    --output "$qualification_dir/${short_name}-reference-comparison.json" \
    2>&1 | tee "$run_dir/logs/qualification-${short_name}-comparison.log"

  PYTHONPATH="$scorer_checkout/g1_root_state_bridge" "$python" \
    "$scorer_checkout/scripts/score_segmented_amo_localization.py" \
    --treatments "$replay_dir/treatments.jsonl" \
    --motive "$capture/motive/frames.jsonl" \
    --lowstate "$capture/lowstate/packets.bin" \
    --windows "$windows" \
    --run-name "$capture_name" \
    --front-plane-to-pelvis-x-m "$front_plane_to_pelvis_x_m" \
    --reference-label motive_front_pelvis_evaluator_only \
    --output "$qualification_dir/${short_name}-motive-score.json" \
    2>&1 | tee "$run_dir/logs/qualification-${short_name}-motive-score.log"
}

run_one walk02 2026-08-22_g1-4123-motive-amo-walk-02-stress
run_one walk03 2026-08-22_g1-4123-motive-amo-walk-03-stress-yaw

"$python" - "$qualification_dir" "$reference_run" <<'PY'
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
reference_run = Path(sys.argv[2])
candidate_name = "kiss_live_selected_pelvis_navigation"
reference_name = "kiss_icp_exact_gyro_deskew_fk_position_livox_gyro_torso_yaw"
rows = []
passed = True
for short_name in ("walk02", "walk03"):
    replay = json.loads((root / "replay" / short_name / "metrics.json").read_text())
    clock = json.loads((root / f"{short_name}-clock-replay.json").read_text())
    comparison = json.loads((root / f"{short_name}-reference-comparison.json").read_text())
    candidate_score = json.loads((root / f"{short_name}-motive-score.json").read_text())
    reference_score = json.loads(
        (
            reference_run
            / "data/scores"
            / f"kiss-icp-{short_name}-v015-exact-gyro-deskew-fk-livox-yaw-mesh-front.json"
        ).read_text()
    )
    candidate = candidate_score["aggregates"][candidate_name]["primary"]
    reference = reference_score["aggregates"][reference_name]["primary"]
    metrics = {
        "planar_rmse_m": candidate["planar_rmse_m"]["mean"],
        "rpe_1s_translation_rmse_m": candidate["rpe_1s_translation_rmse_m"]["mean"],
        "yaw_rmse_deg": candidate["yaw_rmse_deg"]["mean"],
        "availability_fraction": candidate["availability_fraction"]["mean"],
        "planar_delta_m": candidate["planar_rmse_m"]["mean"] - reference["planar_rmse_m"]["mean"],
        "yaw_delta_deg": candidate["yaw_rmse_deg"]["mean"] - reference["yaw_rmse_deg"]["mean"],
        "availability_delta": candidate["availability_fraction"]["mean"] - reference["availability_fraction"]["mean"],
        "runtime_p95_ms": replay["stage_runtime_ms"]["total"]["p95"],
        "runtime_p99_ms": replay["stage_runtime_ms"]["total"]["p99"],
        "pose_age_p95_ms": replay["pose_age_ms"]["p95"],
        "realtime_factor": replay["realtime_factor"],
        "input_overlap_availability": replay["input_overlap_availability"],
        "reference_pose_rmse_m": comparison["planar_rmse_m"],
        "reference_yaw_rmse_deg": comparison["yaw_rmse_deg"],
        "reference_coverage": comparison["reference_coverage"],
        "source_joint_sequence_match_fraction": comparison["source_joint_sequence_match_fraction"],
        "clock_fit_valid_fraction": clock["fit_valid_fraction"],
        "clock_sample_valid_fraction": clock["sample_valid_fraction"],
        "clock_online_to_batch_p95_ms": clock["online_to_batch_abs_delta_ms"]["p95"],
        "clock_observe_p95_us": clock["observe_runtime_us"]["p95"],
    }
    run_pass = (
        comparison["status"] == "pass"
        and clock["status"] == "pass"
        and metrics["planar_delta_m"] <= 0.01
        and metrics["yaw_delta_deg"] <= 0.5
        and metrics["availability_delta"] >= -0.005
        and metrics["runtime_p95_ms"] <= 20.0
        and metrics["runtime_p99_ms"] <= 30.0
        and metrics["pose_age_p95_ms"] <= 50.0
        and metrics["realtime_factor"] >= 2.0
    )
    passed &= run_pass
    rows.append({"run": short_name, "status": "pass" if run_pass else "fail", **metrics})

faults = json.loads((root / "fault-matrix.json").read_text())
passed &= faults["status"] == "pass" and faults["false_healthy_count"] == 0
report = {
    "schema": "g1_kiss_live_localization_offline_qualification_v1",
    "status": "pass" if passed else "fail",
    "motive_role": "evaluator_only",
    "rows": rows,
    "fault_matrix_status": faults["status"],
    "false_healthy_count": faults["false_healthy_count"],
    "live_robot_evidence": False,
}
(root / "qualification-summary.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n"
)
with (root / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream)
    writer.writerow(("run", "metric", "value"))
    for row in rows:
        for metric, value in row.items():
            if metric not in {"run", "status"}:
                writer.writerow((row["run"], metric, value))
print(json.dumps(report, sort_keys=True))
if not passed:
    raise SystemExit(1)
PY
