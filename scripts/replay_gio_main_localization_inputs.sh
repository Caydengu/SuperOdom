#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: replay_gio_main_localization_inputs.sh --input-bag BAG --output-dir DIR --name NAME [--domain ID]

Replays the exact pinned robot-vlm main SuperOdometry producers and exports the
aligned odometry plus every aligned registered scan needed by Gio's continuous
mesh-refine localizer. The workflow is offline and command-incapable.
EOF
}

input_bag=
output_dir=
name=
domain=171
while (( $# )); do
  case "$1" in
    --input-bag) input_bag=$2; shift 2 ;;
    --output-dir) output_dir=$2; shift 2 ;;
    --name) name=$2; shift 2 ;;
    --domain) domain=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ -d "$input_bag" && -n "$output_dir" && "$name" =~ ^[A-Za-z0-9_.-]+$ ]] || {
  usage >&2
  exit 2
}
[[ "$domain" =~ ^[0-9]+$ ]] || { echo "invalid ROS domain" >&2; exit 2; }

workspace=/move/u/caydengu/cayden
main="$workspace/.worktrees/robot-vlm-main-localization-baseline"
search="$workspace/.worktrees/superodom-g1-4123-localization-stack-search"
tool="$workspace/.worktrees/superodom-g1-4123-final-live-localization"
image=tml/superodom-humble:57a6e23-minimal
mkdir -p "$output_dir"
for path in "$output_dir/${name}-laser-odometry-aligned.jsonl" \
  "$output_dir/${name}-registered-scan-aligned.npz"; do
  [[ ! -e "$path" ]] || { echo "refusing to overwrite $path" >&2; exit 2; }
done

scratch="$(mktemp -d /var/tmp/g1-4123-gio-main-inputs.XXXXXX)"
cleanup() {
  case "$scratch" in
    /var/tmp/g1-4123-gio-main-inputs.*) rm -rf -- "$scratch" ;;
    *) echo "refusing to remove unexpected scratch: $scratch" >&2 ;;
  esac
}
trap cleanup EXIT

docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$search:/workspace:ro" -v "$input_bag:/input:ro" -v "$scratch:/scratch:rw" \
  "$image" bash -lc \
  "source /opt/ros/humble/setup.bash && python3 /workspace/scripts/normalize_unitree_pointcloud2_bag.py \
    --input-bag /input --output-bag /scratch/normalized \
    --time-scale 1e-9 --align-header-to-bag-clock"

docker run --rm --network host --cap-drop ALL --security-opt no-new-privileges \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp -e ROS_DOMAIN_ID="$domain" \
  -v "$scratch:/scratch:rw" -v "$main:/main:ro" "$image" bash -lc '
    set -eo pipefail
    source /opt/ros/humble/setup.bash
    source /opt/superodom_ws/install/setup.bash
    set -u
    pids=()
    stop_all() {
      local pid
      for pid in "${pids[@]}"; do kill -INT "$pid" 2>/dev/null || true; done
      sleep 1
      for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
      for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
    }
    trap stop_all EXIT INT TERM
    cfg=/main/deploy/g1/superodom/g1_mid360.yaml
    norm=/main/deploy/g1/superodom/g1_frame_normalizer.yaml
    calib=/opt/superodom_ws/install/share/super_odometry/config/livox/livox_mid360_calibration.yaml
    for node in feature_extraction_node laser_mapping_node imu_preintegration_node; do
      ros2 run super_odometry "$node" --ros-args --params-file "$cfg" \
        -p calibration_file:="$calib" -p use_sim_time:=true &
      pids+=("$!")
    done
    ros2 run super_odometry frame_normalizer_node --ros-args \
      --params-file "$norm" -p use_sim_time:=true &
    pids+=("$!")
    ready=false
    for _ in $(seq 1 80); do
      nodes="$(ros2 node list 2>/dev/null || true)"
      if grep -Fq /feature_extraction_node <<<"$nodes" && \
         grep -Fq /laser_mapping_node <<<"$nodes" && \
         grep -Fq /imu_preintegration_node <<<"$nodes" && \
         grep -Fq /frame_normalizer <<<"$nodes"; then ready=true; break; fi
      sleep 0.25
    done
    [[ "$ready" == true ]] || { echo "main-stack nodes did not become ready" >&2; exit 1; }
    ros2 bag record --use-sim-time -o /scratch/main-stack \
      /laser_odometry_aligned /registered_scan_aligned /laser_cloud_map_aligned &
    recorder=$!; pids+=("$recorder")
    sleep 1
    ros2 bag play /scratch/normalized --clock --rate 1.0 \
      --remap /utlidar/cloud_livox_mid360:=/utlidar/cloud_livox_mid360_sec
    sleep 2
    kill -INT "$recorder" 2>/dev/null || true
    wait "$recorder" 2>/dev/null || true
    pids=("${pids[@]:0:4}")
  '

docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$scratch:/scratch:ro" -v "$tool:/tool:ro" -v "$output_dir:/output:rw" \
  "$image" bash -lc "
    set -eo pipefail
    source /opt/ros/humble/setup.bash
    export PYTHONPATH=/tool/g1_root_state_bridge:\${PYTHONPATH:-}
    python3 /tool/scripts/export_odometry_tracks.py \
      --bag /scratch/main-stack --topic /laser_odometry_aligned \
      --output /output/${name}-laser-odometry-aligned.jsonl
    python3 /tool/scripts/export_registered_map_evidence.py \
      --bag /scratch/main-stack --topic /registered_scan_aligned \
      --expected-frame map --maximum-points-per-cloud 5000 \
      --output /output/${name}-registered-scan-aligned.npz
  "

printf 'exported exact Gio-main inputs to %s\n' "$output_dir"
