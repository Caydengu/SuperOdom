#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: root_state_shadow.sh --output-dir PATH --trial-name NAME --network-interface IFACE --image IMAGE [--ros-domain-id 0..232] [--duration-sec N] [--dry-run]

Run a bounded, non-actuating SuperOdometry-to-pelvis shadow. The external G1
typed joint relay must already be sending UDP port 5576. SuperOdometry remains
the sole /livox/lidar subscriber; the recorder captures only compact inputs,
diagnostics, and pelvis outputs. The selected image must contain an installed
zero-prediction reference configuration and g1_root_state_bridge package.
EOF
}

output_dir=""
trial_name=""
network_interface=""
image=""
ros_domain_id=42
duration_sec=60
dry_run=false

while (( $# )); do
  case "$1" in
    --output-dir)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      output_dir=$2
      shift 2
      ;;
    --trial-name)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      trial_name=$2
      shift 2
      ;;
    --network-interface)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      network_interface=$2
      shift 2
      ;;
    --image)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      image=$2
      shift 2
      ;;
    --ros-domain-id)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      ros_domain_id=$2
      shift 2
      ;;
    --duration-sec)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      duration_sec=$2
      shift 2
      ;;
    --dry-run)
      dry_run=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

[[ -n "$output_dir" ]] || { echo "--output-dir is required" >&2; exit 2; }
[[ -n "$trial_name" ]] || { echo "--trial-name is required" >&2; exit 2; }
[[ -n "$network_interface" ]] || { echo "--network-interface is required" >&2; exit 2; }
[[ -n "$image" ]] || { echo "--image is required" >&2; exit 2; }
[[ "$trial_name" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || {
  echo "Trial name must use letters, digits, underscores, periods, or dashes" >&2
  exit 2
}
[[ "$duration_sec" =~ ^[0-9]+$ ]] && (( 10#$duration_sec >= 10 && 10#$duration_sec <= 900 )) || {
  echo "Duration must be an integer from 10 through 900 seconds" >&2
  exit 2
}

docker image inspect "$image" >/dev/null
output_dir="$(mkdir -p "$output_dir" && cd "$output_dir" && pwd -P)"
trial_dir="$output_dir/data/live/$trial_name"
[[ ! -e "$trial_dir" ]] || {
  echo "Refusing to overwrite existing root-state trial: $trial_dir" >&2
  exit 2
}
mkdir -p "$trial_dir/logs" "$trial_dir/data" "$trial_dir/runtime-input"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
run_wrapper="$script_dir/run.sh"
container_name="superodom-root-state-shadow-$trial_name"
wrapper_args=(
  --data-dir "$trial_dir/runtime-input"
  --output-dir "$trial_dir"
  --ros-domain-id "$ros_domain_id"
  --network-interface "$network_interface"
  --container-name "$container_name"
)
if [[ "$dry_run" == true ]]; then
  wrapper_args+=(--dry-run)
fi

container_script='set -euo pipefail
duration_sec=$1
launch_pid=""
bridge_pid=""
recorder_pid=""

stop_process() {
  local pid=$1
  local label=$2
  [[ -n "$pid" ]] || return 0
  kill -0 "$pid" 2>/dev/null || { wait "$pid" 2>/dev/null || true; return 0; }
  kill -INT "$pid" 2>/dev/null || true
  for _ in $(seq 1 75); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.2
  done
  if kill -0 "$pid" 2>/dev/null; then
    echo "$label did not stop after SIGINT; sending SIGTERM" >&2
    kill -TERM "$pid" 2>/dev/null || true
  fi
  wait "$pid" 2>/dev/null || true
}

cleanup() {
  stop_process "$bridge_pid" root-state-bridge
  stop_process "$recorder_pid" recorder
  stop_process "$launch_pid" superodometry-launch
}
trap cleanup EXIT INT TERM

wait_for_type() {
  local topic=$1
  local expected=$2
  local timeout_sec=$3
  local actual=""
  local deadline=$((SECONDS + timeout_sec))
  while (( SECONDS < deadline )); do
    actual="$(timeout 2s ros2 topic type "$topic" 2>/dev/null || true)"
    if [[ "$actual" == "$expected" ]]; then
      return 0
    fi
    if [[ -n "$actual" && "$actual" != "$expected" ]]; then
      echo "Unexpected type for $topic: $actual (expected $expected)" >&2
      return 31
    fi
    sleep 0.5
  done
  echo "Timed out waiting for $topic with type $expected" >&2
  return 32
}

subscription_count() {
  local topic=$1
  ros2 topic info "$topic" --verbose \
    | sed -n "s/^Subscription count: \\([0-9][0-9]*\\)$/\\1/p" \
    | head -n 1
}

wait_for_type /livox/lidar livox_ros_driver2/msg/CustomMsg 45
wait_for_type /livox/imu sensor_msgs/msg/Imu 45
ros2 topic info /livox/lidar --verbose > /output/logs/lidar_info_before.txt
pre_subscribers="$(subscription_count /livox/lidar)"
[[ "$pre_subscribers" == "0" ]] || {
  echo "Expected zero pre-existing /livox/lidar subscribers, found $pre_subscribers" >&2
  exit 33
}

live_config=/opt/superodom_ws/install/share/super_odometry/config/livox_mid360.yaml
[[ -f "$live_config" ]] || { echo "Missing installed live config" >&2; exit 34; }
grep -q "predict_future_secs: 0.0" "$live_config" || {
  echo "Installed live config does not disable forward prediction" >&2
  exit 35
}
grep -q "min_range: 0.5" "$live_config" || {
  echo "Installed live config does not preserve the 0.5 m terrain range" >&2
  exit 36
}
sha256sum "$live_config" > /output/logs/live_config_sha256.txt

record_topics=(
  /livox/imu
  /state_estimation
  /laser_odometry
  /super_odometry_stats
  /state_estimation_health
  /state_estimation_calibration
  /prediction_source
  /pelvis_state_estimation
  /pelvis_state_bridge/status
)
ros2 bag record --include-unpublished-topics \
  -o /output/data/root_state \
  "${record_topics[@]}" \
  > /output/logs/record.log 2>&1 &
recorder_pid=$!
sleep 1
kill -0 "$recorder_pid" 2>/dev/null || { wait "$recorder_pid"; exit 37; }

date +%s%N > /output/logs/launch_wall_ns.txt
ros2 launch super_odometry livox_humanoid.launch.py \
  use_sim_time:=false config_file:="$live_config" \
  > /output/logs/superodometry.log 2>&1 &
launch_pid=$!

ros2 run g1_root_state_bridge g1-root-state-bridge --ros-args \
  -p replay_jsonl_path:=/output/data/bridge_status.jsonl \
  -p root_state_bind_endpoint:=tcp://*:5575 \
  -p joint_bind_host:=0.0.0.0 \
  -p joint_bind_port:=5576 \
  > /output/logs/root_state_bridge.log 2>&1 &
bridge_pid=$!

wait_for_type /state_estimation nav_msgs/msg/Odometry 45
wait_for_type /pelvis_state_estimation nav_msgs/msg/Odometry 60
timeout 60s ros2 topic echo --once /pelvis_state_estimation \
  > /output/logs/first_pelvis_state.txt
date +%s%N > /output/logs/first_pelvis_receipt_wall_ns.txt

ros2 topic info /livox/lidar --verbose > /output/logs/lidar_info_active.txt
active_subscribers="$(subscription_count /livox/lidar)"
[[ "$active_subscribers" == "1" ]] || {
  echo "Expected one active /livox/lidar subscriber, found $active_subscribers" >&2
  exit 38
}
ros2 topic info /pelvis_state_estimation --verbose > /output/logs/pelvis_info_active.txt
ros2 topic list -t > /output/logs/topic_list_active.txt
ros2 node list > /output/logs/node_list_active.txt

sleep "$duration_sec"
stop_process "$bridge_pid" root-state-bridge
bridge_pid=""
stop_process "$recorder_pid" recorder
recorder_pid=""
stop_process "$launch_pid" superodometry-launch
launch_pid=""

[[ -f /output/data/root_state/metadata.yaml ]] || {
  echo "Root-state shadow did not produce bag metadata" >&2
  exit 39
}
[[ -s /output/data/bridge_status.jsonl ]] || {
  echo "Root-state bridge did not produce a status trace" >&2
  exit 40
}
grep -q packet_published /output/data/bridge_status.jsonl || {
  echo "Root-state bridge published no strict packet" >&2
  exit 41
}
ros2 bag info /output/data/root_state > /output/logs/bag_info.txt
'

SUPERODOM_IMAGE="$image" exec "$run_wrapper" "${wrapper_args[@]}" -- \
  bash -c "$container_script" root-state-shadow "$duration_sec"
