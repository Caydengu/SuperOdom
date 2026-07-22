#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: live_shadow.sh --output-dir PATH --network-interface IFACE [--ros-domain-id 0..232] [--duration-sec N] [--record-lidar] [--dry-run]

Launch the live, non-actuating SuperOdometry stack after the Livox input gate,
and record /livox/imu plus /state_estimation. Raw /livox/lidar recording is an
explicit stress option because it adds a second large-message subscriber. The
default profile uses real wall time and the production 0.5 m minimum range.
EOF
}

output_dir=""
ros_domain_id=42
duration_sec=90
network_interface=""
record_lidar=false
dry_run=false

while (( $# )); do
  case "$1" in
    --output-dir)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      output_dir=$2
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
    --network-interface)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      network_interface=$2
      shift 2
      ;;
    --record-lidar)
      record_lidar=true
      shift
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
[[ -n "$network_interface" ]] || { echo "--network-interface is required for live DDS" >&2; exit 2; }
[[ "$duration_sec" =~ ^[0-9]+$ ]] && (( 10#$duration_sec >= 10 && 10#$duration_sec <= 900 )) || {
  echo "Duration must be an integer from 10 through 900 seconds" >&2
  exit 2
}

output_dir="$(mkdir -p "$output_dir" && cd "$output_dir" && pwd -P)"
mkdir -p "$output_dir/logs" "$output_dir/data" "$output_dir/runtime-input"
if [[ "$record_lidar" == true ]]; then
  bag_name=live_superodometry_shadow_with_raw_lidar
  log_prefix=live_shadow_with_raw_lidar
else
  bag_name=live_superodometry_shadow_output_only
  log_prefix=live_shadow_output_only
fi
bag_path="$output_dir/data/$bag_name"
[[ ! -e "$bag_path" ]] || {
  echo "Refusing to overwrite existing live SuperOdometry shadow: $bag_path" >&2
  exit 2
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
run_wrapper="$script_dir/run.sh"
wrapper_args=(
  --data-dir "$output_dir/runtime-input"
  --output-dir "$output_dir"
  --ros-domain-id "$ros_domain_id"
  --network-interface "$network_interface"
  --container-name superodom-live-shadow
)
if [[ "$dry_run" == true ]]; then
  wrapper_args+=(--dry-run)
fi

container_script='set -euo pipefail
duration_sec=$1
record_lidar=$2
bag_name=$3
log_prefix=$4
launch_pid=""
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

wait_for_type /livox/lidar livox_ros_driver2/msg/CustomMsg 45
wait_for_type /livox/imu sensor_msgs/msg/Imu 45

live_config=/opt/superodom_ws/install/share/super_odometry/config/livox_mid360.yaml
[[ -f "$live_config" ]] || { echo "Missing installed live config: $live_config" >&2; exit 33; }
grep -n "min_range" "$live_config" > "/output/logs/${log_prefix}_config_contract.txt"
grep -q "min_range: 0.5" "$live_config" || {
  echo "Installed live configuration does not contain min_range 0.5" >&2
  exit 34
}
sha256sum "$live_config" > "/output/logs/${log_prefix}_config_sha256.txt"

ros2 topic list -t > "/output/logs/${log_prefix}_topic_list_before.txt"
ros2 node list > "/output/logs/${log_prefix}_node_list_before.txt"
ros2 topic info /livox/lidar --verbose > "/output/logs/${log_prefix}_lidar_info.txt"
ros2 topic info /livox/imu --verbose > "/output/logs/${log_prefix}_imu_info.txt"

record_topics=(
  /livox/imu
  /state_estimation
  /laser_odometry
  /super_odometry_stats
  /state_estimation_health
  /prediction_source
)
if [[ "$record_lidar" == true ]]; then
  record_topics+=(/livox/lidar)
fi
ros2 bag record --include-unpublished-topics \
  -o "/output/data/$bag_name" \
  "${record_topics[@]}" \
  > "/output/logs/${log_prefix}_record.log" 2>&1 &
recorder_pid=$!
sleep 1
kill -0 "$recorder_pid" 2>/dev/null || {
  echo "Recorder exited before SuperOdometry launch" >&2
  wait "$recorder_pid"
}

date +%s%N > "/output/logs/${log_prefix}_launch_wall_ns.txt"
ros2 launch super_odometry livox_humanoid.launch.py \
  use_sim_time:=false config_file:="$live_config" \
  > "/output/logs/${log_prefix}_launch.log" 2>&1 &
launch_pid=$!

wait_for_type /state_estimation nav_msgs/msg/Odometry 45
timeout 60s ros2 topic echo --once /state_estimation \
  > "/output/logs/${log_prefix}_first_state_estimation.txt"
date +%s%N > "/output/logs/${log_prefix}_first_output_receipt_wall_ns.txt"

ros2 topic info /state_estimation --verbose > "/output/logs/${log_prefix}_state_estimation_info.txt"
ros2 topic list -t > "/output/logs/${log_prefix}_topic_list_active.txt"
ros2 node list > "/output/logs/${log_prefix}_node_list_active.txt"

sleep "$duration_sec"
stop_process "$recorder_pid" recorder
recorder_pid=""
stop_process "$launch_pid" superodometry-launch
launch_pid=""

[[ -f "/output/data/$bag_name/metadata.yaml" ]] || {
  echo "Live shadow did not produce metadata.yaml" >&2
  exit 35
}
ros2 bag info "/output/data/$bag_name" > "/output/logs/${log_prefix}_bag_info.txt"
'

exec "$run_wrapper" "${wrapper_args[@]}" -- \
  bash -c "$container_script" live-shadow "$duration_sec" "$record_lidar" "$bag_name" "$log_prefix"
