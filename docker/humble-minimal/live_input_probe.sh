#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: live_input_probe.sh --output-dir PATH --network-interface IFACE [--ros-domain-id 0..232] [--duration-sec N] [--dry-run]

Record only the live /livox/lidar and /livox/imu boundary from the stationary
G1. This probe does not launch SuperOdometry or any robot-control process.
EOF
}

output_dir=""
ros_domain_id=42
duration_sec=30
network_interface=""
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
[[ "$duration_sec" =~ ^[0-9]+$ ]] && (( 10#$duration_sec >= 5 && 10#$duration_sec <= 600 )) || {
  echo "Duration must be an integer from 5 through 600 seconds" >&2
  exit 2
}

output_dir="$(mkdir -p "$output_dir" && cd "$output_dir" && pwd -P)"
mkdir -p "$output_dir/logs" "$output_dir/data" "$output_dir/runtime-input"
bag_path="$output_dir/data/live_input_probe"
[[ ! -e "$bag_path" ]] || {
  echo "Refusing to overwrite existing live input probe: $bag_path" >&2
  exit 2
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
run_wrapper="$script_dir/run.sh"
wrapper_args=(
  --data-dir "$output_dir/runtime-input"
  --output-dir "$output_dir"
  --ros-domain-id "$ros_domain_id"
  --network-interface "$network_interface"
  --container-name superodom-live-input
)
if [[ "$dry_run" == true ]]; then
  wrapper_args+=(--dry-run)
fi

container_script='set -euo pipefail
duration_sec=$1

wait_for_type() {
  local topic=$1
  local expected=$2
  local actual=""
  local deadline=$((SECONDS + 45))
  while (( SECONDS < deadline )); do
    actual="$(timeout 2s ros2 topic type "$topic" 2>/dev/null || true)"
    if [[ "$actual" == "$expected" ]]; then
      return 0
    fi
    if [[ -n "$actual" && "$actual" != "$expected" ]]; then
      echo "Unexpected type for $topic: $actual (expected $expected)" >&2
      return 21
    fi
    sleep 0.5
  done
  echo "Timed out waiting for $topic with type $expected" >&2
  return 22
}

wait_for_type /livox/lidar livox_ros_driver2/msg/CustomMsg
wait_for_type /livox/imu sensor_msgs/msg/Imu

ros2 topic list -t > /output/logs/live_input_topic_list.txt
ros2 node list > /output/logs/live_input_node_list.txt
ros2 topic info /livox/lidar --verbose > /output/logs/live_input_lidar_info.txt
ros2 topic info /livox/imu --verbose > /output/logs/live_input_imu_info.txt

set +e
timeout --signal=INT --kill-after=15s "${duration_sec}s" \
  ros2 bag record -o /output/data/live_input_probe \
  /livox/lidar /livox/imu \
  > /output/logs/live_input_record.log 2>&1
record_status=$?
set -e

if [[ $record_status -ne 0 && $record_status -ne 124 && $record_status -ne 130 ]]; then
  echo "ros2 bag record failed with status $record_status" >&2
  exit "$record_status"
fi
[[ -f /output/data/live_input_probe/metadata.yaml ]] || {
  echo "Input probe did not produce metadata.yaml" >&2
  exit 23
}
ros2 bag info /output/data/live_input_probe > /output/logs/live_input_bag_info.txt
'

exec "$run_wrapper" "${wrapper_args[@]}" -- \
  bash -c "$container_script" live-input-probe "$duration_sec"
