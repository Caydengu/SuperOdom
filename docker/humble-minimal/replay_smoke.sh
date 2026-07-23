#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: replay_smoke.sh --bag /data/BAG --output /output/NAME [--config PATH] [--rate FLOAT]
EOF
}

bag=""
output=""
config=/opt/superodom_ws/install/share/super_odometry/config/livox_mid360_gantry.yaml
rate=1.0

while (( $# )); do
  case "$1" in
    --bag)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      bag=$2
      shift 2
      ;;
    --output)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      output=$2
      shift 2
      ;;
    --config)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      config=$2
      shift 2
      ;;
    --rate)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      rate=$2
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

[[ -n "$bag" && -e "$bag" ]] || { echo "Input bag does not exist: $bag" >&2; exit 2; }
[[ -n "$output" ]] || { echo "--output is required" >&2; exit 2; }
[[ ! -e "$output" ]] || { echo "Refusing to overwrite output: $output" >&2; exit 2; }
[[ -r "$config" ]] || { echo "Configuration is not readable: $config" >&2; exit 2; }
mkdir -p "$(dirname "$output")"

launch_pid=""
record_pid=""

stop_process() {
  local pid=$1
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill -INT "$pid" 2>/dev/null || true
    for _ in {1..20}; do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.25
    done
    kill -TERM "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
}

cleanup() {
  local status=$?
  stop_process "$record_pid"
  stop_process "$launch_pid"
  exit "$status"
}
trap cleanup EXIT INT TERM

find_process() {
  local executable=$1
  local cmdline pid
  local -a process_args
  for cmdline in /proc/[0-9]*/cmdline; do
    [[ -r "$cmdline" ]] || continue
    process_args=()
    mapfile -d '' -t process_args < "$cmdline" || true
    if [[ "${process_args[0]:-}" == "/opt/superodom_ws/install/lib/super_odometry/$executable" ]]; then
      pid=${cmdline#/proc/}
      printf '%s\n' "${pid%/cmdline}"
      return 0
    fi
  done
  return 1
}

assert_process_sim_time() {
  local executable=$1
  local pid param_file found=false
  local -a process_args
  pid="$(find_process "$executable")" || {
    echo "Could not locate process for $executable" >&2
    return 1
  }
  mapfile -d '' -t process_args < "/proc/$pid/cmdline"
  for (( index=0; index + 1 < ${#process_args[@]}; index++ )); do
    if [[ "${process_args[$index]}" != "--params-file" ]]; then
      continue
    fi
    param_file=${process_args[$((index + 1))]}
    if [[ -r "$param_file" ]] && grep -Eq 'use_sim_time:[[:space:]]*true' "$param_file"; then
      found=true
      break
    fi
  done
  [[ "$found" == true ]] || {
    echo "$executable was not launched with use_sim_time=true" >&2
    return 1
  }
  echo "Validated use_sim_time=true for $executable (pid $pid)"
}

assert_recorder_clock_endpoint() {
  # A sim-time recorder does not create its requested topic subscriptions until
  # it sees /clock. Ensure its clock endpoint is present before playback; the
  # player's continuous clock then completes the handshake well before the
  # estimator finishes initialization and begins publishing state estimates.
  local clock_info
  for _ in {1..50}; do
    clock_info="$(ros2 topic info /clock --verbose 2>/dev/null || true)"
    if grep -Fq "Node name: rosbag2_recorder" <<<"$clock_info"; then
      echo "Validated recorder subscription to /clock"
      return 0
    fi
    sleep 0.1
  done
  echo "Recorder did not advertise its /clock subscription" >&2
  return 1
}

ros2 launch super_odometry livox_humanoid.launch.py \
  "config_file:=$config" \
  use_sim_time:=true &
launch_pid=$!

required_nodes=(/feature_extraction_node /laser_mapping_node /imu_preintegration_node)
ready=false
for _ in {1..60}; do
  if ! kill -0 "$launch_pid" 2>/dev/null; then
    echo "SuperOdometry launch exited before its nodes became ready" >&2
    wait "$launch_pid" || true
    exit 1
  fi
  node_list="$(ros2 node list 2>/dev/null || true)"
  ready=true
  for node in "${required_nodes[@]}"; do
    if ! grep -Fxq "$node" <<<"$node_list"; then
      ready=false
      break
    fi
  done
  [[ "$ready" == true ]] && break
  sleep 0.5
done
[[ "$ready" == true ]] || { echo "Timed out waiting for SuperOdometry nodes" >&2; exit 1; }

for executable in feature_extraction_node laser_mapping_node imu_preintegration_node; do
  assert_process_sim_time "$executable"
done

ros2 bag record --use-sim-time -o "$output" \
  /state_estimation \
  /lidar_pipeline_events &
record_pid=$!
kill -0 "$record_pid" 2>/dev/null || { echo "Recorder exited before playback" >&2; exit 1; }
assert_recorder_clock_endpoint

ros2 bag play "$bag" --clock --rate "$rate"
sleep 2

stop_process "$record_pid"
record_pid=""
stop_process "$launch_pid"
launch_pid=""

bag_info="$(ros2 bag info "$output")"
printf '%s\n' "$bag_info"
message_count="$(sed -n '/Topic: \/state_estimation/ s/.*Count: \([0-9][0-9]*\).*/\1/p' <<<"$bag_info" | head -n 1)"
[[ "$message_count" =~ ^[0-9]+$ ]] && (( message_count > 0 )) || {
  echo "Replay produced no /state_estimation messages" >&2
  exit 1
}
echo "Validated /state_estimation messages: $message_count"

pipeline_event_count="$(
  sed -n \
    '/Topic: \/lidar_pipeline_events/ s/.*Count: \([0-9][0-9]*\).*/\1/p' \
    <<<"$bag_info" | head -n 1
)"
[[ "$pipeline_event_count" =~ ^[0-9]+$ ]] && (( pipeline_event_count > 0 )) || {
  echo "Replay produced no /lidar_pipeline_events messages" >&2
  exit 1
}
echo "Validated /lidar_pipeline_events messages: $pipeline_event_count"
