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

for node in "${required_nodes[@]}"; do
  sim_time="$(ros2 param get "$node" use_sim_time 2>/dev/null || true)"
  grep -Fq "True" <<<"$sim_time" || {
    echo "$node did not enable simulated time: $sim_time" >&2
    exit 1
  }
done

ros2 bag record --use-sim-time -o "$output" /state_estimation &
record_pid=$!
sleep 2
kill -0 "$record_pid" 2>/dev/null || { echo "Recorder exited before playback" >&2; exit 1; }

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

