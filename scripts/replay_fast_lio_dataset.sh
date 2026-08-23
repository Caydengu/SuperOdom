#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: replay_fast_lio_dataset.sh --bag PATH --output PATH --install PATH --config PATH [--rate FLOAT]

Replay a normalized G1 PointCloud2/IMU bag through Gio's captured ROS 2
FAST-LIO2 source. The script records /Odometry only and never creates a robot
command publisher.
EOF
}

bag=""
output=""
install=""
config=""
rate=1.0

while (( $# )); do
  case "$1" in
    --bag) bag=$2; shift 2 ;;
    --output) output=$2; shift 2 ;;
    --install) install=$2; shift 2 ;;
    --config) config=$2; shift 2 ;;
    --rate) rate=$2; shift 2 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

[[ -e "$bag" ]] || { echo "Missing input bag: $bag" >&2; exit 2; }
[[ -d "$install" ]] || { echo "Missing FAST-LIO install: $install" >&2; exit 2; }
[[ -r "$config" ]] || { echo "Missing FAST-LIO config: $config" >&2; exit 2; }
[[ -n "$output" && ! -e "$output" ]] || {
  echo "Output must be new: $output" >&2
  exit 2
}
mkdir -p "$(dirname "$output")"
set +u
source "$install/setup.bash"
set -u

launch_pid=""
record_pid=""

stop_process() {
  local pid=$1
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill -INT "$pid" 2>/dev/null || true
    for _ in {1..30}; do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.2
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

config_dir=$(dirname "$config")
config_name=$(basename "$config")
ros2 launch fast_lio mapping.launch.py \
  "config_path:=$config_dir" \
  "config_file:=$config_name" \
  use_sim_time:=true rviz:=false &
launch_pid=$!

ready=false
for _ in {1..80}; do
  if ! kill -0 "$launch_pid" 2>/dev/null; then
    echo "FAST-LIO launch exited before readiness" >&2
    wait "$launch_pid" || true
    exit 1
  fi
  if ros2 node list 2>/dev/null | grep -Fxq /laser_mapping; then
    ready=true
    break
  fi
  sleep 0.25
done
[[ "$ready" == true ]] || { echo "Timed out waiting for /laser_mapping" >&2; exit 1; }
ros2 param get /laser_mapping use_sim_time | grep -Fq 'Boolean value is: True'

ros2 bag record --use-sim-time -o "$output" /Odometry &
record_pid=$!
for _ in {1..50}; do
  clock_info=$(ros2 topic info /clock --verbose 2>/dev/null || true)
  grep -Fq 'Node name: rosbag2_recorder' <<<"$clock_info" && break
  sleep 0.1
done
grep -Fq 'Node name: rosbag2_recorder' <<<"${clock_info:-}" || {
  echo "Recorder did not advertise /clock" >&2
  exit 1
}

ros2 bag play "$bag" --clock --rate "$rate"
sleep 2
stop_process "$record_pid"
record_pid=""
stop_process "$launch_pid"
launch_pid=""

bag_info=$(ros2 bag info "$output")
printf '%s\n' "$bag_info"
message_count=$(sed -n '/Topic: \/Odometry / s/.*Count: \([0-9][0-9]*\).*/\1/p' \
  <<<"$bag_info" | head -n 1)
[[ "$message_count" =~ ^[0-9]+$ ]] && (( message_count > 0 )) || {
  echo "FAST-LIO replay produced no /Odometry" >&2
  exit 1
}
echo "Validated FAST-LIO /Odometry messages: $message_count"
