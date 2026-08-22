#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 --image IMAGE --input-bag BAG_DIR --output-root DIR --name NAME [--domain-id ID]" >&2
}

image=
input_bag=
output_root=
name=
domain_id=86
while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) image=$2; shift 2 ;;
    --input-bag) input_bag=$2; shift 2 ;;
    --output-root) output_root=$2; shift 2 ;;
    --name) name=$2; shift 2 ;;
    --domain-id) domain_id=$2; shift 2 ;;
    *) usage; exit 2 ;;
  esac
done

if [[ -z "$image" || -z "$input_bag" || -z "$output_root" || -z "$name" ]]; then
  usage
  exit 2
fi
input_bag=$(realpath "$input_bag")
output_root=$(realpath "$output_root")
if [[ -e "$output_root/$name" ]]; then
  echo "refusing to overwrite $output_root/$name" >&2
  exit 1
fi
mkdir -p "$output_root"

docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e ROS_DOMAIN_ID="$domain_id" \
  -v "$input_bag:/input:ro" \
  -v "$output_root:/output:rw" \
  "$image" \
  bash -lc '
    set -eo pipefail
    source /opt/ros/humble/setup.bash
    source /opt/superodom_ws/install/setup.bash
    set -u
    launch_pid=
    recorder_pid=
    cleanup() {
      if [[ -n "$recorder_pid" ]]; then kill -INT "$recorder_pid" 2>/dev/null || true; fi
      if [[ -n "$launch_pid" ]]; then kill -INT -- -"$launch_pid" 2>/dev/null || true; fi
      wait "$recorder_pid" 2>/dev/null || true
      wait "$launch_pid" 2>/dev/null || true
    }
    trap cleanup EXIT INT TERM
    setsid ros2 launch super_odometry livox_humanoid.launch.py use_sim_time:=true &
    launch_pid=$!
    sleep 3
    ros2 bag record --use-sim-time -o "/output/'"$name"'" /state_estimation &
    recorder_pid=$!
    sleep 1
    kill -0 "$recorder_pid"
    ros2 bag play /input --clock 100 --rate 1.0
    sleep 2
    kill -INT "$recorder_pid"
    wait "$recorder_pid"
    recorder_pid=
    kill -INT -- -"$launch_pid" 2>/dev/null || true
    wait "$launch_pid" 2>/dev/null || true
    launch_pid=
    ros2 bag info "/output/'"$name"'"
  '
