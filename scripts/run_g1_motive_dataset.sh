#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: run_g1_motive_dataset.sh --run-dir PATH [options]

Records synchronized offline-localization inputs without creating robot command
publishers: Motive, Livox point clouds + IMU, G1 lowstate + pelvis IMU, and
RealSense RGB-D. Use --rigid-body-id auto only when Motive streams one body.

Options:
  --duration-sec N                 default: 15
  --rigid-body-id ID|auto         default: auto
  --rigid-body-name NAME          default: G1_PELVIS
  --motive-server IP              default: 172.24.68.67
  --motive-connection MODE        multicast or unicast; default: multicast
  --robot-host IP                 default: 192.168.123.164
  --robot-user USER               default: unitree
  --robot-interface IFACE         default: auto-detect route interface
  --motive-client-address IP      default: auto-detect route source
  --ros-domain-id ID              default: 42
  --dynamic-port PORT             default: 5589
  --dry-run
EOF
}

run_dir=""
duration_sec=15
rigid_body_id=auto
rigid_body_name=G1_PELVIS
motive_server=172.24.68.67
motive_connection=multicast
robot_host=192.168.123.164
robot_user=unitree
robot_interface=""
motive_client_address=""
ros_domain_id=42
dynamic_port=5589
dry_run=false

while (( $# )); do
  case "$1" in
    --run-dir) run_dir=$2; shift 2 ;;
    --duration-sec) duration_sec=$2; shift 2 ;;
    --rigid-body-id) rigid_body_id=$2; shift 2 ;;
    --rigid-body-name) rigid_body_name=$2; shift 2 ;;
    --motive-server) motive_server=$2; shift 2 ;;
    --motive-connection) motive_connection=$2; shift 2 ;;
    --robot-host) robot_host=$2; shift 2 ;;
    --robot-user) robot_user=$2; shift 2 ;;
    --robot-interface) robot_interface=$2; shift 2 ;;
    --motive-client-address) motive_client_address=$2; shift 2 ;;
    --ros-domain-id) ros_domain_id=$2; shift 2 ;;
    --dynamic-port) dynamic_port=$2; shift 2 ;;
    --dry-run) dry_run=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

[[ -n "$run_dir" ]] || { echo "--run-dir is required" >&2; exit 2; }
[[ "$duration_sec" =~ ^[0-9]+$ ]] && (( 10#$duration_sec >= 10 && 10#$duration_sec <= 600 )) || {
  echo "--duration-sec must be an integer from 10 through 600" >&2
  exit 2
}
[[ "$rigid_body_id" == auto || "$rigid_body_id" =~ ^[0-9]+$ ]] || {
  echo "--rigid-body-id must be nonnegative or auto" >&2
  exit 2
}
[[ "$motive_connection" == multicast || "$motive_connection" == unicast ]] || {
  echo "--motive-connection must be multicast or unicast" >&2
  exit 2
}

for command in docker ip python3 scp ssh; do
  command -v "$command" >/dev/null || { echo "Missing command: $command" >&2; exit 2; }
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_dir="$(cd "$script_dir/.." && pwd -P)"
holosoma_dir=/move/u/caydengu/cayden/holosoma_secret/holosoma

route_to_robot="$(ip route get "$robot_host" | head -1)"
if grep -qE '(^| )via( |$)' <<<"$route_to_robot"; then
  echo "Robot route is not direct: $route_to_robot" >&2
  echo "Restore the dedicated G1 NIC and 192.168.123.11/24 before capture." >&2
  exit 4
fi
if [[ -z "$robot_interface" ]]; then
  robot_interface="$(awk '{for(i=1;i<=NF;i++) if($i=="dev") print $(i+1)}' <<<"$route_to_robot")"
fi
oslo_robot_address="$(awk '{for(i=1;i<=NF;i++) if($i=="src") print $(i+1)}' <<<"$route_to_robot")"
if [[ -z "$motive_client_address" ]]; then
  motive_route="$(ip route get "$motive_server" | head -1)"
  motive_client_address="$(awk '{for(i=1;i<=NF;i++) if($i=="src") print $(i+1)}' <<<"$motive_route")"
fi
[[ -n "$robot_interface" && -n "$oslo_robot_address" && -n "$motive_client_address" ]] || {
  echo "Could not resolve robot or Motive network routes" >&2
  exit 2
}
[[ "$oslo_robot_address" == 192.168.123.* ]] || {
  echo "Unexpected G1-side source address: $oslo_robot_address" >&2
  exit 4
}

if [[ "$dry_run" == true ]]; then
  cat <<EOF
run_dir=$run_dir
duration_sec=$duration_sec
robot=$robot_user@$robot_host interface=$robot_interface oslo_address=$oslo_robot_address
motive_server=$motive_server client=$motive_client_address connection=$motive_connection
rigid_body=$rigid_body_name id=$rigid_body_id
actuation_publishers=0
EOF
  exit 0
fi

[[ ! -e "$run_dir" ]] || { echo "Refusing to overwrite existing run: $run_dir" >&2; exit 2; }
mkdir -p "$run_dir"/{logs,motive,lowstate,lidar,vision,runtime}
run_dir="$(cd "$run_dir" && pwd -P)"
trial_id="$(basename "$run_dir")"
remote_stage="/tmp/g1-localization-capture-${trial_id//[^a-zA-Z0-9_-]/_}"
vision_stage="/tmp/g1-localization-vision-${trial_id//[^a-zA-Z0-9_-]/_}"

python3 - "$run_dir/manifest.json" <<EOF
import json, pathlib, time
path = pathlib.Path(r"$run_dir/manifest.json")
path.write_text(json.dumps({
    "schema": "g1_motive_localization_dataset_manifest_v1",
    "created_realtime_ns": time.time_ns(),
    "duration_sec": $duration_sec,
    "robot_host": "$robot_host",
    "robot_interface": "$robot_interface",
    "motive_server": "$motive_server",
    "motive_client_address": "$motive_client_address",
    "motive_connection": "$motive_connection",
    "rigid_body_name": "$rigid_body_name",
    "rigid_body_id": "$rigid_body_id",
    "streams": ["motive", "livox_lidar", "livox_imu", "g1_lowstate", "pelvis_imu", "rgb", "depth"],
    "coordinate_contract": "Motive native coordinates; calibration and alignment happen offline",
    "actuation_publishers_created": 0,
}, indent=2, sort_keys=True) + "\\n")
EOF

echo "Checking and starting passive onboard services..."
ssh -o BatchMode=yes "$robot_user@$robot_host" "bash -lc '
  if ! tmux has-session -t teleimager_capture 2>/dev/null; then
    tmux new-session -d -s teleimager_capture \\
      \"cd ~/teleimager && exec ~/.local/bin/teleimager-server --rs --no-reload-uvc\"
  fi
  if ! tmux has-session -t g1_lidar_capture 2>/dev/null; then
    tmux new-session -d -s g1_lidar_capture \\
      \"source /home/unitree/G1_localization/ws_slam/install/setup.bash && \\
       export ROS_DOMAIN_ID=$ros_domain_id && \\
       exec ros2 launch livox_ros_driver2 msg_MID360_launch.py\"
  fi
'"
sleep 3

echo "Staging read-only relay and NatNet SDK..."
ssh -o BatchMode=yes "$robot_user@$robot_host" "mkdir -p '$remote_stage'"
scp -q -r "$repo_dir/g1_root_state_bridge/g1_root_state_bridge" "$robot_user@$robot_host:$remote_stage/"
scp -q -r "$robot_user@$robot_host:/home/unitree/G1_localization/mocap_utils" "$run_dir/runtime/"
python3 "$script_dir/probe_g1_clock.py" \
  --target "$robot_user@$robot_host" \
  --output "$run_dir/clock_probe.json" \
  >"$run_dir/logs/clock_probe.log" 2>&1

echo "Starting all passive recorders for ${duration_sec}s..."
python3 "$script_dir/record_natnet_reference.py" \
  --sdk-root "$run_dir/runtime" \
  --server-address "$motive_server" \
  --client-address "$motive_client_address" \
  --connection "$motive_connection" \
  --rigid-body-id "$rigid_body_id" \
  --rigid-body-name "$rigid_body_name" \
  --duration-sec "$duration_sec" \
  --output "$run_dir/motive/frames.jsonl" \
  --summary "$run_dir/motive/summary.json" \
  >"$run_dir/logs/motive.log" 2>&1 &
motive_pid=$!

PYTHONPATH="$repo_dir/g1_root_state_bridge" python3 -m g1_root_state_bridge.g1_dynamic_capture_recorder \
  --bind-port "$dynamic_port" \
  --output "$run_dir/lowstate/packets.bin" \
  --summary "$run_dir/lowstate/summary.json" \
  --duration-sec "$duration_sec" \
  >"$run_dir/logs/lowstate_recorder.log" 2>&1 &
lowstate_pid=$!
sleep 0.5

ssh -o BatchMode=yes "$robot_user@$robot_host" "bash -lc '
  PYTHONPATH=\"$remote_stage\" /home/unitree/miniconda3/envs/sim2sim/bin/python \\
    -m g1_root_state_bridge.g1_dynamic_capture_relay \\
    --target-host \"$oslo_robot_address\" --target-port \"$dynamic_port\" \\
    --network-interface eth0 --domain-id 0 --duration-sec \"$duration_sec\"
'" >"$run_dir/logs/lowstate_relay.log" 2>&1 &
relay_pid=$!

"$repo_dir/docker/humble-minimal/live_input_probe.sh" \
  --output-dir "$run_dir/lidar" \
  --network-interface "$robot_interface" \
  --ros-domain-id "$ros_domain_id" \
  --duration-sec "$duration_sec" \
  >"$run_dir/logs/lidar.log" 2>&1 &
lidar_pid=$!

docker exec holosoma bash -lc "
  set -e
  cd /workspace/holosoma_secret/holosoma
  source scripts/source_inference_setup.sh
  mkdir -p '$vision_stage'
  set +e
  timeout --signal=INT --kill-after=10s '${duration_sec}s' \\
    python -m holosoma_inference.recording.run_video_recorder \\
      --host '$robot_host' --output-dir '$vision_stage' --run-id capture \\
      --record-depth --depth-port 55559 --flush-interval-frames 300
  status=\$?
  [[ \$status -eq 0 || \$status -eq 124 || \$status -eq 130 ]]
" >"$run_dir/logs/vision.log" 2>&1 &
vision_pid=$!

set +e
wait "$motive_pid"; motive_status=$?
wait "$lowstate_pid"; lowstate_status=$?
wait "$relay_pid"; relay_status=$?
wait "$lidar_pid"; lidar_status=$?
wait "$vision_pid"; vision_status=$?
set -e

docker cp "holosoma:$vision_stage/capture" "$run_dir/vision/" >"$run_dir/logs/vision_copy.log" 2>&1 || {
  echo "Failed to copy RGB-D capture; see logs/vision_copy.log" >&2
  vision_status=1
}

python3 - "$run_dir/process_status.json" <<EOF
import json, pathlib
pathlib.Path(r"$run_dir/process_status.json").write_text(json.dumps({
    "motive": $motive_status,
    "lowstate_recorder": $lowstate_status,
    "lowstate_relay": $relay_status,
    "lidar": $lidar_status,
    "vision": $vision_status,
}, indent=2, sort_keys=True) + "\\n")
EOF

python3 "$script_dir/validate_g1_motive_dataset.py" --run-dir "$run_dir" --output "$run_dir/validation.json"
echo "PASS: $run_dir"
