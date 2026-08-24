#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: run_g1_kiss_live_verification.sh --run-dir PATH --network-interface IFACE [options]

Record a bounded, passive G1-4123 localization shadow with Motive evaluator
truth, raw Livox LiDAR/IMU, raw LowState/pelvis IMU, and live localization
outputs. The launcher creates no robot command publisher.

Options:
  --duration-sec N             default: 120; range: 30..600
  --robot-host IP              default: 192.168.123.164
  --robot-user USER            default: unitree
  --robot-dds-interface IFACE  default: eth0
  --motive-server IP           default: 172.24.68.77
  --rigid-body-id ID           default: 42
  --rigid-body-name NAME       default: G1_PELVIS_F_4123
  --ros-domain-id ID           default: 0
  --root-state-port PORT       default: 5575 (tcp://127.0.0.1:5575)
  --dry-run
EOF
}

original_args=("$@")
run_dir=""
network_interface=""
duration_sec=120
robot_host=192.168.123.164
robot_user=unitree
robot_dds_interface=eth0
robot_python=/home/unitree/miniforge3/envs/egonav-deploy/bin/python
robot_localization_root=/home/unitree/geo-179/G1_localization
motive_server=172.24.68.77
rigid_body_id=42
rigid_body_name=G1_PELVIS_F_4123
ros_domain_id=0
root_state_port=5575
live_lowstate_port=5589
record_lowstate_port=5590
dry_run=false

while (( $# )); do
  case "$1" in
    --run-dir) run_dir=$2; shift 2 ;;
    --network-interface) network_interface=$2; shift 2 ;;
    --duration-sec) duration_sec=$2; shift 2 ;;
    --robot-host) robot_host=$2; shift 2 ;;
    --robot-user) robot_user=$2; shift 2 ;;
    --robot-dds-interface) robot_dds_interface=$2; shift 2 ;;
    --motive-server) motive_server=$2; shift 2 ;;
    --rigid-body-id) rigid_body_id=$2; shift 2 ;;
    --rigid-body-name) rigid_body_name=$2; shift 2 ;;
    --ros-domain-id) ros_domain_id=$2; shift 2 ;;
    --root-state-port) root_state_port=$2; shift 2 ;;
    --dry-run) dry_run=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

[[ -n "$run_dir" && -n "$network_interface" ]] || { usage; exit 2; }
[[ "$duration_sec" =~ ^[0-9]+$ ]] && (( 10#$duration_sec >= 30 && 10#$duration_sec <= 600 )) || {
  echo "duration must be an integer from 30 through 600" >&2
  exit 2
}
[[ "$rigid_body_id" =~ ^[0-9]+$ ]] || { echo "invalid rigid-body ID" >&2; exit 2; }
[[ "$root_state_port" =~ ^[0-9]+$ ]] && (( 10#$root_state_port > 0 && 10#$root_state_port <= 65535 )) || {
  echo "invalid root-state port" >&2
  exit 2
}
[[ "$root_state_port" != "$live_lowstate_port" && "$root_state_port" != "$record_lowstate_port" ]] || {
  echo "root-state and LowState ports must differ" >&2
  exit 2
}
for value in "$network_interface" "$robot_dds_interface" "$robot_user" "$rigid_body_name"; do
  [[ "$value" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "unsafe identifier: $value" >&2; exit 2; }
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "$script_dir/.." && pwd -P)"
image="${G1_LOCALIZATION_IMAGE:-tml/g1-kiss-localization:1.3.0-humble}"
route_to_robot="$(ip route get "$robot_host" | head -1)"
[[ "$route_to_robot" != *" via "* ]] || { echo "robot route is not direct: $route_to_robot" >&2; exit 4; }
[[ "$route_to_robot" == *" dev $network_interface "* ]] || { echo "wrong robot NIC: $route_to_robot" >&2; exit 4; }
offboard_robot_address="$(awk '{for(i=1;i<=NF;i++) if($i=="src") print $(i+1)}' <<<"$route_to_robot")"
motive_route="$(ip route get "$motive_server" | head -1)"
motive_client_address="$(awk '{for(i=1;i<=NF;i++) if($i=="src") print $(i+1)}' <<<"$motive_route")"
[[ "$offboard_robot_address" == 192.168.123.* && -n "$motive_client_address" ]] || {
  echo "could not resolve G1 or Motive route" >&2
  exit 4
}

if [[ "$dry_run" == true ]]; then
  cat <<EOF
schema=g1_kiss_live_verification_dry_run_v1
run_dir=$run_dir
duration_sec=$duration_sec
robot=$robot_user@$robot_host
robot_interface=$network_interface
motive=$motive_server client=$motive_client_address
rigid_body=$rigid_body_name id=$rigid_body_id
image=$image
topics=/utlidar/cloud_livox_mid360,/utlidar/imu_livox_mid360,/g1/localization/pelvis_odom,/g1/localization/cloud_registered
capture_readiness=first fresh and fully healthy HSROOT02 packet; robot must remain stationary until READY is printed
root_state_endpoint=tcp://127.0.0.1:$root_state_port
command_capability=structurally_unavailable
EOF
  exit 0
fi

for command in docker git ip python3 scp sha256sum ssh; do
  command -v "$command" >/dev/null || { echo "missing command: $command" >&2; exit 2; }
done
[[ ! -e "$run_dir" ]] || { echo "refusing to overwrite $run_dir" >&2; exit 2; }
mkdir -p "$run_dir"/{logs,motive,lowstate,rosbag,runtime}
run_dir="$(cd "$run_dir" && pwd -P)"
trial_id="$(basename "$run_dir")"
remote_stage="/tmp/g1-kiss-verification-${trial_id//[^A-Za-z0-9_-]/_}"
source_commit="$(git -C "$repo_root" rev-parse HEAD)"
source_dirty_digest="$({ git -C "$repo_root" status --porcelain=v1 --untracked-files=all; } | sha256sum | awk '{print $1}')"
image_id="$(docker image inspect "$image" --format '{{.Id}}')"

python3 - "$run_dir/manifest.json" <<EOF
import json, pathlib, time
pathlib.Path(r"$run_dir/manifest.json").write_text(json.dumps({
  "schema": "g1_kiss_live_verification_manifest_v1",
  "status": "recording",
  "created_realtime_ns": time.time_ns(),
  "duration_sec": $duration_sec,
  "robot_identity": "G1-4123",
  "robot_host": "$robot_host",
  "robot_network_interface": "$network_interface",
  "motive_server": "$motive_server",
  "motive_client_address": "$motive_client_address",
  "rigid_body_id": $rigid_body_id,
  "rigid_body_name": "$rigid_body_name",
  "source_commit": "$source_commit",
  "source_dirty_digest": "$source_dirty_digest",
  "container_image": "$image",
  "container_image_id": "$image_id",
  "motive_role": "evaluator_only",
  "command_capability": "structurally_unavailable",
  "actuation_publishers_created": 0,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
EOF
printf '%q ' "$script_dir/run_g1_kiss_live_verification.sh" "${original_args[@]}" >"$run_dir/command.txt"
printf '\n' >>"$run_dir/command.txt"

ssh -o BatchMode=yes "$robot_user@$robot_host" \
  "test -x '$robot_python' && \
   '$robot_python' -c 'import unitree_sdk2py' && \
   test -d '$robot_localization_root/mocap_utils' && \
   mkdir -p '$remote_stage'"
scp -q -r "$repo_root/g1_root_state_bridge/g1_root_state_bridge" "$robot_user@$robot_host:$remote_stage/"
scp -q -r "$robot_user@$robot_host:$robot_localization_root/mocap_utils" "$run_dir/runtime/"
python3 - "$run_dir/runtime/mocap_utils" <<'PY'
from pathlib import Path
import sys
for path in Path(sys.argv[1]).glob("*.py"):
    source = path.read_bytes()
    normalized = source.replace(b"\xef\xbb\xbf", b"")
    if source != normalized:
        path.write_bytes(normalized)
PY
python3 "$script_dir/probe_g1_clock.py" \
  --target "$robot_user@$robot_host" \
  --output "$run_dir/clock_probe.json" \
  >"$run_dir/logs/clock_probe.log" 2>&1

live_duration=$((10#$duration_sec + 15))
"$script_dir/run_g1_kiss_live_stack.sh" \
  --network-interface "$network_interface" \
  --duration-sec "$live_duration" \
  --robot-host "$robot_host" \
  --robot-user "$robot_user" \
  --robot-dds-interface "$robot_dds_interface" \
  --ros-domain-id "$ros_domain_id" \
  --lowstate-port "$live_lowstate_port" \
  --root-state-port "$root_state_port" \
  >"$run_dir/logs/live_stack.log" 2>&1 &
live_pid=$!

cleanup() {
  for pid in "${bag_pid:-}" "${relay_pid:-}" "${lowstate_pid:-}" "${motive_pid:-}" "${live_pid:-}"; do
    if [[ -n "$pid" ]]; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  for pid in "${bag_pid:-}" "${relay_pid:-}" "${lowstate_pid:-}" "${motive_pid:-}" "${live_pid:-}"; do
    if [[ -n "$pid" ]]; then
      wait "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

# The producer is intentionally fail-closed until the robot has remained
# stationary long enough to estimate Livox gyro bias and both source-clock fits
# are admitted.  Observe its command-incapable HSROOT02 output directly rather
# than adding a temporary ROS DDS participant, which can itself perturb the
# Livox stream during capture startup.
if ! PYTHONPATH="$repo_root/g1_root_state_bridge" python3 \
  "$script_dir/wait_for_root_state.py" \
  --endpoint "tcp://127.0.0.1:$root_state_port" \
  --timeout-sec 12 \
  --output "$run_dir/runtime/readiness-root-state.json" \
  >"$run_dir/logs/readiness.log" 2>&1; then
  echo "localization produced no synchronized root state within 12s; keep the G1 stationary and retry" >&2
  exit 5
fi
python3 - "$run_dir/manifest.json" "$run_dir/runtime/readiness-root-state.json" <<'PY'
import json, pathlib, sys, time
path = pathlib.Path(sys.argv[1])
value = json.loads(path.read_text(encoding="utf-8"))
readiness = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
value["capture_ready_realtime_ns"] = time.time_ns()
value["capture_readiness"] = "first fresh and fully healthy HSROOT02 packet"
value["calibration_digest"] = readiness["calibration_digest"]
value["localization_source_epoch"] = readiness["source_epoch"]
path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
echo "READY FOR OPERATOR-CONTROLLED AMO: recording ${duration_sec}s now"

cyclonedds_uri="<CycloneDDS><Domain id=\"any\"><General><Interfaces><NetworkInterface name=\"$network_interface\" /></Interfaces></General></Domain></CycloneDDS>"

python3 "$script_dir/record_natnet_reference.py" \
  --sdk-root "$run_dir/runtime" \
  --server-address "$motive_server" \
  --client-address "$motive_client_address" \
  --connection multicast \
  --rigid-body-id "$rigid_body_id" \
  --rigid-body-name "$rigid_body_name" \
  --duration-sec "$duration_sec" \
  --minimum-tracking-coverage 0.99 \
  --output "$run_dir/motive/frames.jsonl" \
  --summary "$run_dir/motive/summary.json" \
  >"$run_dir/logs/motive.log" 2>&1 &
motive_pid=$!

PYTHONPATH="$repo_root/g1_root_state_bridge" python3 \
  -m g1_root_state_bridge.g1_dynamic_capture_recorder \
  --bind-port "$record_lowstate_port" \
  --output "$run_dir/lowstate/packets.bin" \
  --summary "$run_dir/lowstate/summary.json" \
  --duration-sec "$duration_sec" \
  >"$run_dir/logs/lowstate_recorder.log" 2>&1 &
lowstate_pid=$!
sleep 0.25

ssh -o BatchMode=yes "$robot_user@$robot_host" "bash -lc '
  PYTHONPATH="$remote_stage" "$robot_python" \
    -m g1_root_state_bridge.g1_dynamic_capture_relay \
    --target-host "$offboard_robot_address" \
    --target-port "$record_lowstate_port" \
    --network-interface "$robot_dds_interface" \
    --domain-id 0 \
    --duration-sec "$duration_sec"
'" >"$run_dir/logs/lowstate_relay.log" 2>&1 &
relay_pid=$!

docker run --rm --network host \
  --user "$(id -u):$(id -g)" \
  --tmpfs /tmp:rw,nosuid,size=256m,mode=1777 \
  --volume "$run_dir:/output" \
  --env "ROS_DOMAIN_ID=$ros_domain_id" \
  --env RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  --env "CYCLONEDDS_URI=$cyclonedds_uri" \
  "$image" bash -lc "
    set +e
    timeout --signal=INT --kill-after=15s '${duration_sec}s' \\
      ros2 bag record -o /output/rosbag/live_localization \\
      /utlidar/cloud_livox_mid360 \\
      /utlidar/imu_livox_mid360 \\
      /g1/localization/pelvis_odom \\
      /g1/localization/cloud_registered
    status=\$?
    [[ \$status -eq 0 || \$status -eq 124 || \$status -eq 130 ]]
  " >"$run_dir/logs/rosbag.log" 2>&1 &
bag_pid=$!

set +e
wait "$motive_pid"; motive_status=$?
wait "$lowstate_pid"; lowstate_status=$?
wait "$relay_pid"; relay_status=$?
wait "$bag_pid"; bag_status=$?
wait "$live_pid"; live_status=$?
set -e
motive_pid=""; lowstate_pid=""; relay_pid=""; bag_pid=""; live_pid=""

python3 - "$run_dir/process_status.json" <<EOF
import json, pathlib
pathlib.Path(r"$run_dir/process_status.json").write_text(json.dumps({
  "live_stack": $live_status,
  "motive": $motive_status,
  "lowstate_recorder": $lowstate_status,
  "lowstate_relay": $relay_status,
  "rosbag": $bag_status,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
EOF
python3 "$script_dir/validate_g1_kiss_live_verification.py" \
  --run-dir "$run_dir" \
  --output "$run_dir/validation.json"
python3 - "$run_dir/manifest.json" <<'PY'
import json, pathlib, sys, time
path = pathlib.Path(sys.argv[1])
value = json.loads(path.read_text(encoding="utf-8"))
value["status"] = "complete"
value["completed_realtime_ns"] = time.time_ns()
path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
echo "PASS: $run_dir"
