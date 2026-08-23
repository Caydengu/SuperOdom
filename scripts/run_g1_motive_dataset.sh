#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: run_g1_motive_dataset.sh --run-dir PATH [options]

Records synchronized offline-localization inputs without creating robot command
publishers: Motive, native MID-360 point clouds + IMU, and G1 lowstate + pelvis
IMU. RealSense RGB-D is an optional diagnostic stream. Use --rigid-body-id auto
only when Motive streams one body.

Options:
  --duration-sec N                 default: 15
  --rigid-body-id ID|auto         default: auto
  --rigid-body-name NAME          default: G1_PELVIS
  --motive-server IP              default: 172.24.68.77
  --motive-connection MODE        multicast or unicast; default: multicast
  --robot-host IP                 default: 192.168.123.164
  --robot-user USER               default: unitree
  --robot-interface IFACE         default: auto-detect route interface
  --motive-client-address IP      default: auto-detect route source
  --ros-domain-id ID              default: 0
  --dynamic-port PORT             default: 5589
  --robot-dds-interface IFACE     default: eth0
  --robot-python PATH             default: /home/unitree/miniforge3/envs/egonav-deploy/bin/python
  --robot-localization-root PATH  default: /home/unitree/geo-179/G1_localization
  --lidar-topic TOPIC             default: /utlidar/cloud_livox_mid360
  --lidar-type TYPE               default: sensor_msgs/msg/PointCloud2
  --imu-topic TOPIC               default: /utlidar/imu_livox_mid360
  --imu-type TYPE                 default: sensor_msgs/msg/Imu
  --vision-mode MODE              disabled or teleimager; default: disabled
  --teleimager-dir PATH           default: /home/unitree/teleimager
  --dry-run
EOF
}

run_dir=""
duration_sec=15
rigid_body_id=auto
rigid_body_name=G1_PELVIS
motive_server=172.24.68.77
motive_connection=multicast
robot_host=192.168.123.164
robot_user=unitree
robot_interface=""
motive_client_address=""
ros_domain_id=0
dynamic_port=5589
robot_dds_interface=eth0
robot_python=/home/unitree/miniforge3/envs/egonav-deploy/bin/python
robot_localization_root=/home/unitree/geo-179/G1_localization
lidar_topic=/utlidar/cloud_livox_mid360
lidar_type=sensor_msgs/msg/PointCloud2
imu_topic=/utlidar/imu_livox_mid360
imu_type=sensor_msgs/msg/Imu
vision_mode=disabled
teleimager_dir=/home/unitree/teleimager
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
    --robot-dds-interface) robot_dds_interface=$2; shift 2 ;;
    --robot-python) robot_python=$2; shift 2 ;;
    --robot-localization-root) robot_localization_root=$2; shift 2 ;;
    --lidar-topic) lidar_topic=$2; shift 2 ;;
    --lidar-type) lidar_type=$2; shift 2 ;;
    --imu-topic) imu_topic=$2; shift 2 ;;
    --imu-type) imu_type=$2; shift 2 ;;
    --vision-mode) vision_mode=$2; shift 2 ;;
    --teleimager-dir) teleimager_dir=$2; shift 2 ;;
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
[[ "$vision_mode" == disabled || "$vision_mode" == teleimager ]] || {
  echo "--vision-mode must be disabled or teleimager" >&2
  exit 2
}
for topic in "$lidar_topic" "$imu_topic"; do
  [[ "$topic" =~ ^/[A-Za-z0-9_/]+$ ]] || {
    echo "Topics must be absolute ROS names containing letters, digits, underscores, and slashes" >&2
    exit 2
  }
done
for message_type in "$lidar_type" "$imu_type"; do
  [[ "$message_type" =~ ^[A-Za-z0-9_]+/msg/[A-Za-z0-9_]+$ ]] || {
    echo "Message types must use package/msg/Type syntax" >&2
    exit 2
  }
done
for remote_path in "$robot_python" "$robot_localization_root" "$teleimager_dir"; do
  [[ "$remote_path" =~ ^/[A-Za-z0-9._/-]+$ ]] || {
    echo "Robot paths must be absolute and shell-safe" >&2
    exit 2
  }
done
[[ "$robot_dds_interface" =~ ^[A-Za-z0-9_.:-]+$ ]] || {
  echo "--robot-dds-interface contains unsupported characters" >&2
  exit 2
}

for command in docker git ip python3 scp sha256sum ssh; do
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
ros_domain_id=$ros_domain_id
lidar=$lidar_topic type=$lidar_type
imu=$imu_topic type=$imu_type
robot_localization_root=$robot_localization_root
robot_dds_interface=$robot_dds_interface
robot_python=$robot_python
vision_mode=$vision_mode
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
source_commit="$(git -C "$repo_dir" rev-parse HEAD)"
source_branch="$(git -C "$repo_dir" branch --show-current)"
source_dirty_digest="$({ git -C "$repo_dir" status --porcelain=v1 --untracked-files=all; } | sha256sum | awk '{print $1}')"
if [[ "$vision_mode" == teleimager ]]; then
  vision_enabled_py=True
else
  vision_enabled_py=False
fi

python3 - "$run_dir/manifest.json" <<EOF
import json, pathlib, time
path = pathlib.Path(r"$run_dir/manifest.json")
path.write_text(json.dumps({
    "schema": "g1_motive_localization_dataset_manifest_v1",
    "status": "recording",
    "created_realtime_ns": time.time_ns(),
    "duration_sec": $duration_sec,
    "robot_host": "$robot_host",
    "robot_interface": "$robot_interface",
    "motive_server": "$motive_server",
    "motive_client_address": "$motive_client_address",
    "motive_connection": "$motive_connection",
    "rigid_body_name": "$rigid_body_name",
    "rigid_body_id": "$rigid_body_id",
    "streams": ["motive", "livox_lidar", "livox_imu", "g1_lowstate", "pelvis_imu"] + (["rgb", "depth"] if $vision_enabled_py else []),
    "robot_localization_root": "$robot_localization_root",
    "robot_dds_interface": "$robot_dds_interface",
    "robot_python": "$robot_python",
    "ros_domain_id": $ros_domain_id,
    "lidar_topic": "$lidar_topic",
    "lidar_type": "$lidar_type",
    "imu_topic": "$imu_topic",
    "imu_type": "$imu_type",
    "vision_mode": "$vision_mode",
    "vision_enabled": $vision_enabled_py,
    "source": {
        "repo_path": "$repo_dir",
        "branch": "$source_branch",
        "commit": "$source_commit",
        "dirty_state_sha256": "$source_dirty_digest",
    },
    "retention": {
        "raw_evidence": "retain immutable Motive, LiDAR/IMU, and lowstate streams",
        "derived_evidence": "validation.json, metrics.json, summary.md, and logs",
    },
    "coordinate_contract": "Motive native coordinates; calibration and alignment happen offline",
    "actuation_publishers_created": 0,
}, indent=2, sort_keys=True) + "\\n")
EOF

effective_command=(
  "$script_dir/run_g1_motive_dataset.sh"
  --run-dir "$run_dir"
  --duration-sec "$duration_sec"
  --rigid-body-id "$rigid_body_id"
  --rigid-body-name "$rigid_body_name"
  --motive-server "$motive_server"
  --motive-connection "$motive_connection"
  --robot-host "$robot_host"
  --robot-user "$robot_user"
  --robot-interface "$robot_interface"
  --motive-client-address "$motive_client_address"
  --ros-domain-id "$ros_domain_id"
  --dynamic-port "$dynamic_port"
  --robot-dds-interface "$robot_dds_interface"
  --robot-python "$robot_python"
  --robot-localization-root "$robot_localization_root"
  --lidar-topic "$lidar_topic"
  --lidar-type "$lidar_type"
  --imu-topic "$imu_topic"
  --imu-type "$imu_type"
  --vision-mode "$vision_mode"
  --teleimager-dir "$teleimager_dir"
)
printf '%q ' "${effective_command[@]}" >"$run_dir/command.txt"
printf '\n' >>"$run_dir/command.txt"

echo "Checking passive onboard sources..."
ssh -o BatchMode=yes "$robot_user@$robot_host" "bash -lc '
  set -e
  test -d \"$robot_localization_root/mocap_utils\"
  test -x \"$robot_python\"
  \"$robot_python\" -c \"import unitree_sdk2py\"
  source \"$robot_localization_root/ws_slam/install/setup.bash\"
  export ROS_DOMAIN_ID=$ros_domain_id
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
  test \"\$(timeout 5s ros2 topic type \"$lidar_topic\")\" = \"$lidar_type\"
  test \"\$(timeout 5s ros2 topic type \"$imu_topic\")\" = \"$imu_type\"
'"
if [[ "$vision_mode" == teleimager ]]; then
  ssh -o BatchMode=yes "$robot_user@$robot_host" "bash -lc '
    set -e
    test -d \"$teleimager_dir\"
    if ! tmux has-session -t teleimager_capture 2>/dev/null; then
      tmux new-session -d -s teleimager_capture \\
        \"cd '$teleimager_dir' && exec ~/.local/bin/teleimager-server --rs --no-reload-uvc\"
    fi
  '"
  sleep 3
fi

echo "Staging read-only relay and NatNet SDK..."
ssh -o BatchMode=yes "$robot_user@$robot_host" "mkdir -p '$remote_stage'"
scp -q -r "$repo_dir/g1_root_state_bridge/g1_root_state_bridge" "$robot_user@$robot_host:$remote_stage/"
scp -q -r "$robot_user@$robot_host:$robot_localization_root/mocap_utils" "$run_dir/runtime/"
python3 - "$run_dir/runtime/mocap_utils" <<'PY'
from pathlib import Path
import sys

sdk_root = Path(sys.argv[1])
for source_path in sdk_root.glob("*.py"):
    source = source_path.read_bytes()
    normalized = source.replace(b"\xef\xbb\xbf", b"")
    if normalized != source:
        source_path.write_bytes(normalized)
PY
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
  PYTHONPATH=\"$remote_stage\" \"$robot_python\" \\
    -m g1_root_state_bridge.g1_dynamic_capture_relay \\
    --target-host \"$oslo_robot_address\" --target-port \"$dynamic_port\" \\
    --network-interface \"$robot_dds_interface\" --domain-id 0 --duration-sec \"$duration_sec\"
'" >"$run_dir/logs/lowstate_relay.log" 2>&1 &
relay_pid=$!

"$repo_dir/docker/humble-minimal/live_input_probe.sh" \
  --output-dir "$run_dir/lidar" \
  --network-interface "$robot_interface" \
  --ros-domain-id "$ros_domain_id" \
  --duration-sec "$duration_sec" \
  --lidar-topic "$lidar_topic" \
  --lidar-type "$lidar_type" \
  --imu-topic "$imu_topic" \
  --imu-type "$imu_type" \
  >"$run_dir/logs/lidar.log" 2>&1 &
lidar_pid=$!

vision_pid=""
if [[ "$vision_mode" == teleimager ]]; then
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
fi

set +e
wait "$motive_pid"; motive_status=$?
wait "$lowstate_pid"; lowstate_status=$?
wait "$relay_pid"; relay_status=$?
wait "$lidar_pid"; lidar_status=$?
vision_status=0
if [[ -n "$vision_pid" ]]; then
  wait "$vision_pid"; vision_status=$?
fi
set -e

if [[ "$vision_mode" == teleimager ]]; then
  docker cp "holosoma:$vision_stage/capture" "$run_dir/vision/" >"$run_dir/logs/vision_copy.log" 2>&1 || {
    echo "Failed to copy RGB-D capture; see logs/vision_copy.log" >&2
    vision_status=1
  }
fi

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

set +e
python3 "$script_dir/validate_g1_motive_dataset.py" --run-dir "$run_dir" --output "$run_dir/validation.json"
validation_status=$?
set -e
python3 - "$run_dir/manifest.json" "$run_dir/validation.json" "$run_dir/metrics.json" "$run_dir/summary.md" <<'PY'
import json
from pathlib import Path
import shutil
import sys
import time

manifest_path, validation_path, metrics_path, summary_path = map(Path, sys.argv[1:])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
validation = json.loads(validation_path.read_text(encoding="utf-8"))
manifest["status"] = "complete" if validation["status"] == "pass" else "incomplete"
manifest["completed_realtime_ns"] = time.time_ns()
manifest["validation"] = {"path": "validation.json", "status": validation["status"]}
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
shutil.copyfile(validation_path, metrics_path)

checks = validation["checks"]
motive = checks["motive"]
lidar = checks["lidar_and_livox_imu"]
lowstate = checks["lowstate_and_pelvis_imu"]
summary_path.write_text(
    "# Did G1-4123 record synchronized passive localization inputs?\n\n"
    "## Question\n\n"
    "Can Oslo simultaneously record Motive, the native MID-360 LiDAR and IMU, "
    "and G1 pelvis lowstate without creating an actuation publisher?\n\n"
    "## Result signature\n\n"
    f"Dataset validation: **{validation['status']}**. Motive frames: "
    f"{motive.get('frames_written', 0)} with tracking coverage "
    f"{motive.get('tracking_coverage', 0.0):.3f}. Native sensor counts: "
    f"{lidar.get('message_counts', {})}. Lowstate packets: "
    f"{lowstate.get('accepted', 0)} at {lowstate.get('accepted_rate_hz', 0.0):.1f} Hz, "
    f"with {lowstate.get('sequence_gaps', 0)} sequence gaps.\n\n"
    "## Decision change\n\n"
    + (
        "The passive G1-4123 recorder is admitted for longer calibration and walking captures.\n\n"
        if validation["status"] == "pass"
        else "Do not use this run for localization scoring; repair the failed checks and rerun.\n\n"
    )
    + "## Visual artifact\n\n"
    "No figure is required for this stationary transport canary; raw streams, logs, and "
    "machine-readable validation are the decision evidence.\n\n"
    "## Boundary\n\n"
    "This is passive capture evidence only and provides no hardware-actuation clearance.\n",
    encoding="utf-8",
)
PY
if (( validation_status != 0 )); then
  exit "$validation_status"
fi
echo "PASS: $run_dir"
