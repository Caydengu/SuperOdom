#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: run_g1_kiss_live_verification.sh --run-dir PATH --network-interface IFACE [options]

Record a bounded G1-4123 localization run with raw Livox LiDAR/IMU, raw
LowState/pelvis IMU, and live localization outputs. Motive may be required as
independent evaluator truth or explicitly disabled; it is never an online
localization input. The launcher creates no robot command publisher.

Options:
  --duration-sec N             default: 120; range: 30..600
  --capture-class CLASS        stationary|amo-walk|amo-stress; default: stationary
  --robot-host IP              default: 192.168.123.164
  --robot-user USER            default: unitree
  --robot-dds-interface IFACE  default: eth0
  --robot-python PATH          default: G1-4123 egonav-deploy Python
  --expected-robot-machine-id-sha256 HEX
                               reject a different physical robot at the same IP
  --motive-server IP           default: 172.24.68.77
  --motive-mode MODE           required|disabled; default: required
  --rigid-body-id ID           default: 42
  --rigid-body-name NAME       default: G1_PELVIS_F_4123
  --ros-domain-id ID           default: 0
  --root-state-port PORT       default: 5575 (tcp://127.0.0.1:5575)
  --integrated-robot-vlm       require Gio UI initialization + map lane + real backend
  --robot-vlm-repo DIR         prepared robot-vlm checkout (required when integrated)
  --glb PATH                   exact Polycam GLB (required when integrated)
  --glb-sha256 HEX             exact GLB digest
  --surface-cache PATH         deterministic mesh target NPZ
  --surface-sha256 HEX         target content digest
  --structural-map PATH        structural localization NPZ
  --structural-map-sha256 HEX  exact localization/map-observation identity
  --map-artifact PATH          robot-vlm raster/occupancy/map observation NPZ
  --map-artifact-sha256 HEX    exact map observation artifact digest
  --motive-map-transform PATH  accepted independent Motive→Polycam evaluator JSON
  --map-key KEY                default: map_xy_structural_2cm
  --map-correction-port PORT   default: 5577
  --ui-port PORT               default: 8082
  --initialization-timeout-sec N default: 300
  --dry-run
EOF
}

original_args=("$@")
run_dir=""
network_interface=""
duration_sec=120
capture_class=stationary
robot_host=192.168.123.164
robot_user=unitree
robot_dds_interface=eth0
robot_python=/home/unitree/miniforge3/envs/egonav-deploy/bin/python
expected_robot_machine_id_sha256=""
robot_localization_root=/home/unitree/geo-179/G1_localization
motive_server=172.24.68.77
motive_mode=required
rigid_body_id=42
rigid_body_name=G1_PELVIS_F_4123
ros_domain_id=0
root_state_port=5575
live_lowstate_port=5589
record_lowstate_port=5590
integrated_robot_vlm=false
robot_vlm_repo=""
glb=""
glb_sha256=""
surface_cache=""
surface_sha256=""
structural_map=""
structural_map_sha256=""
map_artifact=""
map_artifact_sha256=""
motive_map_transform=""
map_key=map_xy_structural_2cm
map_correction_port=5577
ui_port=8082
initialization_timeout_sec=300
dry_run=false

while (( $# )); do
  case "$1" in
    --run-dir) run_dir=$2; shift 2 ;;
    --network-interface) network_interface=$2; shift 2 ;;
    --duration-sec) duration_sec=$2; shift 2 ;;
    --capture-class) capture_class=$2; shift 2 ;;
    --robot-host) robot_host=$2; shift 2 ;;
    --robot-user) robot_user=$2; shift 2 ;;
    --robot-dds-interface) robot_dds_interface=$2; shift 2 ;;
    --robot-python) robot_python=$2; shift 2 ;;
    --expected-robot-machine-id-sha256) expected_robot_machine_id_sha256=$2; shift 2 ;;
    --motive-server) motive_server=$2; shift 2 ;;
    --motive-mode) motive_mode=$2; shift 2 ;;
    --rigid-body-id) rigid_body_id=$2; shift 2 ;;
    --rigid-body-name) rigid_body_name=$2; shift 2 ;;
    --ros-domain-id) ros_domain_id=$2; shift 2 ;;
    --root-state-port) root_state_port=$2; shift 2 ;;
    --integrated-robot-vlm) integrated_robot_vlm=true; shift ;;
    --robot-vlm-repo) robot_vlm_repo=$2; shift 2 ;;
    --glb) glb=$2; shift 2 ;;
    --glb-sha256) glb_sha256=$2; shift 2 ;;
    --surface-cache) surface_cache=$2; shift 2 ;;
    --surface-sha256) surface_sha256=$2; shift 2 ;;
    --structural-map) structural_map=$2; shift 2 ;;
    --structural-map-sha256) structural_map_sha256=$2; shift 2 ;;
    --map-artifact) map_artifact=$2; shift 2 ;;
    --map-artifact-sha256) map_artifact_sha256=$2; shift 2 ;;
    --motive-map-transform) motive_map_transform=$2; shift 2 ;;
    --map-key) map_key=$2; shift 2 ;;
    --map-correction-port) map_correction_port=$2; shift 2 ;;
    --ui-port) ui_port=$2; shift 2 ;;
    --initialization-timeout-sec) initialization_timeout_sec=$2; shift 2 ;;
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
[[ "$capture_class" == stationary || "$capture_class" == amo-walk || "$capture_class" == amo-stress ]] || { echo "invalid capture class" >&2; exit 2; }
[[ "$motive_mode" == required || "$motive_mode" == disabled ]] || { echo "invalid Motive mode: $motive_mode" >&2; exit 2; }
[[ "$rigid_body_id" =~ ^[0-9]+$ ]] || { echo "invalid rigid-body ID" >&2; exit 2; }
[[ "$root_state_port" =~ ^[0-9]+$ ]] && (( 10#$root_state_port > 0 && 10#$root_state_port <= 65535 )) || {
  echo "invalid root-state port" >&2
  exit 2
}
[[ "$root_state_port" != "$live_lowstate_port" && "$root_state_port" != "$record_lowstate_port" ]] || {
  echo "root-state and LowState ports must differ" >&2
  exit 2
}
[[ "$map_correction_port" =~ ^[0-9]+$ ]] && (( 10#$map_correction_port > 0 && 10#$map_correction_port <= 65535 )) || { echo "invalid map-correction port" >&2; exit 2; }
[[ "$ui_port" =~ ^[0-9]+$ ]] && (( 10#$ui_port > 0 && 10#$ui_port <= 65535 )) || { echo "invalid UI port" >&2; exit 2; }
[[ "$root_state_port" != "$map_correction_port" && "$map_correction_port" != "$live_lowstate_port" && "$map_correction_port" != "$record_lowstate_port" ]] || { echo "all localization and LowState ports must differ" >&2; exit 2; }
[[ "$initialization_timeout_sec" =~ ^[0-9]+$ ]] && (( 10#$initialization_timeout_sec >= 30 && 10#$initialization_timeout_sec <= 900 )) || { echo "initialization timeout must be 30..900 seconds" >&2; exit 2; }
[[ "$map_key" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "invalid map key" >&2; exit 2; }
for value in "$network_interface" "$robot_dds_interface" "$robot_user" "$rigid_body_name"; do
  [[ "$value" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "unsafe identifier: $value" >&2; exit 2; }
done
[[ "$robot_python" =~ ^/[A-Za-z0-9._/-]+$ ]] || { echo "invalid robot Python path" >&2; exit 2; }
if [[ -n "$expected_robot_machine_id_sha256" ]]; then
  [[ "$expected_robot_machine_id_sha256" =~ ^[0-9a-fA-F]{64}$ ]] || {
    echo "expected robot machine-ID digest must contain 64 hexadecimal characters" >&2
    exit 2
  }
  expected_robot_machine_id_sha256="${expected_robot_machine_id_sha256,,}"
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "$script_dir/.." && pwd -P)"
if [[ "$integrated_robot_vlm" == true ]]; then
  command -v docker >/dev/null || { echo "missing command: docker" >&2; exit 2; }
  image="${G1_LOCALIZATION_IMAGE:-tml/g1-kiss-localization:1.5.2-ui-init-humble}"
else
  image="${G1_LOCALIZATION_IMAGE:-tml/g1-kiss-localization:1.4.0-humble}"
fi
export G1_LOCALIZATION_IMAGE="$image"
if [[ "$integrated_robot_vlm" == true ]]; then
  [[ -n "$robot_vlm_repo" && -n "$glb" && -n "$surface_cache" && -n "$structural_map" && -n "$map_artifact" ]] || { echo "integrated robot-vlm mode requires all repo/map paths" >&2; exit 2; }
  if [[ "$motive_mode" == required && -z "$motive_map_transform" ]]; then
    echo "required Motive evaluation needs --motive-map-transform" >&2
    exit 2
  fi
  for digest in "$glb_sha256" "$surface_sha256" "$structural_map_sha256" "$map_artifact_sha256"; do
    [[ "$digest" =~ ^[0-9a-fA-F]{64}$ ]] || { echo "integrated map digests must contain 64 hexadecimal characters" >&2; exit 2; }
  done
  robot_vlm_repo="$(realpath "$robot_vlm_repo")"
  glb="$(realpath "$glb")"
  surface_cache="$(realpath "$surface_cache")"
  structural_map="$(realpath "$structural_map")"
  map_artifact="$(realpath "$map_artifact")"
  if [[ "$motive_mode" == required ]]; then
    motive_map_transform="$(realpath "$motive_map_transform")"
  fi
  [[ -x "$robot_vlm_repo/deploy/g1/localize/run_live_humble.sh" ]] || { echo "prepared Humble UI launcher is missing" >&2; exit 2; }
  [[ -f "$robot_vlm_repo/scripts/probe_real_backend_localization.py" ]] || { echo "real-backend localization probe is missing" >&2; exit 2; }
  for path in "$glb" "$surface_cache" "$structural_map" "$map_artifact"; do [[ -f "$path" ]] || { echo "missing integrated input: $path" >&2; exit 2; }; done
  if [[ "$motive_mode" == required && ! -f "$motive_map_transform" ]]; then
    echo "missing integrated Motive evaluator input: $motive_map_transform" >&2
    exit 2
  fi
  [[ "$(sha256sum "$glb" | awk '{print $1}')" == "${glb_sha256,,}" ]] || { echo "GLB digest mismatch" >&2; exit 4; }
  [[ "$(sha256sum "$structural_map" | awk '{print $1}')" == "${structural_map_sha256,,}" ]] || { echo "structural-map digest mismatch" >&2; exit 4; }
  [[ "$(sha256sum "$map_artifact" | awk '{print $1}')" == "${map_artifact_sha256,,}" ]] || { echo "robot-vlm map-artifact digest mismatch" >&2; exit 4; }
  integrated_identity_text="$(docker run --rm --network none \
    --user "$(id -u):$(id -g)" \
    --entrypoint python3 \
    --volume "$surface_cache:/inputs/surface.npz:ro" \
    --volume "$structural_map:/inputs/structural.npz:ro" \
    --volume "$map_artifact:/inputs/map-artifact.npz:ro" \
    "$image" -c '
import numpy as np
with np.load("/inputs/surface.npz", allow_pickle=False) as value:
    print(str(np.asarray(value["source_sha256"]).item()))
    print(str(np.asarray(value["surface_sha256"]).item()))
with np.load("/inputs/structural.npz", allow_pickle=False) as value:
    key = "'"$map_key"'"
    if key not in value or value[key].ndim != 2 or value[key].shape[1] != 2:
        raise SystemExit(f"invalid structural map key: {key}")
    print(key)
with np.load("/inputs/map-artifact.npz", allow_pickle=False) as value:
    print(str(np.asarray(value["source_glb_sha256"]).item()))
    print(str(np.asarray(value["surface_sha256"]).item()))
    print(str(np.asarray(value["scan_hash"]).item()))
')"
  readarray -t integrated_identity <<<"$integrated_identity_text"
  [[ "${#integrated_identity[@]}" -eq 6 ]] || { echo "could not read integrated map identities" >&2; exit 4; }
  [[ "${integrated_identity[0]}" == "${glb_sha256,,}" && "${integrated_identity[1]}" == "${surface_sha256,,}" ]] || { echo "surface target identity mismatch" >&2; exit 4; }
  [[ "${integrated_identity[2]}" == "$map_key" ]] || { echo "structural map key mismatch" >&2; exit 4; }
  [[ "${integrated_identity[3]}" == "${glb_sha256,,}" && "${integrated_identity[4]}" == "${surface_sha256,,}" && "${integrated_identity[5]}" == "${structural_map_sha256,,}" ]] || { echo "robot-vlm map artifact identity mismatch" >&2; exit 4; }
  if [[ "$motive_mode" == required ]]; then
    python3 "$script_dir/validate_motive_polycam_transform.py" \
      --transform "$motive_map_transform" \
      --glb-sha256 "${glb_sha256,,}" \
      --surface-sha256 "${surface_sha256,,}" \
      --structural-map-sha256 "${structural_map_sha256,,}" \
      --pelvis-rigid-body-id "$rigid_body_id" \
      --pelvis-rigid-body-name "$rigid_body_name" \
      >/dev/null
  fi
fi
route_to_robot="$(ip route get "$robot_host" | head -1)"
[[ "$route_to_robot" != *" via "* ]] || { echo "robot route is not direct: $route_to_robot" >&2; exit 4; }
[[ "$route_to_robot" == *" dev $network_interface "* ]] || { echo "wrong robot NIC: $route_to_robot" >&2; exit 4; }
offboard_robot_address="$(awk '{for(i=1;i<=NF;i++) if($i=="src") print $(i+1)}' <<<"$route_to_robot")"
if [[ "$motive_mode" == required ]]; then
  motive_route="$(ip route get "$motive_server" | head -1)"
  motive_client_address="$(awk '{for(i=1;i<=NF;i++) if($i=="src") print $(i+1)}' <<<"$motive_route")"
else
  motive_client_address=disabled
fi
[[ "$offboard_robot_address" == 192.168.123.* ]] || {
  echo "could not resolve the G1 route" >&2
  exit 4
}
if [[ "$motive_mode" == required && -z "$motive_client_address" ]]; then
  echo "could not resolve the Motive route" >&2
  exit 4
fi

if [[ "$dry_run" == true ]]; then
  cat <<EOF
schema=g1_kiss_live_verification_dry_run_v1
run_dir=$run_dir
duration_sec=$duration_sec
capture_class=$capture_class
robot=$robot_user@$robot_host
robot_interface=$network_interface
expected_robot_machine_id_sha256=$expected_robot_machine_id_sha256
motive_mode=$motive_mode
motive=$motive_server client=$motive_client_address
rigid_body=$rigid_body_name id=$rigid_body_id
image=$image
topics=/utlidar/cloud_livox_mid360,/utlidar/imu_livox_mid360,/g1/localization/pelvis_odom,/g1/localization/cloud_registered
capture_readiness=first fresh and fully healthy HSROOT02 packet; robot must remain stationary until READY is printed
root_state_endpoint=tcp://127.0.0.1:$root_state_port
integrated_robot_vlm=$integrated_robot_vlm
robot_vlm_repo=$robot_vlm_repo
glb=$glb
glb_sha256=$glb_sha256
surface_sha256=$surface_sha256
structural_map=$structural_map
structural_map_sha256=$structural_map_sha256
map_artifact=$map_artifact
map_artifact_sha256=$map_artifact_sha256
motive_map_transform=$motive_map_transform
map_correction_endpoint=tcp://127.0.0.1:$map_correction_port
viser_url=http://localhost:$ui_port
command_capability=structurally_unavailable
EOF
  exit 0
fi

for command in docker git ip python3 scp sha256sum ssh; do
  command -v "$command" >/dev/null || { echo "missing command: $command" >&2; exit 2; }
done
[[ ! -e "$run_dir" ]] || { echo "refusing to overwrite $run_dir" >&2; exit 2; }
mkdir -p "$run_dir"/{logs,motive,lowstate,rosbag,runtime,traces}
run_dir="$(cd "$run_dir" && pwd -P)"
if [[ "$integrated_robot_vlm" == true && "$motive_mode" == required ]]; then
  cp -- "$motive_map_transform" "$run_dir/runtime/motive-to-polycam.json"
fi
trial_id="$(basename "$run_dir")"
remote_stage="/tmp/g1-kiss-verification-${trial_id//[^A-Za-z0-9_-]/_}"
source_commit="$(git -C "$repo_root" rev-parse HEAD)"
source_dirty_digest="$({ git -C "$repo_root" status --porcelain=v1 --untracked-files=all; } | sha256sum | awk '{print $1}')"
image_id="$(docker image inspect "$image" --format '{{.Id}}')"
integrated_python=False
[[ "$integrated_robot_vlm" == true ]] && integrated_python=True

python3 - "$run_dir/manifest.json" <<EOF
import json, pathlib, time
pathlib.Path(r"$run_dir/manifest.json").write_text(json.dumps({
  "schema": "g1_kiss_live_verification_manifest_v1",
  "status": "recording",
  "created_realtime_ns": time.time_ns(),
  "duration_sec": $duration_sec,
  "capture_class": "$capture_class",
  "robot_identity": "G1-4123",
  "robot_host": "$robot_host",
  "robot_network_interface": "$network_interface",
  "robot_python": "$robot_python",
  "expected_robot_machine_id_sha256": "$expected_robot_machine_id_sha256",
  "motive_server": "$motive_server",
  "motive_client_address": "$motive_client_address",
  "rigid_body_id": $rigid_body_id,
  "rigid_body_name": "$rigid_body_name",
  "source_commit": "$source_commit",
  "source_dirty_digest": "$source_dirty_digest",
  "container_image": "$image",
  "container_image_id": "$image_id",
  "motive_mode": "$motive_mode",
  "motive_role": "$(if [[ "$motive_mode" == required ]]; then echo evaluator_only; else echo disabled; fi)",
  "command_capability": "structurally_unavailable",
  "actuation_publishers_created": 0,
  "integrated_robot_vlm": $integrated_python,
  "glb_sha256": "${glb_sha256,,}",
  "surface_sha256": "${surface_sha256,,}",
  "structural_map_sha256": "${structural_map_sha256,,}",
  "map_artifact_sha256": "${map_artifact_sha256,,}",
  "motive_map_transform_sha256": "$(if [[ -n "$motive_map_transform" ]]; then sha256sum "$motive_map_transform" | awk '{print $1}'; fi)",
  "map_key": "$map_key",
  "map_correction_port": $map_correction_port,
  "ui_port": $ui_port,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
EOF
printf '%q ' "$script_dir/run_g1_kiss_live_verification.sh" "${original_args[@]}" >"$run_dir/command.txt"
printf '\n' >>"$run_dir/command.txt"

ssh_options=(
  -o BatchMode=yes
  -o ConnectTimeout=5
  -o ServerAliveInterval=2
  -o ServerAliveCountMax=2
)
robot_preflight_log="$run_dir/logs/robot_preflight.log"
mark_preflight_failed() {
  local stage=$1
  local command_status=$2
  python3 - "$run_dir/manifest.json" "$stage" "$command_status" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
value = json.loads(path.read_text(encoding="utf-8"))
value["status"] = "preflight_failed"
value["failure_stage"] = sys.argv[2]
value["failure_command_status"] = int(sys.argv[3])
path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  echo "G1 preflight failed at '$stage' (status $command_status)." >&2
  echo "Inspect $robot_preflight_log" >&2
  if [[ -s "$robot_preflight_log" ]]; then
    tail -40 "$robot_preflight_log" >&2
  fi
  exit 4
}

mark_runtime_failed() {
  local stage=$1
  local detail=$2
  local status=${3:-implementation_failed}
  python3 - "$run_dir/manifest.json" "$stage" "$detail" "$status" <<'PY'
import json
import pathlib
import sys
import time

path = pathlib.Path(sys.argv[1])
value = json.loads(path.read_text(encoding="utf-8"))
value["status"] = sys.argv[4]
value["failure_stage"] = sys.argv[2]
value["failure_detail"] = sys.argv[3]
value["failed_realtime_ns"] = time.time_ns()
path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

echo "[1/5] Verifying SSH access to $robot_user@$robot_host..."
if ssh "${ssh_options[@]}" "$robot_user@$robot_host" true >>"$robot_preflight_log" 2>&1; then
  echo "[1/5] SSH access verified."
else
  command_status=$?
  mark_preflight_failed ssh_access "$command_status"
fi

echo "[2/5] Binding the physical robot fingerprint..."
if machine_id_output="$(ssh "${ssh_options[@]}" "$robot_user@$robot_host" \
  "sha256sum /etc/machine-id" 2>>"$robot_preflight_log")"; then
  actual_robot_machine_id_sha256="$(awk '{print $1}' <<<"$machine_id_output")"
else
  command_status=$?
  mark_preflight_failed robot_machine_id_probe "$command_status"
fi
if [[ ! "$actual_robot_machine_id_sha256" =~ ^[0-9a-f]{64}$ ]]; then
  mark_preflight_failed robot_machine_id_invalid 4
fi
printf 'robot_machine_id_sha256=%s\n' "$actual_robot_machine_id_sha256" >>"$robot_preflight_log"
if [[ -n "$expected_robot_machine_id_sha256" && \
      "$actual_robot_machine_id_sha256" != "$expected_robot_machine_id_sha256" ]]; then
  mark_preflight_failed wrong_robot_identity 4
fi
python3 - "$run_dir/manifest.json" "$actual_robot_machine_id_sha256" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
value = json.loads(path.read_text(encoding="utf-8"))
value["robot_machine_id_sha256"] = sys.argv[2]
path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
echo "[2/5] Physical robot fingerprint accepted: ${actual_robot_machine_id_sha256:0:12}..."

echo "[3/5] Verifying onboard Python: $robot_python"
if ssh "${ssh_options[@]}" "$robot_user@$robot_host" "test -x '$robot_python'" >>"$robot_preflight_log" 2>&1; then
  echo "[3/5] Onboard Python verified."
else
  command_status=$?
  mark_preflight_failed robot_python "$command_status"
fi

echo "[4/5] Verifying passive LowState SDK imports..."
if ssh "${ssh_options[@]}" "$robot_user@$robot_host" \
  "'$robot_python' -c 'from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber; from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_'" \
  >>"$robot_preflight_log" 2>&1; then
  echo "[4/5] Passive LowState SDK imports verified."
else
  command_status=$?
  mark_preflight_failed unitree_sdk_import "$command_status"
fi

echo "[5/5] Staging the command-incapable LowState relay..."
if ssh "${ssh_options[@]}" "$robot_user@$robot_host" "mkdir -p '$remote_stage'" >>"$robot_preflight_log" 2>&1; then
  :
else
  command_status=$?
  mark_preflight_failed remote_stage_create "$command_status"
fi
if scp -q "${ssh_options[@]}" -r "$repo_root/g1_root_state_bridge/g1_root_state_bridge" \
  "$robot_user@$robot_host:$remote_stage/" >>"$robot_preflight_log" 2>&1; then
  echo "[5/5] Passive relay staged."
else
  command_status=$?
  mark_preflight_failed remote_stage_copy "$command_status"
fi
if [[ "$motive_mode" == required ]]; then
  ssh -o BatchMode=yes "$robot_user@$robot_host" \
    "test -d '$robot_localization_root/mocap_utils'"
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
fi
if ! python3 "$script_dir/probe_g1_clock.py" \
  --target "$robot_user@$robot_host" \
  --timeout-sec 5 \
  --output "$run_dir/clock_probe.json" \
  >"$run_dir/logs/clock_probe.log" 2>&1; then
  mark_runtime_failed clock_probe "bounded read-only SSH clock probe failed" infrastructure_failed
  echo "G1 clock probe failed; inspect $run_dir/logs/clock_probe.log" >&2
  tail -40 "$run_dir/logs/clock_probe.log" >&2
  exit 4
fi

live_duration=$((10#$duration_sec + 15))
if [[ "$integrated_robot_vlm" == true ]]; then
  live_duration=$((10#$duration_sec + 10#$initialization_timeout_sec + 45))
fi
"$script_dir/run_g1_kiss_live_stack.sh" \
  --network-interface "$network_interface" \
  --duration-sec "$live_duration" \
  --robot-host "$robot_host" \
  --robot-user "$robot_user" \
  --robot-dds-interface "$robot_dds_interface" \
  --robot-python "$robot_python" \
  --ros-domain-id "$ros_domain_id" \
  --lowstate-port "$live_lowstate_port" \
  --root-state-port "$root_state_port" \
  >"$run_dir/logs/live_stack.log" 2>&1 &
live_pid=$!

cleanup() {
  for pid in "${probe_pid:-}" "${bag_pid:-}" "${relay_pid:-}" "${lowstate_pid:-}" "${motive_pid:-}" "${map_pid:-}" "${ui_pid:-}" "${live_pid:-}"; do
    if [[ -n "$pid" ]]; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  for pid in "${probe_pid:-}" "${bag_pid:-}" "${relay_pid:-}" "${lowstate_pid:-}" "${motive_pid:-}" "${map_pid:-}" "${ui_pid:-}" "${live_pid:-}"; do
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
local_source_epoch="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_epoch"])' "$run_dir/runtime/readiness-root-state.json")"
calibration_digest="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["calibration_digest"])' "$run_dir/runtime/readiness-root-state.json")"

if [[ "$integrated_robot_vlm" == true ]]; then
  initialization_receipt="$run_dir/runtime/ui-initialization.json"
  "$script_dir/run_g1_structural_map_shadow.sh" \
    --network-interface "$network_interface" \
    --map "$structural_map" \
    --map-sha256 "${structural_map_sha256,,}" \
    --map-key "$map_key" \
    --duration-sec "$live_duration" \
    --ros-domain-id "$ros_domain_id" \
    --root-state-port "$root_state_port" \
    --map-correction-port "$map_correction_port" \
    --initialization-mode ui \
    --ui-initialization-receipt "$initialization_receipt" \
    --expected-glb-sha256 "${glb_sha256,,}" \
    --expected-surface-sha256 "${surface_sha256,,}" \
    >"$run_dir/logs/map_stack.log" 2>&1 &
  map_pid=$!
  "$robot_vlm_repo/deploy/g1/localize/run_live_humble.sh" \
    --network-interface "$network_interface" \
    --glb "$glb" \
    --glb-sha256 "${glb_sha256,,}" \
    --surface-cache "$surface_cache" \
    --surface-sha256 "${surface_sha256,,}" \
    --structural-map-sha256 "${structural_map_sha256,,}" \
    --initialization-receipt "$initialization_receipt" \
    --port "$ui_port" \
    --duration-sec "$live_duration" \
    --ros-domain-id "$ros_domain_id" \
    >"$run_dir/logs/ui.log" 2>&1 &
  ui_pid=$!
  sleep 3
  if ! kill -0 "$map_pid" 2>/dev/null; then
    mark_runtime_failed structural_map_startup "map lane exited before UI initialization"
    echo "map lane exited during startup; inspect $run_dir/logs/map_stack.log" >&2
    exit 5
  fi
  if ! kill -0 "$ui_pid" 2>/dev/null; then
    mark_runtime_failed robot_vlm_ui_startup "Gio UI exited before pose initialization"
    echo "Gio UI exited during startup; inspect $run_dir/logs/ui.log" >&2
    exit 5
  fi
  cat <<EOF
LOCAL ODOMETRY READY. Keep G1-4123 stationary.
Open http://localhost:$ui_port, choose "Set initial pose (2 clicks)", then click:
  1) the G1 position on the exact 2026-08-24 Polycam mesh
  2) a point in the direction the G1 faces
The pin-locked Snap must pass the receipt gates before this launcher continues.
EOF
  if ! docker run --rm --user "$(id -u):$(id -g)" \
      --network host \
      --cap-drop ALL \
      --security-opt no-new-privileges \
      --read-only \
      --tmpfs /tmp:rw,noexec,nosuid,size=32m \
      --volume "$run_dir/runtime:/output:rw" \
      "$image" \
      g1-wait-map-correction \
      --endpoint "tcp://127.0.0.1:$map_correction_port" \
      --map-sha256 "${structural_map_sha256,,}" \
      --local-source-epoch "$local_source_epoch" \
      --timeout-sec "$initialization_timeout_sec" \
      --output /output/map-readiness.json \
      >"$run_dir/logs/map-readiness.log" 2>&1; then
    mark_runtime_failed map_initialization \
      "no accepted digest-bound map correction arrived before the UI timeout"
    echo "map initialization failed; inspect $run_dir/logs/map-readiness.log and map_stack.log" >&2
    exit 5
  fi
  echo "MAP INITIALIZATION ACCEPTED; robot-vlm now has a map-frame base pose."
fi

if [[ "$capture_class" == stationary ]]; then
  echo "STATIONARY QUALIFICATION RECORDING STARTED: keep G1-4123 still for ${duration_sec}s"
else
  echo "READY FOR OPERATOR-CONTROLLED AMO: recording ${duration_sec}s now"
fi

cyclonedds_uri="<CycloneDDS><Domain id=\"any\"><General><Interfaces><NetworkInterface name=\"$network_interface\" /></Interfaces></General></Domain></CycloneDDS>"

motive_status=0
if [[ "$motive_mode" == required ]]; then
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
fi

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

probe_status=0
if [[ "$integrated_robot_vlm" == true ]]; then
  (
    cd "$robot_vlm_repo"
    export ROBOT_VLM_ROOT_STATE_ENDPOINT="tcp://127.0.0.1:$root_state_port"
    export ROBOT_VLM_MAP_CORRECTION_ENDPOINT="tcp://127.0.0.1:$map_correction_port"
    export ROBOT_VLM_STRUCTURAL_MAP_SHA256="${structural_map_sha256,,}"
    export ROBOT_VLM_LOCALIZATION_CALIBRATION_SHA256="$calibration_digest"
    export ROBOT_VLM_REQUIRE_INERTIAL_HEADING=1
    export ROBOT_VLM_MAP_POSITION_POLICY=initialize_once_heading_only
    export ROBOT_VLM_MAP_ARTIFACT="$map_artifact"
    PYTHONPATH=src uv run --frozen python scripts/probe_real_backend_localization.py \
      --map-artifact "$map_artifact" \
      --expected-artifact-sha256 "${map_artifact_sha256,,}" \
      --duration-sec "$duration_sec" \
      --sample-hz 50 \
      --writer-lock "$run_dir/runtime/robot-vlm-observer.lock" \
      --output "$run_dir/robot-vlm-real-backend.json" \
      --trace "$run_dir/traces/robot-vlm-base-pose.jsonl"
  ) >"$run_dir/logs/robot-vlm-real-backend.log" 2>&1 &
  probe_pid=$!
fi

set +e
if [[ "$motive_mode" == required ]]; then
  wait "$motive_pid"; motive_status=$?
fi
wait "$lowstate_pid"; lowstate_status=$?
wait "$relay_pid"; relay_status=$?
wait "$bag_pid"; bag_status=$?
if [[ "$integrated_robot_vlm" == true ]]; then
  wait "$probe_pid"; probe_status=$?; probe_pid=""
  kill -TERM "$map_pid" "$ui_pid" "$live_pid" 2>/dev/null || true
  wait "$map_pid"; map_status=$?; map_pid=""
  wait "$ui_pid"; ui_status=$?; ui_pid=""
  wait "$live_pid"; live_status=$?; live_pid=""
  python3 - "$run_dir/service_status.json" "$map_status" "$ui_status" "$live_status" <<'PY'
import json, pathlib, sys
pathlib.Path(sys.argv[1]).write_text(json.dumps({
    "map_stack_after_requested_stop": int(sys.argv[2]),
    "ui_after_requested_stop": int(sys.argv[3]),
    "live_stack_after_requested_stop": int(sys.argv[4]),
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
else
  wait "$live_pid"; live_status=$?; live_pid=""
fi
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
  "robot_vlm_real_backend": $probe_status,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
EOF
python3 "$script_dir/validate_g1_kiss_live_verification.py" \
  --run-dir "$run_dir" \
  --output "$run_dir/validation.json"
if [[ "$integrated_robot_vlm" == true && "$motive_mode" == required ]]; then
  docker run --rm --entrypoint /bin/bash \
    --user "$(id -u):$(id -g)" \
    --network none \
    --cap-drop ALL \
    --security-opt no-new-privileges \
    --read-only \
    --tmpfs /tmp:rw,nosuid,size=256m \
    --env HOME=/tmp/home \
    --volume "$repo_root:/repo:ro" \
    --volume "$run_dir:/run:rw" \
    --volume "$map_artifact:/inputs/map-artifact.npz:ro" \
    "$image" -lc \
    'python3 /repo/scripts/score_robot_vlm_map_localization.py \
      --run-dir /run \
      --motive-map-transform /run/runtime/motive-to-polycam.json \
      --map-artifact /inputs/map-artifact.npz \
      --map-artifact-sha256 "'"${map_artifact_sha256,,}"'" \
      --output /run/robot-vlm-motive-map-score.json \
      --enriched-trace /run/traces/robot-vlm-motive-map-score.jsonl' \
    >"$run_dir/logs/robot-vlm-motive-map-score.log" 2>&1
fi
python3 - "$run_dir/manifest.json" <<'PY'
import json, pathlib, sys, time
path = pathlib.Path(sys.argv[1])
value = json.loads(path.read_text(encoding="utf-8"))
value["status"] = "complete"
value["completed_realtime_ns"] = time.time_ns()
path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
echo "PASS: $run_dir"
