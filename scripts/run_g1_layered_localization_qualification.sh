#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: run_g1_layered_localization_qualification.sh --run-dir DIR --network-interface IFACE --map MAP.npz --map-sha256 HEX --robot-vlm-repo DIR [options]

Run the complete stationary, subscriber-only G1-4123 localization gate: the
fast HSROOT02 lane, automatic RVMAP001 lane, readiness checks, and the actual
robot-vlm consumer with --expected-map present. No Motive or policy is needed.

Options:
  --run-dir DIR               new artifact directory (required)
  --network-interface IFACE   offboard G1 NIC (required)
  --map PATH                  processed Polycam structural NPZ (required)
  --map-sha256 HEX            exact SHA-256 of PATH (required)
  --robot-vlm-repo DIR        checkout containing the qualified probe (required)
  --duration-sec N            producer lifetime, 45..3600; default: 75
  --probe-duration-sec N      robot-vlm observation, 5..300; default: 30
  --robot-host IP             default: 192.168.123.164
  --robot-user USER           default: unitree
  --robot-dds-interface IFACE default: eth0
  --ros-domain-id ID          default: 0
  --lowstate-port PORT        default: 5589
  --root-state-port PORT      default: 5581
  --map-correction-port PORT  default: 5577
  --dry-run                   validate inputs and print both producer commands
EOF
}

original_args=("$@")
run_dir=""
network_interface=""
map_path=""
map_sha256=""
robot_vlm_repo=""
duration_sec=75
probe_duration_sec=30
robot_host=192.168.123.164
robot_user=unitree
robot_dds_interface=eth0
ros_domain_id=0
lowstate_port=5589
root_state_port=5581
map_correction_port=5577
dry_run=false

while (( $# )); do
  case "$1" in
    --run-dir) run_dir=$2; shift 2 ;;
    --network-interface) network_interface=$2; shift 2 ;;
    --map) map_path=$2; shift 2 ;;
    --map-sha256) map_sha256=$2; shift 2 ;;
    --robot-vlm-repo) robot_vlm_repo=$2; shift 2 ;;
    --duration-sec) duration_sec=$2; shift 2 ;;
    --probe-duration-sec) probe_duration_sec=$2; shift 2 ;;
    --robot-host) robot_host=$2; shift 2 ;;
    --robot-user) robot_user=$2; shift 2 ;;
    --robot-dds-interface) robot_dds_interface=$2; shift 2 ;;
    --ros-domain-id) ros_domain_id=$2; shift 2 ;;
    --lowstate-port) lowstate_port=$2; shift 2 ;;
    --root-state-port) root_state_port=$2; shift 2 ;;
    --map-correction-port) map_correction_port=$2; shift 2 ;;
    --dry-run) dry_run=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

[[ -n "$run_dir" && -n "$network_interface" && -n "$map_path" && -n "$map_sha256" && -n "$robot_vlm_repo" ]] || { usage; exit 2; }
[[ "$network_interface" =~ ^[A-Za-z0-9_.:-]+$ ]] || { echo "invalid network interface" >&2; exit 2; }
[[ "$robot_dds_interface" =~ ^[A-Za-z0-9_.:-]+$ ]] || { echo "invalid robot DDS interface" >&2; exit 2; }
[[ "$robot_host" =~ ^[A-Za-z0-9_.:-]+$ ]] || { echo "invalid robot host" >&2; exit 2; }
[[ "$robot_user" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "invalid robot user" >&2; exit 2; }
[[ "$map_sha256" =~ ^[0-9a-fA-F]{64}$ ]] || { echo "invalid map SHA-256" >&2; exit 2; }
[[ "$duration_sec" =~ ^[0-9]+$ ]] && (( 10#$duration_sec >= 45 && 10#$duration_sec <= 3600 )) || { echo "duration must be 45..3600 seconds" >&2; exit 2; }
[[ "$probe_duration_sec" =~ ^[0-9]+$ ]] && (( 10#$probe_duration_sec >= 5 && 10#$probe_duration_sec <= 300 )) || { echo "probe duration must be 5..300 seconds" >&2; exit 2; }
[[ "$ros_domain_id" =~ ^[0-9]+$ ]] && (( 10#$ros_domain_id <= 232 )) || { echo "invalid ROS domain" >&2; exit 2; }
for port in "$lowstate_port" "$root_state_port" "$map_correction_port"; do
  [[ "$port" =~ ^[0-9]+$ ]] && (( 10#$port > 0 && 10#$port <= 65535 )) || { echo "invalid port: $port" >&2; exit 2; }
done
[[ "$lowstate_port" != "$root_state_port" && "$lowstate_port" != "$map_correction_port" && "$root_state_port" != "$map_correction_port" ]] || { echo "all ports must differ" >&2; exit 2; }

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "$script_dir/.." && pwd -P)"
map_path="$(realpath "$map_path")"
robot_vlm_repo="$(realpath "$robot_vlm_repo")"
[[ -f "$map_path" ]] || { echo "map is not a regular file: $map_path" >&2; exit 2; }
[[ -f "$robot_vlm_repo/scripts/probe_layered_localization.py" ]] || { echo "robot-vlm probe is missing" >&2; exit 2; }
observed_map_sha256="$(sha256sum "$map_path" | awk '{print $1}')"
[[ "${observed_map_sha256,,}" == "${map_sha256,,}" ]] || { echo "map SHA-256 mismatch" >&2; exit 4; }

local_command=(
  "$script_dir/run_g1_kiss_live_stack.sh"
  --network-interface "$network_interface"
  --duration-sec "$duration_sec"
  --robot-host "$robot_host"
  --robot-user "$robot_user"
  --robot-dds-interface "$robot_dds_interface"
  --ros-domain-id "$ros_domain_id"
  --lowstate-port "$lowstate_port"
  --root-state-port "$root_state_port"
)
map_command=(
  "$script_dir/run_g1_structural_map_shadow.sh"
  --network-interface "$network_interface"
  --map "$map_path"
  --map-sha256 "$observed_map_sha256"
  --duration-sec "$duration_sec"
  --ros-domain-id "$ros_domain_id"
  --root-state-port "$root_state_port"
  --map-correction-port "$map_correction_port"
)

if [[ "$dry_run" == true ]]; then
  echo "schema=g1_layered_localization_qualification_dry_run_v1"
  echo "command_capability=structurally_unavailable"
  echo "motive_role=not_required_for_transport_gate"
  printf 'local_command='; printf '%q ' "${local_command[@]}"; printf '\n'
  printf 'map_command='; printf '%q ' "${map_command[@]}"; printf '\n'
  echo "robot_vlm_probe=$robot_vlm_repo/scripts/probe_layered_localization.py"
  exit 0
fi

[[ ! -e "$run_dir" ]] || { echo "run directory already exists: $run_dir" >&2; exit 2; }
mkdir -p "$run_dir/logs" "$run_dir/runtime"
printf '%q ' "$script_dir/run_g1_layered_localization_qualification.sh" "${original_args[@]}" >"$run_dir/command.txt"
printf '\n' >>"$run_dir/command.txt"

source_commit="$(git -C "$repo_root" rev-parse HEAD)"
robot_vlm_commit="$(git -C "$robot_vlm_repo" rev-parse HEAD)"
image="${G1_LOCALIZATION_IMAGE:-tml/g1-kiss-localization:1.4.0-humble}"
image_id="$(docker image inspect "$image" --format '{{.Id}}')"
python3 - "$run_dir/manifest.json" <<PY
import json, pathlib, sys, time
path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps({
  "schema": "g1_layered_localization_live_qualification_manifest_v1",
  "status": "running",
  "created_realtime_ns": time.time_ns(),
  "robot_identity": "G1-4123",
  "duration_sec": $duration_sec,
  "probe_duration_sec": $probe_duration_sec,
  "network_interface": "$network_interface",
  "map_path": "$map_path",
  "map_digest": "$observed_map_sha256",
  "producer_commit": "$source_commit",
  "robot_vlm_commit": "$robot_vlm_commit",
  "container_image": "$image",
  "container_image_id": "$image_id",
  "root_state_port": $root_state_port,
  "map_correction_port": $map_correction_port,
  "motive_role": "not_required_for_transport_gate",
  "command_capability": "structurally_unavailable",
  "hardware_actuation_clearance": False,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

local_pid=""
map_pid=""
cleanup() {
  for pid in "$map_pid" "$local_pid"; do
    if [[ -n "$pid" ]]; then kill -TERM "$pid" 2>/dev/null || true; fi
  done
  for pid in "$map_pid" "$local_pid"; do
    if [[ -n "$pid" ]]; then wait "$pid" 2>/dev/null || true; fi
  done
}
on_exit() {
  status=$?
  cleanup
  if [[ -f "$run_dir/manifest.json" && ! -f "$run_dir/validation.json" ]]; then
    python3 - "$run_dir/manifest.json" "$status" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
value = json.loads(path.read_text(encoding="utf-8"))
value["status"] = "incomplete"
value["launcher_exit_status"] = int(sys.argv[2])
path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  fi
  exit "$status"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

"${local_command[@]}" >"$run_dir/logs/local-stack.log" 2>&1 &
local_pid=$!
"${map_command[@]}" >"$run_dir/logs/map-stack.log" 2>&1 &
map_pid=$!

PYTHONPATH="$repo_root/g1_root_state_bridge" python3 "$script_dir/wait_for_root_state.py" \
  --endpoint "tcp://127.0.0.1:$root_state_port" \
  --timeout-sec 20 \
  --output "$run_dir/runtime/root-readiness.json" \
  >"$run_dir/logs/root-readiness.log" 2>&1
local_source_epoch="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_epoch"])' "$run_dir/runtime/root-readiness.json")"
calibration_digest="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["calibration_digest"])' "$run_dir/runtime/root-readiness.json")"

docker run --rm \
  --network host \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=16m \
  "$image" \
  g1-wait-map-correction \
  --endpoint "tcp://127.0.0.1:$map_correction_port" \
  --map-sha256 "$observed_map_sha256" \
  --local-source-epoch "$local_source_epoch" \
  --timeout-sec 40 \
  --output /tmp/map-readiness.json \
  >"$run_dir/runtime/map-readiness.json" \
  2>"$run_dir/logs/map-readiness.log"

echo "BOTH LOCALIZATION LANES READY; RUNNING NON-COMMANDING ROBOT-VLM PROBE"
set +e
(
  cd "$robot_vlm_repo"
  PYTHONPATH=src uv run --frozen python scripts/probe_layered_localization.py \
    --root-endpoint "tcp://127.0.0.1:$root_state_port" \
    --map-endpoint "tcp://127.0.0.1:$map_correction_port" \
    --map-sha256 "$observed_map_sha256" \
    --calibration-sha256 "$calibration_digest" \
    --expected-map present \
    --duration-sec "$probe_duration_sec" \
    --sample-hz 20 \
    --output "$run_dir/robot-vlm-probe.json"
) >"$run_dir/logs/robot-vlm-probe.log" 2>&1
probe_status=$?
wait "$map_pid"; map_status=$?; map_pid=""
wait "$local_pid"; local_status=$?; local_pid=""
set -e

python3 - "$run_dir" "$local_status" "$map_status" "$probe_status" <<'PY'
import json, pathlib, sys, time
run = pathlib.Path(sys.argv[1])
local_status, map_status, probe_status = map(int, sys.argv[2:])
probe_path = run / "robot-vlm-probe.json"
probe = json.loads(probe_path.read_text(encoding="utf-8")) if probe_path.exists() else {}
failures = []
if local_status != 0: failures.append(f"local_process:{local_status}")
if map_status != 0: failures.append(f"map_process:{map_status}")
if probe_status != 0 or probe.get("status") != "pass": failures.append(f"robot_vlm_probe:{probe_status}")
report = {
  "schema": "g1_layered_localization_live_qualification_v1",
  "status": "pass" if not failures else "fail",
  "evidence_class": "passive stationary live two-lane localization shadow",
  "command_capability": "structurally_unavailable",
  "hardware_actuation_clearance": False,
  "completed_realtime_ns": time.time_ns(),
  "process_statuses": {"local": local_status, "map": map_status, "robot_vlm_probe": probe_status},
  "failures": failures,
  "robot_vlm": probe,
}
(run / "validation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
manifest_path = run / "manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
root = json.loads((run / "runtime/root-readiness.json").read_text(encoding="utf-8"))
manifest["status"] = report["status"]
manifest["local_source_epoch"] = root["source_epoch"]
manifest["calibration_digest"] = root["calibration_digest"]
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2, sort_keys=True))
raise SystemExit(0 if not failures else 1)
PY
