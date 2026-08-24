#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: run_motive_polycam_calibration.sh --robot-vlm-repo PATH --sdk-root PATH \
  --glb PATH --glb-sha256 HEX --surface-cache PATH --surface-sha256 HEX \
  --structural-map-sha256 HEX --rigid-body-id ID --rigid-body-name NAME \
  --output-dir PATH [options]

Launch the non-actuating Motive-pointer to Polycam-map calibration UI. The
pointer rigid body's Motive origin must first be pivot-calibrated to its
physical tip. Collect at least four widely separated fit points and one held-out
point. This tool never consumes robot localization, so the ground-truth
transform remains independent of the estimator being evaluated.

Options:
  --motive-server ADDRESS  default: 172.24.68.77
  --motive-client ADDRESS  local address used for NatNet; required unless dry-run
  --pelvis-rigid-body-id ID default: 42
  --pelvis-rigid-body-name NAME default: G1_PELVIS_F_4123
  --port PORT              Viser web port; default: 8083
  --duration-sec N         bounded launch, 60..3600; default: 1200
  --image IMAGE            default: tml/robot-vlm-localization-ui:2026-08-24-humble
  --dry-run                validate identities and print the launch only
EOF
}

robot_vlm_repo=""
sdk_root=""
glb=""
glb_sha256=""
surface_cache=""
surface_sha256=""
structural_map_sha256=""
rigid_body_id=""
rigid_body_name=""
pelvis_rigid_body_id=42
pelvis_rigid_body_name=G1_PELVIS_F_4123
output_dir=""
motive_server=172.24.68.77
motive_client=""
port=8083
duration_sec=1200
image=tml/robot-vlm-localization-ui:2026-08-24-humble
dry_run=false
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

while (( $# )); do
  case "$1" in
    --robot-vlm-repo) robot_vlm_repo=$2; shift 2 ;;
    --sdk-root) sdk_root=$2; shift 2 ;;
    --glb) glb=$2; shift 2 ;;
    --glb-sha256) glb_sha256=$2; shift 2 ;;
    --surface-cache) surface_cache=$2; shift 2 ;;
    --surface-sha256) surface_sha256=$2; shift 2 ;;
    --structural-map-sha256) structural_map_sha256=$2; shift 2 ;;
    --rigid-body-id) rigid_body_id=$2; shift 2 ;;
    --rigid-body-name) rigid_body_name=$2; shift 2 ;;
    --pelvis-rigid-body-id) pelvis_rigid_body_id=$2; shift 2 ;;
    --pelvis-rigid-body-name) pelvis_rigid_body_name=$2; shift 2 ;;
    --output-dir) output_dir=$2; shift 2 ;;
    --motive-server) motive_server=$2; shift 2 ;;
    --motive-client) motive_client=$2; shift 2 ;;
    --port) port=$2; shift 2 ;;
    --duration-sec) duration_sec=$2; shift 2 ;;
    --image) image=$2; shift 2 ;;
    --dry-run) dry_run=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

[[ -d "$robot_vlm_repo/deploy/g1/localize" ]] || { echo "robot-vlm localize package is missing" >&2; exit 2; }
[[ -d "$sdk_root/mocap_utils" ]] || { echo "NatNet SDK package is missing: $sdk_root/mocap_utils" >&2; exit 2; }
[[ -f "$glb" ]] || { echo "GLB is missing: $glb" >&2; exit 2; }
[[ -f "$surface_cache" ]] || { echo "surface cache is missing: $surface_cache" >&2; exit 2; }
for value in "$glb_sha256" "$surface_sha256" "$structural_map_sha256"; do
  [[ "$value" =~ ^[0-9a-fA-F]{64}$ ]] || { echo "all SHA-256 identities must be 64 hexadecimal characters" >&2; exit 2; }
done
[[ "$rigid_body_id" =~ ^[0-9]+$ ]] || { echo "rigid-body ID must be a nonnegative integer" >&2; exit 2; }
[[ -n "$rigid_body_name" && "$rigid_body_name" != *$'\n'* ]] || { echo "invalid rigid-body name" >&2; exit 2; }
[[ "$pelvis_rigid_body_id" =~ ^[0-9]+$ ]] || { echo "pelvis rigid-body ID must be a nonnegative integer" >&2; exit 2; }
[[ -n "$pelvis_rigid_body_name" && "$pelvis_rigid_body_name" != *$'\n'* ]] || { echo "invalid pelvis rigid-body name" >&2; exit 2; }
[[ -n "$output_dir" ]] || { echo "output directory is required" >&2; exit 2; }
[[ "$port" =~ ^[0-9]+$ ]] && (( 10#$port > 0 && 10#$port <= 65535 )) || { echo "invalid port" >&2; exit 2; }
[[ "$duration_sec" =~ ^[0-9]+$ ]] && (( 10#$duration_sec >= 60 && 10#$duration_sec <= 3600 )) || { echo "duration must be from 60 through 3600 seconds" >&2; exit 2; }

robot_vlm_repo="$(realpath "$robot_vlm_repo")"
sdk_root="$(realpath "$sdk_root")"
glb="$(realpath "$glb")"
surface_cache="$(realpath "$surface_cache")"
output_dir="$(realpath -m "$output_dir")"
observed_glb_sha256="$(sha256sum "$glb" | awk '{print $1}')"
[[ "$observed_glb_sha256" == "${glb_sha256,,}" ]] || {
  echo "GLB SHA-256 mismatch: expected ${glb_sha256,,}, observed $observed_glb_sha256" >&2
  exit 4
}

# The cache-file digest binds transport bytes. The application loads the cache
# through mesh_map.py, recomputes its surface digest, and compares that digest to
# --surface-sha256 before opening the UI.
cache_file_sha256="$(sha256sum "$surface_cache" | awk '{print $1}')"

if [[ "$dry_run" == false ]]; then
  [[ -n "$motive_client" ]] || { echo "--motive-client is required for a live calibration" >&2; exit 2; }
  for command in docker timeout; do
    command -v "$command" >/dev/null || { echo "missing command: $command" >&2; exit 2; }
  done
  mkdir -p "$output_dir"
fi

container_name="g1-motive-polycam-calibration-${BASHPID}"
docker_command=(
  docker run --rm --name "$container_name"
  --user "$(id -u):$(id -g)"
  --network host
  --cap-drop ALL
  --security-opt no-new-privileges
  --read-only
  --tmpfs /tmp:rw,noexec,nosuid,size=512m
  --env HOME=/tmp/home
  --volume "$script_dir:/opt/calibration:ro"
  --volume "$robot_vlm_repo/deploy/g1/localize:/opt/robot-vlm-localize:ro"
  --volume "$sdk_root:/opt/natnet:ro"
  --volume "$glb:/opt/map/fieldbay.glb:ro"
  --volume "$surface_cache:/opt/map/surface.npz:ro"
  --volume "$output_dir:/opt/output:rw"
  "$image"
  python3 /opt/calibration/motive_polycam_calibration_app.py
  --localize-dir /opt/robot-vlm-localize
  --glb /opt/map/fieldbay.glb
  --surface-cache /opt/map/surface.npz
  --glb-sha256 "${glb_sha256,,}"
  --surface-sha256 "${surface_sha256,,}"
  --structural-map-sha256 "${structural_map_sha256,,}"
  --sdk-root /opt/natnet
  --server-address "$motive_server"
  --client-address "$motive_client"
  --rigid-body-id "$rigid_body_id"
  --rigid-body-name "$rigid_body_name"
  --pelvis-rigid-body-id "$pelvis_rigid_body_id"
  --pelvis-rigid-body-name "$pelvis_rigid_body_name"
  --control-points /opt/output/control-points.json
  --transform-output /opt/output/motive-to-polycam.json
  --port "$port"
)

if [[ "$dry_run" == true ]]; then
  cat <<EOF
schema=g1_motive_polycam_calibration_dry_run_v1
image=$image
glb=$glb
glb_sha256=$observed_glb_sha256
surface_cache=$surface_cache
surface_cache_file_sha256=$cache_file_sha256
surface_sha256=${surface_sha256,,}
structural_map_sha256=${structural_map_sha256,,}
rigid_body_id=$rigid_body_id
rigid_body_name=$rigid_body_name
pelvis_rigid_body_id=$pelvis_rigid_body_id
pelvis_rigid_body_name=$pelvis_rigid_body_name
motive_server=$motive_server
motive_client=$motive_client
output_dir=$output_dir
ui=http://localhost:$port
command_capability=structurally_unavailable
EOF
  printf 'docker_command='
  printf '%q ' "${docker_command[@]}"
  printf '\n'
  exit 0
fi

container_pid=""
cleanup() {
  docker stop --time 3 "$container_name" >/dev/null 2>&1 || true
  if [[ -n "$container_pid" ]]; then
    kill -TERM "$container_pid" 2>/dev/null || true
    wait "$container_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

set +e
timeout --signal=INT --kill-after=15s "${duration_sec}s" "${docker_command[@]}" &
container_pid=$!
wait "$container_pid"
status=$?
container_pid=""
set -e

[[ "$status" -eq 0 || "$status" -eq 124 ]] || exit "$status"
[[ -s "$output_dir/motive-to-polycam.json" ]] || {
  echo "calibration ended without an accepted/rejected transform artifact" >&2
  exit 5
}
python3 - "$output_dir/motive-to-polycam.json" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8"))
if value.get("status") != "accepted" or value.get("evaluator_ready") is not True:
    raise SystemExit("calibration did not finish the independent pelvis-heading gate")
PY
