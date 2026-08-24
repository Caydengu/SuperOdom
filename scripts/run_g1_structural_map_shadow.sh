#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: run_g1_structural_map_shadow.sh --network-interface IFACE --map MAP.npz --map-sha256 HEX [options]

Run only the slow, command-incapable structural-map correction lane alongside
an already-running G1 KISS local-localization producer. The map is mounted
read-only and its exact digest is verified before Docker starts.

Options:
  --network-interface IFACE  ROS/DDS interface carrying the G1 streams (required)
  --map PATH                 processed Polycam structural NPZ (required)
  --map-sha256 HEX           required exact SHA-256 of PATH
  --map-key KEY              default: map_xy_all_5cm
  --map-version N            default: 1
  --map-epoch N              default: 1
  --duration-sec N           bounded run, 10..3600; default: 300
  --ros-domain-id ID         default: 0
  --root-state-port PORT     HSROOT02 input; default: 5575
  --map-correction-port PORT RVMAP001 output; default: 5577
  --container-name NAME      optional stable Docker name
  --dry-run                  resolve and print the launch; no Docker I/O
EOF
}

network_interface=""
map_path=""
map_sha256=""
map_key=map_xy_all_5cm
map_version=1
map_epoch=1
duration_sec=300
ros_domain_id=0
root_state_port=5575
map_correction_port=5577
container_name="g1-structural-map-${BASHPID}"
dry_run=false

while (( $# )); do
  case "$1" in
    --network-interface) network_interface=$2; shift 2 ;;
    --map) map_path=$2; shift 2 ;;
    --map-sha256) map_sha256=$2; shift 2 ;;
    --map-key) map_key=$2; shift 2 ;;
    --map-version) map_version=$2; shift 2 ;;
    --map-epoch) map_epoch=$2; shift 2 ;;
    --duration-sec) duration_sec=$2; shift 2 ;;
    --ros-domain-id) ros_domain_id=$2; shift 2 ;;
    --root-state-port) root_state_port=$2; shift 2 ;;
    --map-correction-port) map_correction_port=$2; shift 2 ;;
    --container-name) container_name=$2; shift 2 ;;
    --dry-run) dry_run=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

[[ "$network_interface" =~ ^[A-Za-z0-9_.:-]+$ ]] || { usage; exit 2; }
[[ -f "$map_path" ]] || { echo "map is not a regular file: $map_path" >&2; exit 2; }
[[ "$map_sha256" =~ ^[0-9a-fA-F]{64}$ ]] || { echo "map SHA-256 must contain 64 hexadecimal characters" >&2; exit 2; }
[[ "$map_key" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "invalid map key" >&2; exit 2; }
[[ "$map_version" =~ ^[0-9]+$ ]] && (( 10#$map_version > 0 )) || { echo "invalid map version" >&2; exit 2; }
[[ "$map_epoch" =~ ^[0-9]+$ ]] && (( 10#$map_epoch > 0 )) || { echo "invalid map epoch" >&2; exit 2; }
[[ "$duration_sec" =~ ^[0-9]+$ ]] && (( 10#$duration_sec >= 10 && 10#$duration_sec <= 3600 )) || { echo "duration must be an integer from 10 through 3600" >&2; exit 2; }
[[ "$ros_domain_id" =~ ^[0-9]+$ ]] && (( 10#$ros_domain_id <= 232 )) || { echo "invalid ROS domain" >&2; exit 2; }
for port in "$root_state_port" "$map_correction_port"; do
  [[ "$port" =~ ^[0-9]+$ ]] && (( 10#$port > 0 && 10#$port <= 65535 )) || { echo "invalid port: $port" >&2; exit 2; }
done
[[ "$root_state_port" != "$map_correction_port" ]] || { echo "root-state and map-correction ports must differ" >&2; exit 2; }
[[ "$container_name" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "invalid container name" >&2; exit 2; }

map_path="$(realpath "$map_path")"
observed_sha256="$(sha256sum "$map_path" | awk '{print $1}')"
[[ "${observed_sha256,,}" == "${map_sha256,,}" ]] || {
  echo "map SHA-256 mismatch: expected ${map_sha256,,}, observed $observed_sha256" >&2
  exit 4
}

image="${G1_LOCALIZATION_IMAGE:-tml/g1-kiss-localization:1.4.0-humble}"
cyclonedds_uri="<CycloneDDS><Domain id=\"any\"><General><Interfaces><NetworkInterface name=\"$network_interface\" /></Interfaces></General></Domain></CycloneDDS>"
container_map=/opt/structural-map/fieldbay.npz

if [[ "$dry_run" == false ]]; then
  for command in docker timeout; do
    command -v "$command" >/dev/null || { echo "missing command: $command" >&2; exit 2; }
  done
  [[ -d "/sys/class/net/$network_interface" ]] || { echo "network interface is absent" >&2; exit 4; }
fi

docker_command=(
  docker run --rm --name "$container_name"
  --network host
  --cap-drop ALL
  --security-opt no-new-privileges
  --read-only
  --tmpfs /tmp:rw,noexec,nosuid,size=256m
  --volume "$map_path:$container_map:ro"
  --env "ROS_DOMAIN_ID=$ros_domain_id"
  --env RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
  --env "CYCLONEDDS_URI=$cyclonedds_uri"
  "$image"
  g1-structural-map-localization
  --map "$container_map"
  --map-key "$map_key"
  --map-version "$map_version"
  --map-epoch "$map_epoch"
  --root-endpoint "tcp://127.0.0.1:$root_state_port"
  --map-bind "tcp://*:$map_correction_port"
)

if [[ "$dry_run" == true ]]; then
  cat <<EOF
schema=g1_structural_map_shadow_dry_run_v1
image=$image
map=$map_path
map_sha256=$observed_sha256
map_key=$map_key
duration_sec=$duration_sec
root_state_endpoint=tcp://127.0.0.1:$root_state_port
map_correction_endpoint=tcp://127.0.0.1:$map_correction_port
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
