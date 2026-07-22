#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: run.sh --data-dir PATH --output-dir PATH [--rmw cyclonedds|fastrtps] [--ros-domain-id 0..232] [--network-interface IFACE] [--container-name NAME] [--dry-run] -- COMMAND...
EOF
}

data_dir=""
output_dir=""
rmw="cyclonedds"
ros_domain_id=0
container_name=""
network_interface=""
dry_run=false

while (( $# )); do
  case "$1" in
    --data-dir)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      data_dir=$2
      shift 2
      ;;
    --output-dir)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      output_dir=$2
      shift 2
      ;;
    --rmw)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      rmw=$2
      shift 2
      ;;
    --ros-domain-id)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      ros_domain_id=$2
      shift 2
      ;;
    --container-name)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      container_name=$2
      shift 2
      ;;
    --network-interface)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      network_interface=$2
      shift 2
      ;;
    --dry-run)
      dry_run=true
      shift
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

[[ -n "$data_dir" && -d "$data_dir" ]] || {
  echo "--data-dir must name an existing directory" >&2
  exit 2
}
[[ -n "$output_dir" && -d "$output_dir" ]] || {
  echo "--output-dir must name an existing directory" >&2
  exit 2
}
(( $# > 0 )) || { echo "A command is required after --" >&2; usage; exit 2; }

if [[ ! "$ros_domain_id" =~ ^[0-9]+$ ]] || (( 10#$ros_domain_id > 232 )); then
  echo "ROS domain ID must be an integer from 0 through 232" >&2
  exit 2
fi
ros_domain_id=$((10#$ros_domain_id))

if [[ -n "$container_name" && ! "$container_name" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
  echo "Container name must use Docker-compatible letters, digits, underscores, periods, or dashes" >&2
  exit 2
fi

if [[ -n "$network_interface" ]]; then
  if [[ ! "$network_interface" =~ ^[A-Za-z0-9_.:-]+$ ]] || [[ ! -d "/sys/class/net/$network_interface" ]]; then
    echo "Network interface must name an existing host interface" >&2
    exit 2
  fi
  cyclonedds_uri="<CycloneDDS><Domain id=\"any\"><General><Interfaces><NetworkInterface name=\"$network_interface\" /></Interfaces></General></Domain></CycloneDDS>"
fi

data_dir="$(cd "$data_dir" && pwd -P)"
output_dir="$(cd "$output_dir" && pwd -P)"

case "$rmw" in
  cyclonedds) rmw_implementation=rmw_cyclonedds_cpp ;;
  fastrtps) rmw_implementation=rmw_fastrtps_cpp ;;
  *)
    echo "Unsupported middleware: $rmw (expected cyclonedds or fastrtps)" >&2
    exit 2
    ;;
esac

image="${SUPERODOM_IMAGE:-tml/superodom-humble:57a6e23-minimal}"
docker_command=(
  docker run --rm
)
if [[ -n "$container_name" ]]; then
  docker_command+=(--name "$container_name")
fi
docker_command+=(
  --network host
  --cap-drop ALL
  --security-opt no-new-privileges
  --user "$(id -u):$(id -g)"
  --env HOME=/tmp
  --env "RMW_IMPLEMENTATION=$rmw_implementation"
  --env "ROS_DOMAIN_ID=$ros_domain_id"
)
if [[ -n "$network_interface" ]]; then
  docker_command+=(--env "CYCLONEDDS_URI=$cyclonedds_uri")
fi
docker_command+=(
  --mount "type=bind,src=$data_dir,dst=/data,readonly"
  --mount "type=bind,src=$output_dir,dst=/output"
  "$image"
  "$@"
)

if [[ "$dry_run" == true ]]; then
  printf '%q ' "${docker_command[@]}"
  printf '\n'
  exit 0
fi

exec "${docker_command[@]}"
