#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: run_live.sh --network-interface IFACE [--ros-domain-id ID] [--lowstate-port PORT] [--container-name NAME]" >&2
}

network_interface=""
ros_domain_id=0
lowstate_port=5589
container_name=""
while (( $# )); do
  case "$1" in
    --network-interface) network_interface=$2; shift 2 ;;
    --ros-domain-id) ros_domain_id=$2; shift 2 ;;
    --lowstate-port) lowstate_port=$2; shift 2 ;;
    --container-name) container_name=$2; shift 2 ;;
    *) usage; exit 2 ;;
  esac
done
[[ -n "$network_interface" && -d "/sys/class/net/$network_interface" ]] || { usage; exit 2; }
[[ "$ros_domain_id" =~ ^[0-9]+$ ]] && (( 10#$ros_domain_id <= 232 )) || { echo "invalid ROS domain" >&2; exit 2; }
[[ "$lowstate_port" =~ ^[0-9]+$ ]] && (( 10#$lowstate_port > 0 && 10#$lowstate_port <= 65535 )) || { echo "invalid port" >&2; exit 2; }
[[ -z "$container_name" || "$container_name" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "invalid container name" >&2; exit 2; }

image="${G1_LOCALIZATION_IMAGE:-tml/g1-kiss-localization:1.3.0-humble}"
cyclonedds_uri="<CycloneDDS><Domain id=\"any\"><General><Interfaces><NetworkInterface name=\"$network_interface\" /></Interfaces></General></Domain></CycloneDDS>"
docker_args=(run --rm)
if [[ -n "$container_name" ]]; then
  docker_args+=(--name "$container_name")
fi
exec docker "${docker_args[@]}" \
  --network host \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=256m \
  --env "ROS_DOMAIN_ID=$ros_domain_id" \
  --env RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  --env "CYCLONEDDS_URI=$cyclonedds_uri" \
  "$image" \
  g1-kiss-live-localization \
    --lowstate-bind-port "$lowstate_port" \
    --gyro-bias-radps 0.025702817208593076 -0.02178237836035201 -0.01574406003550275
