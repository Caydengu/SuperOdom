#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: run_g1_kiss_live_stack.sh --network-interface IFACE [options]

Bounded passive shadow launcher for the G1-4123 localization producer. It
stages and starts only the subscriber-side LowState relay on the robot, then
runs the ROS 2/KISS producer in the pinned offboard container. It never starts
an AMO policy or creates a Unitree command publisher.

Options:
  --network-interface IFACE       offboard NIC on 192.168.123.0/24 (required)
  --duration-sec N                bounded run, 10..3600; default: 300
  --robot-host IP                 default: 192.168.123.164
  --robot-user USER               default: unitree
  --offboard-robot-address IP     default: resolve from route to robot
  --robot-dds-interface IFACE     default: eth0
  --robot-python PATH             default: egonav-deploy Python on G1
  --ros-domain-id ID              default: 0
  --lowstate-port PORT            default: 5589
  --dry-run                       print resolved commands; no network or Docker I/O
EOF
}

network_interface=""
duration_sec=300
robot_host=192.168.123.164
robot_user=unitree
offboard_robot_address=""
robot_dds_interface=eth0
robot_python=/home/unitree/miniforge3/envs/egonav-deploy/bin/python
ros_domain_id=0
lowstate_port=5589
dry_run=false

while (( $# )); do
  case "$1" in
    --network-interface) network_interface=$2; shift 2 ;;
    --duration-sec) duration_sec=$2; shift 2 ;;
    --robot-host) robot_host=$2; shift 2 ;;
    --robot-user) robot_user=$2; shift 2 ;;
    --offboard-robot-address) offboard_robot_address=$2; shift 2 ;;
    --robot-dds-interface) robot_dds_interface=$2; shift 2 ;;
    --robot-python) robot_python=$2; shift 2 ;;
    --ros-domain-id) ros_domain_id=$2; shift 2 ;;
    --lowstate-port) lowstate_port=$2; shift 2 ;;
    --dry-run) dry_run=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

[[ "$network_interface" =~ ^[A-Za-z0-9_.:-]+$ ]] || { usage; exit 2; }
[[ "$robot_dds_interface" =~ ^[A-Za-z0-9_.:-]+$ ]] || { echo "invalid robot DDS interface" >&2; exit 2; }
[[ "$robot_user" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "invalid robot user" >&2; exit 2; }
[[ "$robot_host" =~ ^[A-Za-z0-9_.:-]+$ ]] || { echo "invalid robot host" >&2; exit 2; }
[[ "$robot_python" =~ ^/[A-Za-z0-9._/-]+$ ]] || { echo "invalid robot Python path" >&2; exit 2; }
[[ "$duration_sec" =~ ^[0-9]+$ ]] && (( 10#$duration_sec >= 10 && 10#$duration_sec <= 3600 )) || {
  echo "duration must be an integer from 10 through 3600" >&2
  exit 2
}
[[ "$ros_domain_id" =~ ^[0-9]+$ ]] && (( 10#$ros_domain_id <= 232 )) || { echo "invalid ROS domain" >&2; exit 2; }
[[ "$lowstate_port" =~ ^[0-9]+$ ]] && (( 10#$lowstate_port > 0 && 10#$lowstate_port <= 65535 )) || { echo "invalid port" >&2; exit 2; }

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "$script_dir/.." && pwd -P)"
remote_stage=/tmp/g1-kiss-live-localization
remote_pid_file="$remote_stage/relay-${lowstate_port}.pid"
container_name="g1-kiss-live-${BASHPID}"

if [[ "$dry_run" == false ]]; then
  for command in docker ip scp ssh timeout; do
    command -v "$command" >/dev/null || { echo "missing command: $command" >&2; exit 2; }
  done
  [[ -d "/sys/class/net/$network_interface" ]] || { echo "network interface is absent" >&2; exit 4; }
  route_to_robot="$(ip route get "$robot_host" | head -1)"
  [[ "$route_to_robot" != *" via "* ]] || { echo "robot route is not direct: $route_to_robot" >&2; exit 4; }
  [[ "$route_to_robot" == *" dev $network_interface "* ]] || { echo "robot route uses the wrong NIC: $route_to_robot" >&2; exit 4; }
  if [[ -z "$offboard_robot_address" ]]; then
    offboard_robot_address="$(awk '{for(i=1;i<=NF;i++) if($i=="src") print $(i+1)}' <<<"$route_to_robot")"
  fi
  [[ "$offboard_robot_address" == 192.168.123.* ]] || { echo "unexpected offboard address: $offboard_robot_address" >&2; exit 4; }
else
  offboard_robot_address="${offboard_robot_address:-192.168.123.11}"
fi

relay_command="PYTHONPATH=$remote_stage $robot_python -m g1_root_state_bridge.g1_dynamic_capture_relay --target-host $offboard_robot_address --target-port $lowstate_port --network-interface $robot_dds_interface --domain-id 0 --duration-sec $duration_sec"
local_command="$repo_root/docker/kiss-live/run_live.sh --network-interface $network_interface --ros-domain-id $ros_domain_id --lowstate-port $lowstate_port --container-name $container_name"

if [[ "$dry_run" == true ]]; then
  cat <<EOF
schema=g1_kiss_live_stack_dry_run_v1
robot=$robot_user@$robot_host
offboard_robot_address=$offboard_robot_address
duration_sec=$duration_sec
stage_source=$repo_root/g1_root_state_bridge/g1_root_state_bridge
stage_target=$remote_stage/g1_root_state_bridge
remote_relay=$relay_command
local_producer=$local_command
command_capability=structurally_unavailable
EOF
  exit 0
fi

ssh -o BatchMode=yes "$robot_user@$robot_host" \
  "test -x '$robot_python' && '$robot_python' -c 'import unitree_sdk2py' && mkdir -p '$remote_stage'"
scp -q -r "$repo_root/g1_root_state_bridge/g1_root_state_bridge" \
  "$robot_user@$robot_host:$remote_stage/"

relay_pid=""
producer_pid=""
cleanup() {
  docker stop --time 3 "$container_name" >/dev/null 2>&1 || true
  ssh -o BatchMode=yes -o ConnectTimeout=3 "$robot_user@$robot_host" \
    "if test -s '$remote_pid_file'; then relay_pid=\$(cat '$remote_pid_file'); kill -TERM \"\$relay_pid\" 2>/dev/null || true; fi; pkill -TERM -f '[g]1_dynamic_capture_relay.*--target-port $lowstate_port' 2>/dev/null || true; rm -f '$remote_pid_file'" \
    >/dev/null 2>&1 || true
  for pid in "$producer_pid" "$relay_pid"; do
    if [[ -n "$pid" ]]; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  for pid in "$producer_pid" "$relay_pid"; do
    if [[ -n "$pid" ]]; then
      wait "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

ssh -o BatchMode=yes "$robot_user@$robot_host" \
  "echo \$\$ > '$remote_pid_file'; exec timeout --signal=TERM --kill-after=5s '${duration_sec}s' bash -lc '$relay_command'" &
relay_pid=$!
sleep 2
kill -0 "$relay_pid" 2>/dev/null || { echo "passive LowState relay exited during startup" >&2; wait "$relay_pid"; exit 5; }

set +e
timeout --signal=INT --kill-after=15s "${duration_sec}s" bash -lc "$local_command" &
producer_pid=$!
wait "$producer_pid"
producer_status=$?
producer_pid=""
wait "$relay_pid"
relay_status=$?
set -e
relay_pid=""

[[ "$producer_status" -eq 0 || "$producer_status" -eq 124 ]] || { echo "local producer failed: $producer_status" >&2; exit "$producer_status"; }
[[ "$relay_status" -eq 0 || "$relay_status" -eq 124 ]] || { echo "passive relay failed: $relay_status" >&2; exit "$relay_status"; }
