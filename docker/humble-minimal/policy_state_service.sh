#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: policy_state_service.sh --network-interface IFACE --ros-domain-id ID
       --output-dir PATH [--livox-config PATH] [--image IMAGE]
       [--cpuset-cpus LIST] [--dry-run]

Run the minimal, non-actuating SuperOdometry and atomic pelvis-state service.
When --livox-config is provided, this service receives the Mid-360 directly on
the offboard host and owns that driver. Otherwise an external Livox ROS
publisher must already be running. The typed joint relay must already be
running. This service records no sensor streams.
EOF
}

network_interface=""
ros_domain_id=""
output_dir=""
image="${SUPERODOM_IMAGE:-tml/superodom-humble:h12-shadow-minimal}"
cpuset_cpus=""
livox_config=""
dry_run=false

while (( $# )); do
  case "$1" in
    --network-interface)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      network_interface=$2
      shift 2
      ;;
    --ros-domain-id)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      ros_domain_id=$2
      shift 2
      ;;
    --output-dir)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      output_dir=$2
      shift 2
      ;;
    --image)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      image=$2
      shift 2
      ;;
    --cpuset-cpus)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      cpuset_cpus=$2
      shift 2
      ;;
    --livox-config)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      livox_config=$2
      shift 2
      ;;
    --dry-run)
      dry_run=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

[[ -n "$network_interface" ]] || {
  echo "--network-interface is required" >&2
  exit 2
}
[[ -d "/sys/class/net/$network_interface" ]] || {
  echo "Network interface is unavailable: $network_interface" >&2
  exit 2
}
[[ "$ros_domain_id" =~ ^[0-9]+$ ]] && (( 10#$ros_domain_id <= 232 )) || {
  echo "--ros-domain-id must be an integer from 0 through 232" >&2
  exit 2
}
[[ -n "$output_dir" ]] || {
  echo "--output-dir is required" >&2
  exit 2
}
[[ "$image" =~ ^[A-Za-z0-9][A-Za-z0-9._/@:-]*$ ]] || {
  echo "Image reference contains unsupported characters" >&2
  exit 2
}
[[ -z "$cpuset_cpus" ||
   "$cpuset_cpus" =~ ^[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*$ ]] || {
  echo "--cpuset-cpus must be a Docker-compatible CPU list" >&2
  exit 2
}
if [[ -n "$livox_config" ]]; then
  [[ "$livox_config" == /* && -f "$livox_config" ]] || {
    echo "--livox-config must name an existing absolute file" >&2
    exit 2
  }
  livox_config="$(cd "$(dirname "$livox_config")" && pwd -P)/$(basename "$livox_config")"
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "$script_dir/../.." && pwd -P)"
run_wrapper="$script_dir/run.sh"
source_commit="$(git -C "$repo_root" rev-parse HEAD)"
if [[ -n "$(git -C "$repo_root" status --porcelain)" ]]; then
  source_dirty=true
else
  source_dirty=false
fi
image_id="$(docker image inspect "$image" --format '{{.Id}}')"

mkdir -p "$output_dir"
output_dir="$(cd "$output_dir" && pwd -P)"
if find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  echo "Refusing to reuse non-empty service output: $output_dir" >&2
  exit 2
fi
mkdir -p "$output_dir/logs" "$output_dir/runtime-input"
livox_config_sha256=""
livox_source="external_ros"
if [[ -n "$livox_config" ]]; then
  livox_config_sha256="$(sha256sum "$livox_config" | awk '{print $1}')"
  cp -- "$livox_config" "$output_dir/runtime-input/livox_mid360.json"
  chmod a-w "$output_dir/runtime-input/livox_mid360.json"
  livox_source="oslo_direct"
fi

python3 - "$output_dir/service_identity.json" "$source_commit" \
  "$source_dirty" "$image" "$image_id" "$network_interface" \
  "$ros_domain_id" "$cpuset_cpus" "$livox_source" \
  "$livox_config_sha256" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
record = {
    "schema": "policy_state_service_identity_v1",
    "source_commit": sys.argv[2],
    "source_dirty": sys.argv[3] == "true",
    "image": sys.argv[4],
    "image_id": sys.argv[5],
    "network_interface": sys.argv[6],
    "ros_domain_id": int(sys.argv[7]),
    "cpuset_cpus": sys.argv[8] or None,
    "livox_source": sys.argv[9],
    "livox_config_sha256": sys.argv[10] or None,
}
path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
PY

wrapper_args=(
  --data-dir "$output_dir/runtime-input"
  --output-dir "$output_dir"
  --ros-domain-id "$ros_domain_id"
  --network-interface "$network_interface"
  --container-name h12-superodom-policy-state
)
if [[ -n "$cpuset_cpus" ]]; then
  wrapper_args+=(--cpuset-cpus "$cpuset_cpus")
fi
if [[ "$dry_run" == true ]]; then
  wrapper_args+=(--dry-run)
fi

container_script='set -euo pipefail
source_commit=$1
image_id=$2
direct_livox=$3
livox_config_sha256=$4
launch_pid=""
bridge_pid=""
livox_pid=""
cleaned=false
service_status_max_lines=10000
status_lines=0
heartbeat_index=0
status_path=/output/service_status.jsonl

append_status() {
  local event=$1
  if (( status_lines >= service_status_max_lines )); then
    return 0
  fi
  printf "{\"schema\":\"policy_state_service_status_v1\",\"event\":\"%s\",\"realtime_ns\":%s,\"launch_pid\":%s,\"bridge_pid\":%s,\"source_commit\":\"%s\",\"image_id\":\"%s\"}\n" \
    "$event" "$(date +%s%N)" "${launch_pid:-0}" "${bridge_pid:-0}" \
    "$source_commit" "$image_id" >> "$status_path"
  status_lines=$((status_lines + 1))
}

stop_process() {
  local pid=$1
  local label=$2
  [[ -n "$pid" ]] || return 0
  kill -0 -- "-$pid" 2>/dev/null || {
    wait "$pid" 2>/dev/null || true
    return 0
  }
  kill -INT -- "-$pid" 2>/dev/null || true
  for _ in $(seq 1 75); do
    kill -0 -- "-$pid" 2>/dev/null || break
    sleep 0.2
  done
  if kill -0 -- "-$pid" 2>/dev/null; then
    echo "$label did not stop after SIGINT; sending SIGTERM" >&2
    kill -TERM -- "-$pid" 2>/dev/null || true
    for _ in $(seq 1 25); do
      kill -0 -- "-$pid" 2>/dev/null || break
      sleep 0.2
    done
  fi
  if kill -0 -- "-$pid" 2>/dev/null; then
    echo "$label did not stop after SIGTERM; sending SIGKILL" >&2
    kill -KILL -- "-$pid" 2>/dev/null || true
  fi
  wait "$pid" 2>/dev/null || true
}

cleanup() {
  local status=$?
  if [[ "$cleaned" == true ]]; then
    return
  fi
  cleaned=true
  stop_process "$bridge_pid" policy-state-bridge
  stop_process "$launch_pid" superodometry-launch
  stop_process "$livox_pid" livox-driver
  printf "{\"schema\":\"policy_state_service_stop_v1\",\"realtime_ns\":%s,\"exit_status\":%s,\"launch_pid\":%s,\"bridge_pid\":%s,\"livox_pid\":%s}\n" \
    "$(date +%s%N)" "$status" "${launch_pid:-0}" "${bridge_pid:-0}" "${livox_pid:-0}" \
    > /output/stop_receipt.json
}
trap cleanup EXIT
trap "exit 130" INT TERM

wait_for_type() {
  local topic=$1
  local expected=$2
  local deadline=$((SECONDS + 45))
  local actual=""
  while (( SECONDS < deadline )); do
    actual="$(timeout 2s ros2 topic type "$topic" 2>/dev/null || true)"
    if [[ "$actual" == "$expected" ]]; then
      return 0
    fi
    if [[ -n "$actual" && "$actual" != "$expected" ]]; then
      echo "Unexpected type for $topic: $actual" >&2
      return 31
    fi
    sleep 0.5
  done
  echo "Timed out waiting for $topic" >&2
  return 32
}

subscription_count() {
  ros2 topic info "$1" --verbose \
    | sed -n "s/^Subscription count: \([0-9][0-9]*\)$/\1/p" \
    | head -n 1
}

publisher_count() {
  ros2 topic info "$1" --verbose \
    | sed -n "s/^Publisher count: \([0-9][0-9]*\)$/\1/p" \
    | head -n 1
}

if [[ "$direct_livox" == true ]]; then
  [[ -f /data/livox_mid360.json ]] || {
    echo "Direct Livox configuration is missing" >&2
    exit 30
  }
  actual_livox_sha256="$(sha256sum /data/livox_mid360.json | awk "{print \$1}")"
  [[ "$actual_livox_sha256" == "$livox_config_sha256" ]] || {
    echo "Direct Livox configuration identity mismatch" >&2
    exit 30
  }
  append_status livox_starting
  setsid ros2 run livox_ros_driver2 livox_ros_driver2_node --ros-args \
    -r __node:=h12_livox_oslo_direct \
    -p xfer_format:=1 \
    -p multi_topic:=0 \
    -p data_src:=0 \
    -p publish_freq:=10.0 \
    -p output_data_type:=0 \
    -p frame_id:=livox_frame \
    -p user_config_path:=/data/livox_mid360.json \
    > /output/logs/livox_driver.log 2>&1 &
  livox_pid=$!
fi

wait_for_type /livox/lidar livox_ros_driver2/msg/CustomMsg
wait_for_type /livox/imu sensor_msgs/msg/Imu
if [[ "$direct_livox" == true ]]; then
  python3 - <<'"'"'PY'"'"'
import json
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Imu


class InputProbe(Node):
    def __init__(self) -> None:
        super().__init__("h12_livox_oslo_input_preflight")
        self.receipt_monotonic_ns: list[int] = []
        self.timestamp_error_ns: list[int] = []
        self.header_ns: list[int] = []
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=64,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.create_subscription(Imu, "/livox/imu", self._on_imu, qos)

    def _on_imu(self, message: Imu) -> None:
        receipt_realtime_ns = time.time_ns()
        self.receipt_monotonic_ns.append(time.monotonic_ns())
        header_ns = (
            int(message.header.stamp.sec) * 1_000_000_000
            + int(message.header.stamp.nanosec)
        )
        self.header_ns.append(header_ns)
        self.timestamp_error_ns.append(receipt_realtime_ns - header_ns)


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, math.ceil(probability * len(ordered)) - 1),
    )
    return ordered[index]


rclpy.init()
node = InputProbe()
deadline = time.monotonic() + 5.0
try:
    while len(node.receipt_monotonic_ns) < 500 and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.02)
finally:
    node.destroy_node()
    rclpy.shutdown()

gaps_ms = [
    (current - previous) / 1_000_000.0
    for previous, current in zip(
        node.receipt_monotonic_ns,
        node.receipt_monotonic_ns[1:],
    )
]
timestamp_error_abs_ms = [
    abs(value) / 1_000_000.0 for value in node.timestamp_error_ns
]
header_monotonic = all(
    current > previous
    for previous, current in zip(node.header_ns, node.header_ns[1:])
)
metrics_available = (
    len(node.receipt_monotonic_ns) >= 500
    and bool(gaps_ms)
    and bool(timestamp_error_abs_ms)
)
record = {
    "schema": "livox_input_preflight_v1",
    "source": "oslo_direct",
    "samples": len(node.receipt_monotonic_ns),
    "header_monotonic": header_monotonic,
    "gap_p95_ms": percentile(gaps_ms, 0.95) if gaps_ms else None,
    "gap_p99_ms": percentile(gaps_ms, 0.99) if gaps_ms else None,
    "gap_max_ms": max(gaps_ms) if gaps_ms else None,
    "timestamp_error_abs_p95_ms": (
        percentile(timestamp_error_abs_ms, 0.95)
        if timestamp_error_abs_ms
        else None
    ),
    "timestamp_error_abs_max_ms": (
        max(timestamp_error_abs_ms)
        if timestamp_error_abs_ms
        else None
    ),
}
record["valid"] = bool(
    metrics_available
    and header_monotonic
    and record["gap_p99_ms"] <= 8.0
    and record["gap_max_ms"] <= 12.0
    and record["timestamp_error_abs_p95_ms"] <= 5.0
    and record["timestamp_error_abs_max_ms"] <= 10.0
)
with open("/output/livox_input_preflight.json", "x", encoding="utf-8") as stream:
    json.dump(record, stream, indent=2, sort_keys=True)
    stream.write("\n")
if not record["valid"]:
    raise SystemExit(f"direct Livox input preflight failed: {record}")
PY
fi
pre_subscribers="$(subscription_count /livox/lidar)"
[[ "$pre_subscribers" == "0" ]] || {
  echo "Expected no pre-existing /livox/lidar subscribers, found $pre_subscribers" >&2
  exit 33
}
pre_publishers="$(publisher_count /livox/lidar)"
[[ "$pre_publishers" == "1" ]] || {
  echo "Expected one /livox/lidar publisher, found $pre_publishers" >&2
  exit 34
}

live_config=/opt/superodom_ws/install/share/super_odometry/config/livox_mid360.yaml
[[ -f "$live_config" ]] || { echo "Installed live configuration is missing" >&2; exit 35; }
grep -q "predict_future_secs: 0.0" "$live_config" || {
  echo "Forward prediction must remain disabled" >&2
  exit 36
}
grep -q "min_range: 0.5" "$live_config" || {
  echo "Live terrain range contract is invalid" >&2
  exit 37
}
sha256sum "$live_config" > /output/live_config_sha256.txt

append_status starting
setsid ros2 launch super_odometry livox_humanoid.launch.py \
  use_sim_time:=false config_file:="$live_config" \
  > /output/logs/superodometry.log 2>&1 &
launch_pid=$!

setsid ros2 run g1_root_state_bridge g1-root-state-bridge --ros-args \
  -p root_state_bind_endpoint:=tcp://*:5575 \
  -p policy_state_bind_endpoint:=tcp://*:5576 \
  -p joint_bind_host:=0.0.0.0 \
  -p joint_bind_port:=5576 \
  > /output/logs/policy_state_bridge.log 2>&1 &
bridge_pid=$!
append_status children_started

wait_for_type /state_estimation nav_msgs/msg/Odometry
wait_for_type /pelvis_state_estimation nav_msgs/msg/Odometry
active_subscribers="$(subscription_count /livox/lidar)"
[[ "$active_subscribers" == "1" ]] || {
  echo "Expected exactly one active /livox/lidar subscriber, found $active_subscribers" >&2
  exit 38
}
active_publishers="$(publisher_count /livox/lidar)"
[[ "$active_publishers" == "1" ]] || {
  echo "Expected exactly one active /livox/lidar publisher, found $active_publishers" >&2
  exit 39
}

python3 - <<'"'"'PY'"'"'
import json
import time

import zmq

from g1_root_state_bridge.policy_state_protocol import (
    deserialize_policy_state_v1,
)

context = zmq.Context()
subscriber = context.socket(zmq.SUB)
subscriber.setsockopt(zmq.LINGER, 0)
subscriber.setsockopt(zmq.CONFLATE, 1)
subscriber.setsockopt(zmq.SUBSCRIBE, b"")
subscriber.connect("tcp://127.0.0.1:5576")
deadline = time.monotonic() + 60.0
accepted = []
try:
    while time.monotonic() < deadline and len(accepted) < 2:
        if not subscriber.poll(100, zmq.POLLIN):
            continue
        packet = deserialize_policy_state_v1(subscriber.recv())
        now_ns = time.time_ns()
        if not packet.strictly_valid:
            continue
        if now_ns - packet.estimate_time_ns > 10_000_000:
            continue
        if now_ns - packet.correction_time_ns > 250_000_000:
            continue
        if abs(packet.joint_sync_gap_ns) > 10_000_000:
            continue
        if accepted and (
            packet.source_epoch != accepted[-1]["source_epoch"]
            or packet.sequence <= accepted[-1]["sequence"]
        ):
            accepted.clear()
            continue
        accepted.append(
            {
                "sequence": packet.sequence,
                "source_epoch": packet.source_epoch,
                "estimate_time_ns": packet.estimate_time_ns,
                "correction_time_ns": packet.correction_time_ns,
                "joint_sync_gap_ns": packet.joint_sync_gap_ns,
            }
        )
finally:
    subscriber.close(linger=0)
    context.term()
if len(accepted) != 2:
    raise SystemExit("no two advancing strict same-epoch policy-state packets")
receipt = {
    "schema": "policy_state_service_ready_v1",
    "receipt_realtime_ns": time.time_ns(),
    "accepted": accepted,
}
with open("/output/policy_state_ready.json", "x", encoding="utf-8") as stream:
    json.dump(receipt, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY

append_status ready
while true; do
  if [[ "$direct_livox" == true ]]; then
    kill -0 -- "-$livox_pid" 2>/dev/null || {
      append_status livox_exited
      exit 42
    }
  fi
  kill -0 -- "-$launch_pid" 2>/dev/null || {
    append_status superodometry_exited
    exit 40
  }
  kill -0 -- "-$bridge_pid" 2>/dev/null || {
    append_status bridge_exited
    exit 41
  }
  sleep 1
  heartbeat_index=$((heartbeat_index + 1))
  if (( heartbeat_index % 5 == 0 )); then
    append_status heartbeat
  fi
done
'

SUPERODOM_IMAGE="$image" exec "$run_wrapper" "${wrapper_args[@]}" -- \
  bash -c "$container_script" policy-state-service "$source_commit" "$image_id" \
  "$([[ -n "$livox_config" ]] && echo true || echo false)" \
  "$livox_config_sha256"
