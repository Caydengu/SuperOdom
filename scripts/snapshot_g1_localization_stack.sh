#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: snapshot_g1_localization_stack.sh --output-dir PATH [options]

Creates a read-only local snapshot of the G1's localization source, configs,
and runtime inventory. The command never writes to the robot.

Options:
  --output-dir PATH                 required; must not already exist
  --target USER@HOST                default: unitree@192.168.123.164
  --dry-run                         print the planned remote reads only
EOF
}

output_dir=""
target="unitree@192.168.123.164"
dry_run=false

while (( $# )); do
  case "$1" in
    --output-dir) output_dir=$2; shift 2 ;;
    --target) target=$2; shift 2 ;;
    --dry-run) dry_run=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

[[ -n "$output_dir" ]] || { echo "--output-dir is required" >&2; exit 2; }
for command in ssh sha256sum; do
  command -v "$command" >/dev/null || { echo "Missing command: $command" >&2; exit 2; }
done

remote_directories=(
  "/home/unitree/geo-179/G1_localization/ws_slam/src"
  "/home/unitree/G1_localization/ws_slam/src"
  "/home/unitree/geo-179/g1_localize"
  "/home/unitree/geo-179/g1_localize_super_odom"
  "/home/unitree/geo-179/G1_localization/mocap_utils"
  "/home/unitree/G1_localization/mocap_utils"
)
remote_files=(
  "/home/unitree/cyclonedds_ws/cyclonedds.xml"
  "/home/unitree/geo-179/G1_localization/ws_slam/src/FAST_LIO_ROS2/config/utlidar.yaml"
  "/home/unitree/G1_localization/ws_slam/src/FAST_LIO_ROS2/config/utlidar.yaml"
  "/home/unitree/geo-179/g1_localize/run_slam.sh"
)

if [[ "$dry_run" == true ]]; then
  echo "target=$target"
  echo "output_dir=$output_dir"
  printf 'remote_directory=%s\n' "${remote_directories[@]}"
  printf 'remote_file=%s\n' "${remote_files[@]}"
  echo "remote_write_capability=none"
  exit 0
fi

[[ ! -e "$output_dir" ]] || {
  echo "Refusing to overwrite existing output: $output_dir" >&2
  exit 2
}
mkdir -p "$output_dir"/{archives,files,metadata}
output_dir="$(cd "$output_dir" && pwd -P)"

ssh_options=(-o BatchMode=yes -o ConnectTimeout=10)

ssh "${ssh_options[@]}" "$target" 'bash -s' >"$output_dir/metadata/runtime_inventory.txt" <<'REMOTE'
set -u
echo "snapshot_realtime=$(date --iso-8601=ns 2>/dev/null || date)"
echo "hostname=$(hostname)"
echo "uname=$(uname -a)"
echo "whoami=$(whoami)"
echo "uptime=$(uptime)"
echo
echo "=== OS release ==="
cat /etc/os-release 2>/dev/null || true
echo
echo "=== storage ==="
df -h /home/unitree 2>/dev/null || true
echo
echo "=== localization processes ==="
ps -ef | grep -E 'fast.?lio|super.?odom|livox|utlidar|localiz|teleimager' | grep -v grep || true
echo
echo "=== Docker containers ==="
docker ps --no-trunc 2>/dev/null || true
echo
echo "=== Docker images ==="
docker images --digests --no-trunc 2>/dev/null || true
echo
echo "=== ROS environments ==="
for setup in /opt/ros/foxy/setup.bash /home/unitree/cyclonedds_ws/install/setup.bash; do
  if [ -f "$setup" ]; then echo "present $setup"; else echo "missing $setup"; fi
done
echo
echo "=== candidate trees and Git state ==="
for path in \
  /home/unitree/geo-179/G1_localization/ws_slam \
  /home/unitree/G1_localization/ws_slam \
  /home/unitree/geo-179/g1_localize \
  /home/unitree/geo-179/g1_localize_super_odom; do
  echo "--- $path ---"
  if [ ! -e "$path" ]; then
    echo missing
    continue
  fi
  du -sh "$path" 2>/dev/null || true
  if git -C "$path" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git -C "$path" status --short --branch 2>&1 || true
    git -C "$path" rev-parse HEAD 2>&1 || true
    git -C "$path" remote -v 2>&1 || true
    git -C "$path" diff --stat 2>&1 || true
    git -C "$path" ls-files --others --exclude-standard 2>&1 || true
  else
    echo "not a Git worktree"
  fi
done
echo
echo "=== relevant config checksums ==="
find \
  /home/unitree/geo-179/G1_localization/ws_slam/src \
  /home/unitree/G1_localization/ws_slam/src \
  /home/unitree/geo-179/g1_localize \
  /home/unitree/geo-179/g1_localize_super_odom \
  -type f \( -name '*.yaml' -o -name '*.yml' -o -name '*.launch.py' -o -name '*.xml' -o -name 'run_slam.sh' \) \
  -print0 2>/dev/null | sort -z | xargs -0 -r sha256sum
REMOTE

archive_index="$output_dir/metadata/archive_index.tsv"
printf 'remote_path\tlocal_archive\tstatus\n' >"$archive_index"
for remote_path in "${remote_directories[@]}"; do
  label="${remote_path#/home/unitree/}"
  label="${label//\//__}"
  archive="$output_dir/archives/${label}.tar.gz"
  if ssh "${ssh_options[@]}" "$target" "test -d '$remote_path'"; then
    ssh "${ssh_options[@]}" "$target" \
      "tar --exclude='./build' --exclude='./install' --exclude='./log' \
       --exclude='./bags' --exclude='./runs' --exclude='./data' \
       --exclude='./__pycache__' -C '$remote_path' -czf - ." >"$archive"
    printf '%s\t%s\tpresent\n' "$remote_path" "$archive" >>"$archive_index"
  else
    printf '%s\t\tmissing\n' "$remote_path" >>"$archive_index"
  fi
done

file_index="$output_dir/metadata/file_index.tsv"
printf 'remote_path\tlocal_file\tstatus\n' >"$file_index"
for remote_path in "${remote_files[@]}"; do
  label="${remote_path#/home/unitree/}"
  label="${label//\//__}"
  local_file="$output_dir/files/$label"
  if ssh "${ssh_options[@]}" "$target" "test -f '$remote_path'"; then
    ssh "${ssh_options[@]}" "$target" "cat '$remote_path'" >"$local_file"
    printf '%s\t%s\tpresent\n' "$remote_path" "$local_file" >>"$file_index"
  else
    printf '%s\t\tmissing\n' "$remote_path" >>"$file_index"
  fi
done

(
  cd "$output_dir"
  find archives files metadata -type f ! -name SHA256SUMS -print0 \
    | sort -z \
    | xargs -0 -r sha256sum >SHA256SUMS
)

cat <<EOF
PASS: read-only G1 localization snapshot
target=$target
output_dir=$output_dir
checksums=$output_dir/SHA256SUMS
EOF
