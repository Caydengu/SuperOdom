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
  --robot-id ID                     operator-asserted robot identity
  --include-docker-image IMAGE      stream a read-only docker image export
  --dry-run                         print the planned remote reads only
EOF
}

output_dir=""
target="unitree@192.168.123.164"
robot_id="unverified"
docker_image=""
dry_run=false

while (( $# )); do
  case "$1" in
    --output-dir) output_dir=$2; shift 2 ;;
    --target) target=$2; shift 2 ;;
    --robot-id) robot_id=$2; shift 2 ;;
    --include-docker-image) docker_image=$2; shift 2 ;;
    --dry-run) dry_run=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

[[ -n "$output_dir" ]] || { echo "--output-dir is required" >&2; exit 2; }
for command in gzip ssh sha256sum; do
  command -v "$command" >/dev/null || { echo "Missing command: $command" >&2; exit 2; }
done

remote_directories=(
  "/home/unitree/geo-179/G1_localization"
  "/home/unitree/G1_localization/ws_slam/src"
  "/home/unitree/geo-179/g1_localize"
  "/home/unitree/geo-179/g1_localize_super_odom"
  "/home/unitree/G1_localization/mocap_utils"
  "/home/unitree/geo-179/superodom"
  "/home/unitree/geo-179/superodom-humanoid"
  "/home/unitree/geo-179/superodom-mid360"
  "/home/unitree/geo-179/superodom-mid360_ws/src"
  "/home/unitree/geo-179/holosoma_g1_side"
  "/home/unitree/geo-179/tml_humanoid_deploy"
  "/home/unitree/geo-179/g1_dex3_viser"
  "/home/unitree/geo-179/g1_grasp_viser"
  "/home/unitree/geo-179/g1_right_arm_ik"
  "/home/unitree/geo-179/g1_camera_calibration"
  "/home/unitree/geo-179/datasets"
  "/home/unitree/g1-onboard"
  "/home/unitree/g1_grasp_tools"
)
remote_files=(
  "/home/unitree/cyclonedds_ws/cyclonedds.xml"
  "/home/unitree/geo-179/G1_localization/ws_slam/src/FAST_LIO_ROS2/config/utlidar.yaml"
  "/home/unitree/G1_localization/ws_slam/src/FAST_LIO_ROS2/config/utlidar.yaml"
  "/home/unitree/geo-179/g1_localize/run_slam.sh"
)

if [[ "$dry_run" == true ]]; then
  echo "target=$target"
  echo "robot_id=$robot_id"
  [[ -z "$docker_image" ]] || echo "docker_image=$docker_image"
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

cat >"$output_dir/metadata/operator_claims.txt" <<EOF
robot_id=$robot_id
robot_id_source=operator_assertion
required_hands=DEX3
remote_target=$target
remote_write_capability=none
EOF

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
ps -ef | grep -Ei 'fast.?lio|super.?odom|livox|utlidar|localiz|teleimager|dex3|grasp|holosoma|amo|policy' | grep -v grep || true
echo
echo "=== Docker containers ==="
docker ps --no-trunc 2>/dev/null || true
echo
echo "=== Docker images ==="
docker images --digests --no-trunc 2>/dev/null || true
if command -v docker >/dev/null 2>&1; then
  docker image inspect superodom:humble-arm64 2>/dev/null || true
  docker history --no-trunc superodom:humble-arm64 2>/dev/null || true
fi
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
  /home/unitree/geo-179/g1_localize_super_odom \
  /home/unitree/geo-179/superodom \
  /home/unitree/geo-179/superodom-humanoid \
  /home/unitree/geo-179/superodom-mid360 \
  /home/unitree/geo-179/tml_humanoid_deploy \
  /home/unitree/geo-179/g1_camera_calibration \
  /home/unitree/g1-onboard; do
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
echo "=== geo-179 top-level sizes ==="
for path in /home/unitree/geo-179/*; do
  du -sh "$path" 2>/dev/null || true
done | sort -h
echo
echo "=== Git worktrees under geo-179 and g1-onboard ==="
find /home/unitree/geo-179 /home/unitree/g1-onboard -maxdepth 5 -type d -name .git -print 2>/dev/null | sort
echo
echo "=== Conda environments ==="
CONDA=/home/unitree/miniforge3/bin/conda
if [ -x "$CONDA" ]; then
  "$CONDA" env list 2>&1 || true
  echo
  echo "--- egonav-deploy explicit packages ---"
  "$CONDA" list --explicit -n egonav-deploy 2>&1 || true
  echo
  echo "--- egonav-deploy pip freeze ---"
  /home/unitree/miniforge3/envs/egonav-deploy/bin/python -m pip freeze 2>&1 || true
fi
echo
echo "=== user services relevant to localization/perception ==="
systemctl --user list-unit-files 2>/dev/null | grep -Ei 'livox|lidar|localiz|teleimager|dex3|grasp|holosoma|policy' || true
systemctl --user --no-pager --type=service --all 2>/dev/null | grep -Ei 'livox|lidar|localiz|teleimager|dex3|grasp|holosoma|policy' || true
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
      "tar --exclude='./build' --exclude='*/build' \
       --exclude='./install' --exclude='*/install' \
       --exclude='./log' --exclude='*/log' \
       --exclude='./bags' --exclude='*/bags' \
       --exclude='./runs' --exclude='*/runs' \
       --exclude='./data' --exclude='*/data' \
       --exclude='./datasets' --exclude='*/datasets' \
       --exclude='./__pycache__' --exclude='*/__pycache__' \
       --exclude='./.cache' --exclude='*/.cache' \
       --exclude='./third_party' --exclude='*/third_party' \
       -C '$remote_path' -czf - ." >"$archive"
    printf '%s\t%s\tpresent\n' "$remote_path" "$archive" >>"$archive_index"
  else
    printf '%s\t\tmissing\n' "$remote_path" >>"$archive_index"
  fi
done

if [[ -n "$docker_image" ]]; then
  docker_label="${docker_image//\//_}"
  docker_label="${docker_label//:/_}"
  docker_archive="$output_dir/archives/docker__${docker_label}.tar.gz"
  ssh "${ssh_options[@]}" "$target" "docker image inspect '$docker_image' >/dev/null && docker image save '$docker_image'" \
    | gzip -1 >"$docker_archive"
  printf 'docker://%s\t%s\tpresent\n' "$docker_image" "$docker_archive" >>"$archive_index"
fi

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
robot_id=$robot_id
output_dir=$output_dir
checksums=$output_dir/SHA256SUMS
EOF
