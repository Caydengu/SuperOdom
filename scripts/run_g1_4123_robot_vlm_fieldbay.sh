#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: run_g1_4123_robot_vlm_fieldbay.sh --run-dir PATH [options]

Run the deployment-faithful, passive G1-4123 localization recorder with the
frozen 2026-08-24 Field Bay Polycam map, Gio's two-click UI initialization,
the actual robot-vlm RealBackend consumer, and optional Motive evaluator truth.

This wrapper never launches AMO and creates no robot command publisher.

Options:
  --capture-class CLASS          stationary|amo-walk|amo-stress (default: stationary)
  --duration-sec N               default: 60 stationary, 120 otherwise
  --motive-map-transform PATH    accepted calibration JSON
  --record-unmapped-motive       record ID 42 but do not score absolute map accuracy
  --without-motive-ground-truth run deployment stack without Motive evaluation
  --workspace-root PATH          default: /move/u/caydengu/cayden
  --map-voxel-m M                initial cloud voxel size; default: 0.05
  --snap-src-points N            initial ICP source cap; 0 uses all retained points
  --initial-pose-xyyaw X Y R     supplied map pose; otherwise use Gio's UI
  --dry-run
EOF
}

run_dir=""
capture_class=stationary
duration_sec=""
workspace_root=/move/u/caydengu/cayden
motive_map_transform=""
motive_mode=required
map_voxel_m=0.05
snap_src_points=40000
initial_pose_xyyaw=()
dry_run=false

while (( $# )); do
  case "$1" in
    --run-dir) run_dir=$2; shift 2 ;;
    --capture-class) capture_class=$2; shift 2 ;;
    --duration-sec) duration_sec=$2; shift 2 ;;
    --motive-map-transform) motive_map_transform=$2; shift 2 ;;
    --record-unmapped-motive) motive_mode=unmapped; shift ;;
    --without-motive-ground-truth) motive_mode=disabled; shift ;;
    --workspace-root) workspace_root=$2; shift 2 ;;
    --map-voxel-m) map_voxel_m=$2; shift 2 ;;
    --snap-src-points) snap_src_points=$2; shift 2 ;;
    --initial-pose-xyyaw)
      (( $# >= 4 )) || { echo "--initial-pose-xyyaw requires X Y YAW_RAD" >&2; exit 2; }
      initial_pose_xyyaw=("$2" "$3" "$4"); shift 4 ;;
    --dry-run) dry_run=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

[[ -n "$run_dir" ]] || { usage; exit 2; }
[[ "$capture_class" == stationary || "$capture_class" == amo-walk || "$capture_class" == amo-stress ]] || {
  echo "invalid capture class: $capture_class" >&2
  exit 2
}
if [[ -z "$duration_sec" ]]; then
  if [[ "$capture_class" == stationary ]]; then
    duration_sec=60
  else
    duration_sec=120
  fi
fi
[[ "$duration_sec" =~ ^[0-9]+$ ]] && (( 10#$duration_sec >= 30 && 10#$duration_sec <= 600 )) || {
  echo "duration must be an integer from 30 through 600" >&2
  exit 2
}
[[ "$map_voxel_m" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "invalid map voxel size" >&2; exit 2; }
python3 -c 'import sys; assert float(sys.argv[1]) > 0.0' "$map_voxel_m" 2>/dev/null || { echo "map voxel size must be positive" >&2; exit 2; }
[[ "$snap_src_points" =~ ^[0-9]+$ ]] || { echo "invalid Snap source point cap" >&2; exit 2; }
for value in "${initial_pose_xyyaw[@]}"; do
  python3 -c 'import math,sys; assert math.isfinite(float(sys.argv[1]))' "$value" 2>/dev/null || { echo "initial pose values must be finite" >&2; exit 2; }
done

workspace_root="$(realpath "$workspace_root")"
export G1_LOCALIZATION_IMAGE="${G1_LOCALIZATION_IMAGE:-tml/g1-kiss-localization:1.5.3-imu-gap-bridge-humble}"
localization_repo="$workspace_root/.worktrees/superodom-g1-4123-final-live-localization"
robot_vlm_repo="$workspace_root/.worktrees/robot-vlm-g1-layered-localization-ui-live"
research_run="$workspace_root/research/perceptive-humanoid-diffusion/runs/2026-08-24_g1-4123-robot-vlm-live-localization"
map_audit="$workspace_root/research/perceptive-humanoid-diffusion/runs/2026-08-24_polycam-fieldbay-map-quality-audit"

glb="$map_audit/data/source/8_24_2026.glb"
surface_cache="$research_run/data/maps/polycam-2026-08-24-surface-500k-seed24.npz"
structural_map="$map_audit/data/maps/polycam-2026-08-24-structural.npz"
map_artifact="$research_run/data/maps/src-fieldbay-2026-08-24-robot-vlm-map.npz"
if [[ -z "$motive_map_transform" && "$motive_mode" == required ]]; then
  motive_map_transform="$research_run/calibration/motive-polycam/motive-to-polycam.json"
fi
if [[ "$motive_mode" != required && -n "$motive_map_transform" ]]; then
  echo "--motive-map-transform cannot be combined with an unmapped or disabled Motive mode" >&2
  exit 2
fi

for path in "$localization_repo" "$robot_vlm_repo" "$glb" "$surface_cache" "$structural_map" "$map_artifact"; do
  [[ -e "$path" ]] || { echo "missing frozen input: $path" >&2; exit 2; }
done
if [[ "$motive_mode" == required ]]; then
  [[ -f "$motive_map_transform" ]] || {
    echo "missing accepted Motive-to-Polycam transform: $motive_map_transform" >&2
    echo "finish the independent transform calibration or pass --without-motive-ground-truth" >&2
    exit 2
  }
fi

robot_host=192.168.123.164
robot_python=/home/unitree/miniforge3/envs/egonav-deploy/bin/python
robot_machine_id_sha256=9fd82e8a530408897fdd47c7c1b2f4314778a91339fcedd5db07a03b8c2b1d20
route_to_robot="$(ip route get "$robot_host" | head -1)"
network_interface="$(awk '{for(i=1;i<=NF;i++) if($i=="dev") print $(i+1)}' <<<"$route_to_robot")"
[[ -n "$network_interface" ]] || { echo "could not resolve the G1 network interface: $route_to_robot" >&2; exit 4; }

command=(
  "$localization_repo/scripts/run_g1_kiss_live_verification.sh"
  --run-dir "$run_dir"
  --network-interface "$network_interface"
  --robot-python "$robot_python"
  --expected-robot-machine-id-sha256 "$robot_machine_id_sha256"
  --duration-sec "$duration_sec"
  --capture-class "$capture_class"
  --motive-server 172.24.68.77
  --rigid-body-id 42
  --rigid-body-name G1_PELVIS_F_4123
  --integrated-robot-vlm
  --robot-vlm-repo "$robot_vlm_repo"
  --glb "$glb"
  --glb-sha256 634e7de6eb7023b67beef8048c9b6d0aa26752641b40fe6adab34c8ce329e8cf
  --surface-cache "$surface_cache"
  --surface-sha256 f2e9c375304a3ceb0dd8998d58cda861b529ee8b4668e64c5dd7cf0d6cae2fc4
  --structural-map "$structural_map"
  --structural-map-sha256 3ce3af22d5eaf49d46878b8a9a07c86a7b0a132b6ba0c7ccb95487d7dc25863c
  --map-artifact "$map_artifact"
  --map-artifact-sha256 bce9bf243b9ba864e8c5a9ab41b5b18a6f45fe9fd0c835a37ec7cdcbd85c334c
  --map-voxel-m "$map_voxel_m"
  --snap-src-points "$snap_src_points"
)
if (( ${#initial_pose_xyyaw[@]} )); then
  command+=(--initial-pose-xyyaw "${initial_pose_xyyaw[@]}")
fi
if [[ "$motive_mode" == required ]]; then
  command+=(--motive-mode required --motive-map-transform "$motive_map_transform")
else
  command+=(--motive-mode "$motive_mode")
fi
if [[ "$dry_run" == true ]]; then
  command+=(--dry-run)
fi

echo "G1-4123 Field Bay localization qualification"
echo "  class: $capture_class"
echo "  duration: ${duration_sec}s"
echo "  G1 interface: $network_interface"
echo "  G1 Python: $robot_python"
echo "  G1 fingerprint: ${robot_machine_id_sha256:0:12}..."
echo "  Gio UI: http://localhost:8082"
echo "  initial pose: ${initial_pose_xyyaw[*]:-Gio UI}"
echo "  initialization points: voxel=${map_voxel_m}m cap=${snap_src_points}"
if [[ "$motive_mode" == disabled ]]; then
  echo "  Motive ground truth: disabled (absolute accuracy is not scored)"
elif [[ "$motive_mode" == unmapped ]]; then
  echo "  Motive ground truth: recording raw ID 42 (absolute accuracy is not scored)"
else
  echo "  Motive ground truth: required"
fi
echo "  command capability: structurally unavailable"
exec "${command[@]}"
