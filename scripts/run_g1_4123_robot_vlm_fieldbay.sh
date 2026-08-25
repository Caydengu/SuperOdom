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
  --without-motive-ground-truth run deployment stack without Motive evaluation
  --workspace-root PATH          default: /move/u/caydengu/cayden
  --dry-run
EOF
}

run_dir=""
capture_class=stationary
duration_sec=""
workspace_root=/move/u/caydengu/cayden
motive_map_transform=""
without_motive_ground_truth=false
dry_run=false

while (( $# )); do
  case "$1" in
    --run-dir) run_dir=$2; shift 2 ;;
    --capture-class) capture_class=$2; shift 2 ;;
    --duration-sec) duration_sec=$2; shift 2 ;;
    --motive-map-transform) motive_map_transform=$2; shift 2 ;;
    --without-motive-ground-truth) without_motive_ground_truth=true; shift ;;
    --workspace-root) workspace_root=$2; shift 2 ;;
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

workspace_root="$(realpath "$workspace_root")"
localization_repo="$workspace_root/.worktrees/superodom-g1-4123-final-live-localization"
robot_vlm_repo="$workspace_root/.worktrees/robot-vlm-g1-layered-localization-ui-live"
research_run="$workspace_root/research/perceptive-humanoid-diffusion/runs/2026-08-24_g1-4123-robot-vlm-live-localization"
map_audit="$workspace_root/research/perceptive-humanoid-diffusion/runs/2026-08-24_polycam-fieldbay-map-quality-audit"

glb="$map_audit/data/source/8_24_2026.glb"
surface_cache="$research_run/data/maps/polycam-2026-08-24-surface-500k-seed24.npz"
structural_map="$map_audit/data/maps/polycam-2026-08-24-structural.npz"
map_artifact="$research_run/data/maps/src-fieldbay-2026-08-24-robot-vlm-map.npz"
if [[ -z "$motive_map_transform" && "$without_motive_ground_truth" == false ]]; then
  motive_map_transform="$research_run/calibration/motive-polycam/motive-to-polycam.json"
fi

for path in "$localization_repo" "$robot_vlm_repo" "$glb" "$surface_cache" "$structural_map" "$map_artifact"; do
  [[ -e "$path" ]] || { echo "missing frozen input: $path" >&2; exit 2; }
done
if [[ "$without_motive_ground_truth" == false ]]; then
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
)
if [[ "$without_motive_ground_truth" == true ]]; then
  command+=(--motive-mode disabled)
else
  command+=(--motive-mode required --motive-map-transform "$motive_map_transform")
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
if [[ "$without_motive_ground_truth" == true ]]; then
  echo "  Motive ground truth: disabled (absolute accuracy is not scored)"
else
  echo "  Motive ground truth: required"
fi
echo "  command capability: structurally unavailable"
exec "${command[@]}"
