#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: run.sh --data-dir PATH --output-dir PATH [--rmw cyclonedds|fastrtps] [--dry-run] -- COMMAND...
EOF
}

data_dir=""
output_dir=""
rmw="cyclonedds"
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
  --network host
  --cap-drop ALL
  --security-opt no-new-privileges
  --user "$(id -u):$(id -g)"
  --env HOME=/tmp
  --env "RMW_IMPLEMENTATION=$rmw_implementation"
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

