#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "$script_dir/../.." && pwd -P)"

if (( $# > 1 )); then
  echo "Usage: $0 [IMAGE_TAG]" >&2
  exit 2
fi

# shellcheck disable=SC1091
source "$script_dir/dependency-lock.env"

image_tag="${1:-tml/superodom-humble:57a6e23-minimal}"

exec docker build \
  --file "$script_dir/Dockerfile" \
  --tag "$image_tag" \
  --build-arg "ROS_BASE_IMAGE=$ROS_BASE_IMAGE" \
  --build-arg "SUPERODOM_REVISION=$SUPERODOM_REVISION" \
  --build-arg "LIVOX_DRIVER_REVISION=$LIVOX_DRIVER_REVISION" \
  --build-arg "LIVOX_SDK2_REVISION=$LIVOX_SDK2_REVISION" \
  --build-arg "GTSAM_REVISION=$GTSAM_REVISION" \
  --build-arg "SOPHUS_REVISION=$SOPHUS_REVISION" \
  --build-arg "PINOCCHIO_DEB_VERSION=$PINOCCHIO_DEB_VERSION" \
  --build-arg "PYBULLET_VERSION=$PYBULLET_VERSION" \
  "$repo_root"
