#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
image="${G1_LOCALIZATION_IMAGE:-tml/g1-kiss-localization:1.5.3-imu-gap-bridge-humble}"
exec docker build --file "$repo_root/docker/kiss-live/Dockerfile" --tag "$image" "$repo_root"
