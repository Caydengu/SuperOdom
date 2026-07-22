#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/humble/setup.bash

workspace_setup=/opt/superodom_ws/install/setup.bash
if [[ ! -r "$workspace_setup" ]]; then
  echo "SuperOdometry workspace is missing: $workspace_setup" >&2
  exit 1
fi
source "$workspace_setup"

set -u
exec "$@"
