#!/usr/bin/env bash
set -euo pipefail
mkdir -p "$HOME" "$ROS_HOME"
source /opt/ros/humble/setup.bash
source /opt/g1_localization_ws/install/setup.bash
exec "$@"
