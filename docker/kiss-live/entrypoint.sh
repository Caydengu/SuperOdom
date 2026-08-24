#!/usr/bin/env bash
set -eo pipefail
mkdir -p "$HOME" "$ROS_HOME"

# ROS 2 generated setup files probe optional trace variables without guarding
# them for Bash nounset.  Source both environments with nounset disabled, then
# restore strict argument checking for the runtime command.
set +u
source /opt/ros/humble/setup.bash
source /opt/g1_localization_ws/install/setup.bash
export PATH="/opt/g1_localization_ws/install/lib/g1_root_state_bridge:$PATH"
set -u
exec "$@"
