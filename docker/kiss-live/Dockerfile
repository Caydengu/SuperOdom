ARG ROS_BASE_IMAGE=ros:humble-ros-base-jammy@sha256:afb40d6be65331c20a114d4e229a7ef099fed1b17bf6370daee193514b32aa16
FROM ${ROS_BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3-colcon-common-extensions \
      python3-pip \
      ros-humble-rmw-cyclonedds-cpp \
      ros-humble-rosbag2 \
    && rm -rf /var/lib/apt/lists/*

COPY docker/kiss-live/requirements.lock.txt /tmp/requirements.lock.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.lock.txt \
    && python3 - <<'PY'
import importlib.metadata
assert importlib.metadata.version("kiss-icp") == "1.3.0"
PY

WORKDIR /opt/g1_localization_ws/src
COPY g1_root_state_bridge /opt/g1_localization_ws/src/g1_root_state_bridge
WORKDIR /opt/g1_localization_ws
RUN source /opt/ros/humble/setup.bash \
    && colcon build --merge-install --event-handlers console_direct+ \
         --packages-select g1_root_state_bridge \
    && source install/setup.bash \
    && ros2 pkg prefix g1_root_state_bridge

COPY docker/kiss-live/entrypoint.sh /usr/local/bin/g1-localization-entrypoint
RUN chmod 0755 /usr/local/bin/g1-localization-entrypoint

ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ENV HOME=/tmp/home
ENV ROS_HOME=/tmp/ros
ENV PYTHONDONTWRITEBYTECODE=1
ENTRYPOINT ["/usr/local/bin/g1-localization-entrypoint"]
CMD ["bash"]
