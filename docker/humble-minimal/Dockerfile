ARG ROS_BASE_IMAGE=ros:humble-ros-base-jammy@sha256:afb40d6be65331c20a114d4e229a7ef099fed1b17bf6370daee193514b32aa16
FROM ${ROS_BASE_IMAGE}

ARG SUPERODOM_REVISION
ARG LIVOX_DRIVER_REVISION
ARG LIVOX_SDK2_REVISION
ARG GTSAM_REVISION
ARG SOPHUS_REVISION

LABEL org.opencontainers.image.title="TML SuperOdometry ROS 2 Humble"
LABEL org.opencontainers.image.description="Pinned CPU-only Mid-360 SuperOdometry runtime"
LABEL tml.superodom.upstream_revision="${SUPERODOM_REVISION}"
LABEL tml.superodom.livox_driver_revision="${LIVOX_DRIVER_REVISION}"
LABEL tml.superodom.livox_sdk2_revision="${LIVOX_SDK2_REVISION}"
LABEL tml.superodom.gtsam_revision="${GTSAM_REVISION}"
LABEL tml.superodom.sophus_revision="${SOPHUS_REVISION}"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      cmake \
      git \
      libapr1-dev \
      libboost-all-dev \
      libceres-dev \
      libeigen3-dev \
      libgoogle-glog-dev \
      libpcl-dev \
      libsuitesparse-dev \
      libtbb-dev \
      python3-colcon-common-extensions \
      ros-humble-ament-cmake-auto \
      ros-humble-cv-bridge \
      ros-humble-image-transport \
      ros-humble-pcl-conversions \
      ros-humble-rmw-cyclonedds-cpp \
      ros-humble-rmw-fastrtps-cpp \
      ros-humble-rosbag2 \
      ros-humble-rosidl-default-generators \
      ros-humble-tf2-eigen \
      ros-humble-tf2-geometry-msgs \
      ros-humble-tf2-ros \
      ros-humble-visualization-msgs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/source

RUN test -n "${LIVOX_SDK2_REVISION}" \
    && git clone --filter=blob:none https://github.com/Livox-SDK/Livox-SDK2.git \
    && git -C Livox-SDK2 checkout --detach "${LIVOX_SDK2_REVISION}" \
    && cmake -S Livox-SDK2 -B Livox-SDK2/build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build Livox-SDK2/build --parallel "$(nproc)" \
    && cmake --install Livox-SDK2/build \
    && rm -rf Livox-SDK2

RUN test -n "${SOPHUS_REVISION}" \
    && git clone --filter=blob:none https://github.com/strasdat/Sophus.git \
    && git -C Sophus checkout --detach "${SOPHUS_REVISION}" \
    && cmake -S Sophus -B Sophus/build \
         -DCMAKE_BUILD_TYPE=Release \
         -DBUILD_TESTS=OFF \
    && cmake --build Sophus/build --parallel "$(nproc)" \
    && cmake --install Sophus/build \
    && rm -rf Sophus

RUN test -n "${GTSAM_REVISION}" \
    && git clone --filter=blob:none https://github.com/borglab/gtsam.git \
    && git -C gtsam checkout --detach "${GTSAM_REVISION}" \
    && cmake -S gtsam -B gtsam/build \
         -DCMAKE_BUILD_TYPE=Release \
         -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
         -DGTSAM_BUILD_SHARED_LIBS=ON \
         -DGTSAM_BUILD_TESTS=OFF \
         -DGTSAM_BUILD_UNSTABLE=ON \
         -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF \
         -DGTSAM_INSTALL_CPPUNITLITE=OFF \
         -DGTSAM_USE_SYSTEM_EIGEN=ON \
    && cmake --build gtsam/build --parallel "$(nproc)" \
    && cmake --install gtsam/build \
    && rm -rf gtsam \
    && ldconfig

WORKDIR /opt/superodom_ws/src
RUN test -n "${LIVOX_DRIVER_REVISION}" \
    && git clone --filter=blob:none https://github.com/Livox-SDK/livox_ros_driver2.git \
    && git -C livox_ros_driver2 checkout --detach "${LIVOX_DRIVER_REVISION}" \
    && cp livox_ros_driver2/package_ROS2.xml livox_ros_driver2/package.xml \
    && cp -a livox_ros_driver2/launch_ROS2 livox_ros_driver2/launch

COPY super_odometry /opt/superodom_ws/src/super_odometry
COPY super_odometry_msgs /opt/superodom_ws/src/super_odometry_msgs

WORKDIR /opt/superodom_ws
RUN source /opt/ros/humble/setup.bash \
    && colcon build \
         --merge-install \
         --event-handlers console_direct+ \
         --packages-select livox_ros_driver2 super_odometry_msgs super_odometry \
         --cmake-args \
           -DCMAKE_BUILD_TYPE=Release \
           -DROS_EDITION=ROS2 \
           -DHUMBLE_ROS=humble \
    && source install/setup.bash \
    && ros2 pkg prefix livox_ros_driver2 \
    && ros2 pkg prefix super_odometry_msgs \
    && ros2 pkg prefix super_odometry \
    && rm -rf build log src/livox_ros_driver2/.git

COPY docker/humble-minimal/entrypoint.sh /usr/local/bin/superodom-entrypoint
COPY docker/humble-minimal/replay_smoke.sh /usr/local/bin/superodom-replay-smoke
RUN chmod 0755 /usr/local/bin/superodom-entrypoint /usr/local/bin/superodom-replay-smoke

ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
WORKDIR /output
ENTRYPOINT ["/usr/local/bin/superodom-entrypoint"]
CMD ["bash"]
