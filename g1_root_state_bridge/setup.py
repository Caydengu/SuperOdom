from setuptools import find_packages, setup


package_name = "g1_root_state_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test", "tests")),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    package_data={
        package_name: [
            "assets/*.urdf",
            "config/*.json",
        ]
    },
    install_requires=["setuptools", "numpy", "pyzmq", "kiss-icp==1.3.0"],
    zip_safe=False,
    maintainer="Cayden Gu",
    maintainer_email="caydengu@stanford.edu",
    description="Typed G1 LiDAR-inertial local-odometry and pelvis-state bridge.",
    license="BSD-3-Clause",
    entry_points={
        "console_scripts": [
            "g1-dynamic-capture-relay = "
            "g1_root_state_bridge.g1_dynamic_capture_relay:main",
            "g1-dynamic-capture-recorder = "
            "g1_root_state_bridge.g1_dynamic_capture_recorder:main",
            "g1-kiss-localization-replay = "
            "g1_root_state_bridge.replay_cli:main",
            "g1-kiss-live-localization = "
            "g1_root_state_bridge.live_node:main",
            "g1-kiss-compare-replay = "
            "g1_root_state_bridge.compare_replay_cli:main",
            "g1-kiss-fault-matrix = "
            "g1_root_state_bridge.fault_matrix_cli:main",
            "g1-kiss-clock-replay = "
            "g1_root_state_bridge.clock_replay_cli:main",
        ],
    },
)
