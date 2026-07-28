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
    install_requires=["setuptools"],
    zip_safe=False,
    maintainer="Cayden Gu",
    maintainer_email="caydengu@stanford.edu",
    description="Typed SuperOdometry-to-Unitree-G1 pelvis state bridge.",
    license="BSD-3-Clause",
    entry_points={
        "console_scripts": [
            "g1-root-state-bridge = g1_root_state_bridge.bridge_node:main",
            "g1-dynamic-reference-score = "
            "g1_root_state_bridge.dynamic_reference_cli:main",
            "g1-optitrack-normalize = "
            "g1_root_state_bridge.optitrack_reference_cli:main",
            "g1-optitrack-clock-audit = "
            "g1_root_state_bridge.optitrack_clock_audit_cli:main",
            "g1-policy-state-replay = "
            "g1_root_state_bridge.policy_state_trace_cli:main",
            "g1-dynamic-capture-relay = "
            "g1_root_state_bridge.g1_dynamic_capture_relay:main",
            "g1-dynamic-capture-recorder = "
            "g1_root_state_bridge.g1_dynamic_capture_recorder:main",
        ],
    },
)
