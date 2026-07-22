# Minimal Humble SuperOdometry Container Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pinned, CPU-only ROS 2 Humble image on Oslo that compiles the current humanoid Mid-360 SuperOdometry branch, reproduces a timestamp-valid legacy bag replay, and remains ready for a later measured Foxy-to-Humble live-shadow boundary.

**Architecture:** Keep the G1 and its ROS 2 Foxy installation unchanged. Build SuperOdometry, its message package, Livox ROS Driver 2 message support, Livox SDK2, Sophus, and GTSAM into one Humble image. Run it without GPU, GUI, X11, or `--privileged`; mount raw bags read-only and outputs separately. Preserve the legacy algorithm changes only where they are explicit and testable, keep the normal 0.5 m odometry range as the default, and provide the legacy 2.0 m gantry filter as a named replay profile.

**Tech Stack:** Docker/BuildKit, Ubuntu 22.04, ROS 2 Humble, C++17, colcon, GTSAM, Ceres, PCL, Livox SDK2, Livox ROS Driver 2, Bash, pytest.

## Global Constraints

- SuperOdometry source is pinned to `57a6e233c348372485b7c64ae8238d53c1c7c2ad` before local changes.
- Base image is `ros:humble-ros-base-jammy@sha256:afb40d6be65331c20a114d4e229a7ef099fed1b17bf6370daee193514b32aa16`.
- Livox ROS Driver 2 is pinned to `6b9356cadf77084619ba406e6a0eb41163b08039`.
- Livox SDK2 is pinned to `6a940156dd7151c3ab6a52442d86bc83613bd11b`.
- GTSAM is pinned to `4abef9248edc4c49943d8fd8a84c028deb486f4c`.
- Sophus is pinned to `97e71617749a32b83cdd411591ebb2ede9d330f0`.
- `/move/u/caydengu/superodom_legacy_drop` is immutable reference evidence; never edit it.
- Do not copy the legacy Dockerfile, hard-coded laptop paths, GPU flags, X11 setup, GUI packages, or unpinned Git clones.
- Default runtime middleware is CycloneDDS; middleware selection remains configurable for replay comparison.
- The image must not require NVIDIA Container Runtime, a GPU, X11, `--privileged`, or host IPC.
- Default `min_range` remains `0.5`; legacy gantry replay uses an explicit `2.0` profile.
- Do not make either legacy calibration file a new empirical calibration claim.
- Replay must explicitly pass `use_sim_time:=true`; live launch defaults to `false`.
- Output topic is `/state_estimation`.
- Scope excludes live G1 access, direct Foxy-to-Humble DDS validation, root FK, elevation mapping, policy inference, and actuation.

---

### Task 1: Freeze the executable contract in failing tests

**Files:**
- Create: `tests/test_minimal_container_contract.py`
- Create: `scripts/test_minimal_container.sh`

**Interfaces:**
- Consumes: the upstream repository at the pinned source revision.
- Produces: one pytest suite that specifies source patches, dependency pins, image restrictions, wrapper behavior, and replay safety.

- [ ] **Step 1: Write the failing contract tests**

Create `tests/test_minimal_container_contract.py` with tests that:

```python
from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_source_contract_uses_real_output_and_safe_profiles() -> None:
    config = read("super_odometry/config/livox_mid360.yaml")
    gantry = read("super_odometry/config/livox_mid360_gantry.yaml")
    parameters = read("super_odometry/src/parameter/parameter.cpp")
    preintegration = read(
        "super_odometry/src/ImuPreintegration/imuPreintegration_current.cpp"
    )

    assert 'odom_topic: "/state_estimation"' in config
    assert "min_range: 0.5" in config
    assert "min_range: 2.0" in gantry
    assert 'declare_parameter<bool>("use_imu_roll_pitch", false)' in parameters
    assert 'get_parameter("use_imu_roll_pitch").as_bool()' in parameters
    assert "g_world_dir.x() * config_.imuGravity" in preintegration


def test_replay_time_is_explicit_and_live_default_is_safe() -> None:
    launch = read("super_odometry/launch/livox_humanoid.launch.py")
    replay = read("docker/humble-minimal/replay_smoke.sh")

    assert 'DeclareLaunchArgument("use_sim_time"' in launch
    assert 'default_value="false"' in launch
    assert '"use_sim_time": use_sim_time' in launch
    assert "use_sim_time:=true" in replay


def test_dependency_lock_contains_full_revisions() -> None:
    lock = read("docker/humble-minimal/dependency-lock.env")
    expected = {
        "ROS_BASE_IMAGE": "ros:humble-ros-base-jammy@sha256:afb40d6be65331c20a114d4e229a7ef099fed1b17bf6370daee193514b32aa16",
        "SUPERODOM_REVISION": "57a6e233c348372485b7c64ae8238d53c1c7c2ad",
        "LIVOX_DRIVER_REVISION": "6b9356cadf77084619ba406e6a0eb41163b08039",
        "LIVOX_SDK2_REVISION": "6a940156dd7151c3ab6a52442d86bc83613bd11b",
        "GTSAM_REVISION": "4abef9248edc4c49943d8fd8a84c028deb486f4c",
        "SOPHUS_REVISION": "97e71617749a32b83cdd411591ebb2ede9d330f0",
    }
    for key, value in expected.items():
        assert f"{key}={value}" in lock


def test_dockerfile_is_cpu_only_and_pinned() -> None:
    dockerfile = read("docker/humble-minimal/Dockerfile")
    lowered = dockerfile.lower()

    assert "humble-ros-base-jammy" in dockerfile
    assert "rmw-cyclonedds-cpp" in dockerfile
    assert "--packages-select" in dockerfile
    for forbidden in ("desktop-full", "nvidia", "rviz", "plotjuggler", "x11"):
        assert forbidden not in lowered


def test_runtime_wrapper_drops_privilege_and_gpu_requirements(tmp_path: Path) -> None:
    data = tmp_path / "data"
    output = tmp_path / "output"
    data.mkdir()
    output.mkdir()
    completed = subprocess.run(
        [
            str(ROOT / "docker/humble-minimal/run.sh"),
            "--dry-run",
            "--data-dir",
            str(data),
            "--output-dir",
            str(output),
            "--",
            "ros2",
            "pkg",
            "list",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
        env={**os.environ, "SUPERODOM_IMAGE": "test/superodom:contract"},
    )
    command = completed.stdout
    assert "--network host" in command
    assert "readonly" in command
    assert "--cap-drop ALL" in command
    assert "RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" in command
    assert "--privileged" not in command
    assert "--gpus" not in command
    assert "--runtime=nvidia" not in command


def test_shell_scripts_are_syntactically_valid() -> None:
    scripts = [
        ROOT / "docker/humble-minimal/build_image.sh",
        ROOT / "docker/humble-minimal/run.sh",
        ROOT / "docker/humble-minimal/entrypoint.sh",
        ROOT / "docker/humble-minimal/replay_smoke.sh",
        ROOT / "scripts/test_minimal_container.sh",
    ]
    for script in scripts:
        subprocess.run(["bash", "-n", str(script)], check=True)
```

Create `scripts/test_minimal_container.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$repo_root"
python3 -m pytest -q tests/test_minimal_container_contract.py
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
bash scripts/test_minimal_container.sh
```

Expected: failures for the missing gantry profile, dependency lock, Dockerfile, runtime scripts, and upstream source behavior.

- [ ] **Step 3: Commit the executable contract**

```bash
git add tests/test_minimal_container_contract.py scripts/test_minimal_container.sh
git commit -m "test: define minimal Humble container contract"
```

### Task 2: Port only the explicit legacy-compatible SuperOdometry fixes

**Files:**
- Modify: `super_odometry/config/livox_mid360.yaml`
- Create: `super_odometry/config/livox_mid360_gantry.yaml`
- Modify: `super_odometry/launch/livox_humanoid.launch.py`
- Modify: `super_odometry/src/ImuPreintegration/imuPreintegration_current.cpp`
- Modify: `super_odometry/src/parameter/parameter.cpp`

**Interfaces:**
- Consumes: legacy diff from `/move/u/caydengu/superodom_legacy_drop/source/SuperOdom`, used only as evidence.
- Produces: safe live defaults, explicit replay-time control, a named gantry profile, and working parameter/gravity logic.

- [ ] **Step 1: Patch the default configuration**

Change `odom_topic` to `/state_estimation`, set acceleration limits to `5.0`, explain why `use_imu_roll_pitch` remains false, and retain `min_range: 0.5`.

- [ ] **Step 2: Add the named gantry replay profile**

Copy the complete Mid-360 YAML into `livox_mid360_gantry.yaml` and change only `min_range` to `2.0`, with a comment that the profile is for legacy gantry regression and must not feed the terrain mapper.

- [ ] **Step 3: Make replay time explicit per node**

Add a Boolean `use_sim_time` launch argument with `default_value="false"`, wrap it in `ParameterValue`, and pass it to all three nodes through their `parameters` lists. Remove the global hard-coded `SetParameter` action.

- [ ] **Step 4: Restore the computed gravity vector and parameter loading**

Set `n_gravity` from all three components of `g_world_dir * imuGravity`, declare/load `use_imu_roll_pitch`, and align C++ acceleration defaults with the YAML.

- [ ] **Step 5: Run the focused source tests and verify GREEN**

Run:

```bash
python3 -m pytest -q \
  tests/test_minimal_container_contract.py::test_source_contract_uses_real_output_and_safe_profiles \
  tests/test_minimal_container_contract.py::test_replay_time_is_explicit_and_live_default_is_safe
```

Expected: the source test passes; replay test may still fail only because `replay_smoke.sh` has not yet been created.

- [ ] **Step 6: Commit the source contract**

```bash
git add super_odometry/config super_odometry/launch/livox_humanoid.launch.py \
  super_odometry/src/ImuPreintegration/imuPreintegration_current.cpp \
  super_odometry/src/parameter/parameter.cpp
git commit -m "fix: make humanoid Mid-360 replay explicit"
```

### Task 3: Build the pinned CPU-only Humble image

**Files:**
- Create: `.dockerignore`
- Create: `docker/humble-minimal/dependency-lock.env`
- Create: `docker/humble-minimal/Dockerfile`
- Create: `docker/humble-minimal/entrypoint.sh`
- Create: `docker/humble-minimal/build_image.sh`

**Interfaces:**
- Consumes: the pinned revisions in `dependency-lock.env` and local SuperOdometry packages.
- Produces: image `tml/superodom-humble:57a6e23-minimal` with `/opt/superodom_ws/install/setup.bash`.

- [ ] **Step 1: Add the dependency lock**

Create `dependency-lock.env` with exactly the six assignments listed in Global Constraints.

- [ ] **Step 2: Add a minimal Docker build context**

Create `.dockerignore` excluding `.git`, build/install/log trees, pytest caches, videos, bags, database files, and generated outputs while retaining `super_odometry`, `super_odometry_msgs`, and `docker/humble-minimal`.

- [ ] **Step 3: Add the entrypoint**

Create an entrypoint that sources `/opt/ros/humble/setup.bash`, requires `/opt/superodom_ws/install/setup.bash`, sources it, and `exec`s the supplied command.

- [ ] **Step 4: Add the Dockerfile**

The Dockerfile must:

1. accept the pinned base image and revisions as build arguments;
2. install only compiler, numerical, PCL/Ceres, ROS message, rosbag2, and CycloneDDS dependencies using `--no-install-recommends`;
3. clone and checkout Livox SDK2, Sophus, GTSAM, and Livox ROS Driver 2 at full revisions;
4. copy only the two local ROS packages and runtime scripts;
5. build `livox_ros_driver2`, `super_odometry_msgs`, and `super_odometry` with colcon in Release mode;
6. run `ros2 pkg prefix` checks during the build;
7. expose revision labels and use the minimal entrypoint.

- [ ] **Step 5: Add the host build wrapper**

`build_image.sh` sources `dependency-lock.env`, passes every revision as a build argument, defaults to `tml/superodom-humble:57a6e23-minimal`, and supports an optional tag as its only positional argument.

- [ ] **Step 6: Run focused static tests and verify GREEN**

Run:

```bash
python3 -m pytest -q \
  tests/test_minimal_container_contract.py::test_dependency_lock_contains_full_revisions \
  tests/test_minimal_container_contract.py::test_dockerfile_is_cpu_only_and_pinned
```

Expected: two passing tests.

- [ ] **Step 7: Commit the image definition**

```bash
git add .dockerignore docker/humble-minimal
git commit -m "build: add pinned CPU-only Humble image"
```

### Task 4: Add a safe host wrapper and deterministic replay driver

**Files:**
- Create: `docker/humble-minimal/run.sh`
- Create: `docker/humble-minimal/replay_smoke.sh`

**Interfaces:**
- `run.sh --data-dir PATH --output-dir PATH [--rmw cyclonedds|fastrtps] [--dry-run] -- COMMAND...`
- `replay_smoke.sh --bag /data/BAG --output /output/NAME [--config PATH] [--rate FLOAT]`
- Produces: a read-only input mount at `/data`, writable output mount at `/output`, and a replay bag containing `/state_estimation`.

- [ ] **Step 1: Implement `run.sh` minimally**

The script resolves both paths, rejects missing directories, maps `cyclonedds` to `rmw_cyclonedds_cpp` and `fastrtps` to `rmw_fastrtps_cpp`, uses host networking, drops all capabilities, sets `no-new-privileges`, runs as the caller UID/GID, mounts `/data` read-only, mounts `/output` read-write, and prints a shell-escaped command under `--dry-run`.

- [ ] **Step 2: Run the wrapper test and verify GREEN**

Run:

```bash
python3 -m pytest -q \
  tests/test_minimal_container_contract.py::test_runtime_wrapper_drops_privilege_and_gpu_requirements
```

Expected: one passing test.

- [ ] **Step 3: Implement `replay_smoke.sh` with fail-closed process handling**

The script must refuse an existing output path, install an EXIT trap, launch SuperOdometry with `use_sim_time:=true`, wait up to 30 seconds for the three named nodes, start a sim-time recorder for `/state_estimation`, play the input with `--clock`, stop recorder and nodes with SIGINT, and fail unless `ros2 bag info` reports at least one output message.

- [ ] **Step 4: Run the full static contract suite and verify GREEN**

Run:

```bash
bash scripts/test_minimal_container.sh
```

Expected: all contract tests pass.

- [ ] **Step 5: Commit the runtime and replay wrappers**

```bash
git add docker/humble-minimal/run.sh docker/humble-minimal/replay_smoke.sh
git commit -m "feat: add safe SuperOdometry replay wrapper"
```

### Task 5: Build and validate the container without the G1

**Files:**
- Create outside code repo: `research/perceptive-humanoid-diffusion/runs/2026-07-22_superodom-minimal-humble-container/manifest.json`
- Create outside code repo: `research/perceptive-humanoid-diffusion/runs/2026-07-22_superodom-minimal-humble-container/command.txt`
- Create outside code repo: `research/perceptive-humanoid-diffusion/runs/2026-07-22_superodom-minimal-humble-container/metrics.csv`
- Create outside code repo: `research/perceptive-humanoid-diffusion/runs/2026-07-22_superodom-minimal-humble-container/summary.md`
- Create outside code repo: `research/perceptive-humanoid-diffusion/runs/2026-07-22_superodom-minimal-humble-container/logs/`

**Interfaces:**
- Consumes: legacy raw bag `/move/u/caydengu/superodom_legacy_drop/bags/known_good/g1_walk_20260413_080726` read-only.
- Produces: a built image, package-health output, a new `/state_estimation` bag, build/replay logs, and timestamp/count metrics. This is pipeline evidence only, not motion-accuracy evidence.

- [ ] **Step 1: Initialize the artifact run before executable validation**

Run `scripts/research_manifest.py init` from the Cayden root with the implementation plan as `--source-path`, no research-map handoff, and the manual boundary “Offline Docker build and legacy replay only; no live G1, policy, mapping, or actuation.”

- [ ] **Step 2: Build the image and retain the full log**

Run:

```bash
docker/humble-minimal/build_image.sh \
  tml/superodom-humble:57a6e23-minimal \
  2>&1 | tee <run-dir>/logs/docker-build.log
```

Expected: exit code zero and all three ROS packages found by the in-image prefix checks.

- [ ] **Step 3: Run image health checks**

Run the image with:

```bash
ros2 pkg prefix livox_ros_driver2
ros2 pkg prefix super_odometry_msgs
ros2 pkg prefix super_odometry
ldd /opt/superodom_ws/install/lib/super_odometry/laser_mapping_node
```

Expected: package prefixes under `/opt/superodom_ws/install` and no missing shared libraries.

- [ ] **Step 4: Replay the transferred legacy bag**

Use the explicit gantry profile and CycloneDDS. Mount the legacy `bags` directory read-only and the run `data` directory read-write. Capture replay logs under the run directory.

- [ ] **Step 5: Extract validation metrics**

Record at minimum:

- image ID and compressed/virtual size;
- package prefix checks;
- input `/livox/lidar` and `/livox/imu` message counts;
- output `/state_estimation` message count and duration;
- first/last input and output CDR header timestamps;
- difference between output header time and input sensor time;
- whether all three nodes used sim time;
- whether any process exited unexpectedly or accumulated a backlog.

- [ ] **Step 6: Interpret against the predeclared outcome map**

- A: build, package checks, sim time, and nonempty output pass — promote the image to the live-shadow candidate.
- B: build passes but replay fails — classify source/config/runtime failure before changing architecture.
- C: replay is timestamp-valid but differs from legacy motion envelope — retain pipeline success and schedule a matched middleware/config separator; do not call it odometry failure without motion ground truth.
- D: legacy output itself cannot provide a valid comparison — retain build evidence and use a new measured fixture bag later.

- [ ] **Step 7: Validate the artifact directory**

Run:

```bash
python3 scripts/research_manifest.py validate --run-dir <run-dir>
```

Expected: valid manifest. A missing-figure warning is acceptable only if `summary.md` says that a container build/replay contract has no decision-improving visual until a trajectory comparison is valid.

### Task 6: Document the runnable handoff

**Files:**
- Create: `docker/humble-minimal/README.md`
- Modify: `readme.md`

**Interfaces:**
- Consumes: verified commands and metrics from Task 5.
- Produces: exact build, shell, replay, middleware, mount, and live-shadow-next-step commands.

- [ ] **Step 1: Write the minimal-container README**

Document dependency pins, why the legacy Dockerfile was not reused, read-only legacy provenance, build command, interactive command, replay command, output locations, middleware selection, expected topics, and the distinction between pipeline parity and motion accuracy.

- [ ] **Step 2: Add one discoverability link to the upstream README**

Add a short “TML minimal Humble container” section linking to `docker/humble-minimal/README.md`; do not rewrite the upstream project documentation.

- [ ] **Step 3: Run fresh verification before completion**

Run:

```bash
bash scripts/test_minimal_container.sh
docker image inspect tml/superodom-humble:57a6e23-minimal
docker/humble-minimal/run.sh --data-dir <legacy-bags> --output-dir <run-data> -- \
  ros2 pkg prefix super_odometry
python3 /move/u/caydengu/cayden/scripts/research_manifest.py validate --run-dir <run-dir>
git status --short
git diff --check
```

Expected: tests pass, image exists, package prefix resolves, artifact manifest validates, and only intended repository files are modified.

- [ ] **Step 4: Commit the verified documentation**

```bash
git add docker/humble-minimal/README.md readme.md
git commit -m "docs: document minimal Humble SuperOdometry workflow"
```

## Self-Review

- Spec coverage: dependency provenance, CPU-only execution, safe defaults, replay timing, immutable legacy input, output topic, middleware selection, artifact evidence, and future live-shadow boundary each map to a task.
- Placeholder scan: angle-bracket paths occur only in operator commands whose value is produced by the immediately preceding task; no implementation behavior is left undefined.
- Type/interface consistency: both host wrappers use `/data` as read-only input and `/output` as writable output; replay always publishes `/state_estimation`; dependency names match the lock and Docker build arguments.
- Scope: direct Foxy/Humble live transport, root FK, mapping, and policy integration are deliberately separate successor gates.
