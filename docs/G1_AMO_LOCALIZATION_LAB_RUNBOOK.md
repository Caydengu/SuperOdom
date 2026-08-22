# Two-hour G1 walking-localization lab runbook

## Decision this session should enable

Collect enough frozen, synchronized evidence to decide whether the current
root-aware SuperOdometry pipeline generalizes to a new walk and whether a fresh
Field Bay structural map improves global localization. Preserve the exact
onboard FAST-LIO implementation so it can be compared offline on matched input.

The minimum successful session returns with:

1. one read-only snapshot of the robot-side localization source and configs;
2. the unchanged Motive rigid-body asset/profile plus its visible axes and
   physical relation to the G1 pelvis;
3. one passing stationary capture;
4. one slow map-coverage capture and one independent map-revisit capture;
5. one frozen AMO confirmation walk with explicit policy-start/stop events; and
6. room-layout photos, measurements, and checksums for every artifact.

## Robot selection

Use the sensor-equipped `G-2296` only if its right ankle has been repaired or a
current robot owner confirms it is safe for AMO walking. This preserves the
same MID-360, camera, extrinsics, onboard workspace, and Motive body 39
(`G1-PELVIS-2296`) used by walks 01 and 02.

The last retained diagnostic for G-2296 found severe right-ankle pitch/roll
under-response, so an unrepaired G-2296 is not admitted for walking. If that
condition is unresolved, do not substitute G-1323 silently. Either collect
stationary/supported calibration and map evidence only, or start a separately
named G-1323 campaign after verifying its sensor mount and defining a distinct
Motive rigid body and calibration.

## Questions to send Gio before lab

Ask Gio for the current provenance rather than assuming a particular external
mapping product. The two answers most likely to change today's plan are which
physical G1 is current and whether G-2296's right ankle is cleared for AMO.

Send this compact checklist:

1. Which physical G1 should I use for the next localization walk, and has
   G-2296's right ankle issue been repaired and cleared?
2. What exact workflow/tool produced the current Field Bay and SRC kitchen
   maps: onboard MID-360 with FAST-LIO/SuperOdometry, Polycam, another scanner,
   or a combination?
3. Where are the original map inputs and projects, not only the derived
   GLB/PCD? Please include the source recording or bag, generation command,
   config, branch/commit, date, units, coordinate origin/axes, voxel size, and
   any LiDAR-to-robot extrinsics.
4. Which robot-side localization workspace and configuration are current? Is
   `/home/unitree/geo-179/G1_localization/ws_slam` still authoritative, and
   which `utlidar.yaml`, FAST-LIO/SuperOdometry revision, time-unit setting,
   and launch command were used?
5. Is there newer or unpushed localization/map code or a newer room scan that
   we should preserve before collecting data?
6. Does he have a preferred room-rescan procedure, and can it export the raw
   point cloud/project in metric units in addition to any textured mesh?

Do not wait on these answers to preserve raw onboard MID-360, Livox IMU,
lowstate, RGB-D, and Motive streams. Those raw streams remain useful even if
Gio's preferred map-generation pipeline changes.

## Bring and preserve

- Oslo, its G1 Ethernet adapter, charger, and enough `/move` storage.
- Measuring tape plus preferably calipers; painter's tape; a marker; and a
  phone for photographs.
- Keep the room static: move people out of the scan, park movable chairs/carts,
  and do not rearrange the room between map and evaluation captures.
- Photograph the room from all corners and any furniture that may later move.
- Record two independent metric scale checks, such as wall-to-wall distance and
  door width.
- Photograph the robot serial label, MID-360 mount, pelvis marker cluster, and
  cable routing. Record any repair or mount change since walks 01/02.

## 0:00-0:15 — physical and non-actuating preflight

From Oslo:

```bash
cd /move/u/caydengu/cayden/holosoma_secret/holosoma
scripts/doctor_holosoma.sh

ip -br addr
ip route get 192.168.123.164
ping -I 192.168.123.11 -c 2 192.168.123.164
docker inspect holosoma --format 'NetworkMode={{.HostConfig.NetworkMode}}'
```

Run the non-actuating depth/network/clock health check inside the Holosoma
container after replacing `<robot-nic>` with the direct-route interface:

```bash
cd /workspace/holosoma_secret/holosoma
source scripts/source_inference_setup.sh
python3 -m holosoma_inference.g1_health_check \
  --host 192.168.123.164 \
  --expected-source 192.168.123.11 \
  --expected-interface <robot-nic> \
  --ping-source 192.168.123.11 \
  --depth-port 55559 --height 480 --width 640 --frames 60 \
  --out-dir data/g1_health
```

Do not walk if the robot owner cannot confirm G-2296's ankle repair, the robot
is not physically supported/clear, the direct network route is absent, or the
non-actuating health check fails.

## 0:15-0:30 — preserve the exact robot stack and Motive asset

Snapshot all candidate robot-side locations without writing to the G1:

```bash
cd /move/u/caydengu/cayden/.worktrees/superodom-g1-motive-amo-localization-eval
scripts/snapshot_g1_localization_stack.sh \
  --output-dir /move/u/caydengu/cayden/research/perceptive-humanoid-diffusion/runs/2026-08-22_g2296-localization-stack-snapshot
```

In Motive, before modifying anything:

1. Select `G1-PELVIS-2296`; verify Streaming ID `39`.
2. Enable Bones/Bone Orientation and Marker Constraints.
3. Screenshot the Properties, Info, and 3D views with the G1 facing a known
   room direction. Record which local axis points robot-forward, left, and up.
4. Use **File -> Export Assets** and also **Export Profile As**, including
   Assets. Copy the `.motive` file and the relevant Take/session to the run
   folder after capture.
5. Do not reset or rotate the ID-39 pivot before the frozen walk. If a cleaner
   pelvis-aligned asset is desired, make a versioned duplicate with a new
   Streaming ID after the frozen walk.

Measure and photograph the relation between the displayed rigid-body pivot and
the G1 `pelvis` URDF origin. The pelvis origin is centered between the left and
right hip-pitch joint origins; those origins are at lateral `+/-64.452 mm` and
`102.7 mm` below the pelvis origin. Record translation in robot coordinates
(`+x` forward, `+y` left, `+z` up), measurement uncertainty, and the exact
physical surfaces used. A ruler visible in each photograph is better than a
verbal estimate.

## 0:30-0:40 — stationary canary

Keep the G1 upright and still for the entire 30 seconds:

```bash
cd /move/u/caydengu/cayden/.worktrees/superodom-g1-motive-amo-localization-eval
scripts/run_g1_motive_dataset.sh \
  --run-dir /move/u/caydengu/cayden/research/perceptive-humanoid-diffusion/runs/2026-08-22_g1-motive-stationary-03 \
  --duration-sec 30 \
  --rigid-body-id 39 \
  --rigid-body-name G1-PELVIS-2296 \
  --motive-server 172.24.68.77 \
  --motive-connection multicast
```

Require Motive tracking coverage at least 95%, nonempty LiDAR and Livox IMU,
lowstate at least 200 Hz with zero invalid packets and zero sequence gaps. RGB-D
is useful context but is not required by the current LiDAR/IMU/FK localization
core. Stop and repair capture if a localization-critical stream fails.

## 0:40-0:55 — calibration motion capture

Collect a separate 120-second calibration trajectory. Begin and end with 10
seconds stationary. Include slow straight translation, left and right turns,
at least three distinct headings, and pauses. Do not tune the estimator against
this run; it is for clock, axis, IMU-sign, and hand-eye calibration.

Use the same capture command with run name
`2026-08-22_g1-motive-pelvis-calibration-01` and `--duration-sec 120`. Mark
events from another terminal:

```bash
RUN=/move/u/caydengu/cayden/research/perceptive-humanoid-diffusion/runs/2026-08-22_g1-motive-pelvis-calibration-01
python3 scripts/mark_g1_motive_event.py --run-dir "$RUN" --label policy_enable
python3 scripts/mark_g1_motive_event.py --run-dir "$RUN" --label motion_begin
python3 scripts/mark_g1_motive_event.py --run-dir "$RUN" --label motion_end
python3 scripts/mark_g1_motive_event.py --run-dir "$RUN" --label policy_disable
```

## 0:55-1:20 — fresh Field Bay map captures

The primary map source should be the same onboard MID-360, not an external
mesh. Motive can later place scans in a drift-free reference frame inside its
coverage; FAST-LIO and SuperOdometry can also be replayed offline from the same
raw bag. Do not run an online localization estimator during raw acquisition.

Collect two non-overlapping artifacts:

- `2026-08-22_field-bay-map-coverage-01`, 480 seconds: slow perimeter and
  interior loops, both clockwise and counter-clockwise, multiple headings at
  corners, and 10 seconds static at each end.
- `2026-08-22_field-bay-map-revisit-01`, 240 seconds: a different route through
  the same room, kept separate for map quality/registration validation.

Use `run_g1_motive_dataset.sh` with the same Motive arguments and the respective
run name/duration. Move slowly, avoid abrupt turns, keep people out of the
LiDAR field, observe walls/corners from more than one angle, and include the
feature-poor region where localization is expected to be hardest. Do not move
furniture between the two passes.

If Gio identifies an external scanning workflow, reproduce that exact workflow
as a matched secondary baseline rather than introducing a new scanner during a
time-limited lab session. It is useful as a collision/visual asset, but it is
not a substitute for the raw MID-360 captures. Export the original project,
metric `.e57` or `.ply` point cloud, and textured `.glb`/`.obj`; record the
app/device/version, units, axis convention, scale checks, and photos of the
chosen map origin. Never preserve only a decimated GLB.

## 1:20-1:40 — frozen confirmation walk

Use exactly the same AMO policy, checkpoint, control rate, and launch command as
walks 01/02. Do not change localization parameters or use Motive online.
Before launch, save the exact command, policy checkpoint path and SHA256, source
branch/commit/status, controller/interface name, and operator command sequence
to `lab_metadata/policy_provenance.txt`.

Capture `2026-08-22_g1-motive-amo-walk-03` for 180 seconds. Suggested trajectory:

1. 15 seconds stationary on the gantry before enabling the policy;
2. straight segment, stop, and restart;
3. left and right 90-degree turns;
4. a curved or figure-eight segment;
5. traverse both feature-rich and feature-poor room regions;
6. return near the start and remain stationary for the final 10 seconds.

Mark `policy_enable`, `walk_begin`, `walk_end`, and `policy_disable` with
`mark_g1_motive_event.py`. Motive remains evaluator-only. If tracking falls
below 95%, lowstate has a material gap, or LiDAR/IMU is missing, preserve the
failed run and use the remaining time for one fresh non-overwriting retry.

If time and robot safety allow, collect a second independently named
`walk-04-stress` emphasizing repeated turns and the feature-poor wall. Keep it
sealed until walk-03 scoring and interpretation are complete.

## 1:40-2:00 — validation and departure gate

For every run:

```bash
python3 scripts/validate_g1_motive_dataset.py \
  --run-dir <run-dir> --output <run-dir>/validation.json
du -sh <run-dir>
find <run-dir> -type f -print0 | sort -z | xargs -0 sha256sum > <run-dir>/SHA256SUMS
```

Before leaving, confirm each directory contains:

- `manifest.json`, `clock_probe.json`, `validation.json`, and logs;
- `motive/frames.jsonl` and `motive/summary.json`;
- `lowstate/packets.bin` and `lowstate/summary.json`;
- the ROS bag under `lidar/data/live_input_probe`;
- `events/operator_events.jsonl` for moving runs;
- RGB/depth files when the recorder succeeded;
- Motive `.motive` asset/profile, relevant Take/session, room photos,
  measurements, and notes copied into a sibling `lab_metadata/` directory; and
- `lab_metadata/policy_provenance.txt`, robot/sensor photographs, robot serial,
  and any current hardware-repair or sensor-mount notes; and
- the robot software snapshot and its `SHA256SUMS`.

Leave raw data immutable. Do not rename runs after capture, edit the Motive
trajectory, downsample the only point cloud, or overwrite the old Field Bay
map. New derived maps will receive a new ID, date, source-run list, coordinate
contract, voxel size, and checksum offline.

## Outcome map

- New walk passes with the frozen pipeline: proceed to calibrated absolute
  pelvis scoring and live shadow integration.
- Local odometry passes but fresh-map correction fails: inspect map frame,
  overlap, degeneracy, and dynamic-object filtering; keep the fast local lane.
- Both SuperOdometry and recovered FAST-LIO fail on the same segments: focus on
  timing, deskew, extrinsics, vibration, or sensor calibration.
- FAST-LIO wins on matched raw input: replace the local LIO core while retaining
  dynamic FK, pelvis gravity, map correction, and health contracts.
- Only the external scan fails while the MID-360 map works: retain the external
  mesh for visualization/collision, not automatic localization.
- Robot or critical-stream preflight fails: no walking evidence is collected;
  preserve diagnostics and use the session for calibration and software/map
  provenance only.
