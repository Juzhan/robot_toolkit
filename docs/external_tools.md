# External perception and planning tools

The toolkit owns contracts, clients and a small catalogue. External processes own
GPU runtimes and models. Importing this module, constructing clients, loading
configuration, and listing tools never start services or download anything.
No robot connection, camera read or motion is performed by these APIs.

## Current readiness

| Tool | Toolkit status | Still required |
| --- | --- | --- |
| SAM3 | ENPIRE HTTP client/protocol tests; verified ModelScope weights; real model inference on 14 rendered RGB frames (2026-10-27), with misses and semantic false positives | Dedicated GPU environment, running service, real-camera and closed-loop acceptance |
| BundleSDF | Fixed-camera push/start/read/end client; offline protocol tests | Compatible external runtime/weights, source-frame provenance, tracking and reconstruction/export acceptance |
| AnyGrasp | RGB-D/mask candidate client; offline protocol tests | Licensed SDK/checkpoint, grasp-to-TCP calibration and full-scene collision validation |
| cuRobo | Catalogue entry only | Compatible CUDA environment, robot/tool geometry, collision scene and planner adapter |

These are not four installed, inference-validated tools. The catalogue pins
ENPIRE's adapter source to `99ee90acf65b5b18957c8382ad580db999528be3` and records the
referenced cuRobo version. It does not pin uninstalled weights or transitive CUDA
dependencies. Before deployment, record the actual source commit, dependency
lock, model snapshot revision and artifact checksums in the external runtime.

## Additional reconstruction candidate

[RaySt3R](https://github.com/Duisterhof/rayst3r), inspected at
`2c93f81b696909695c6dd5367b81753225b7254e`, is a candidate for object completion,
not an installed service or a replacement for persistent object tracking.
Its evaluation interface takes RGB, a binary object mask, depth, camera
intrinsics and a camera-to-world transform. It predicts novel-view depth maps
and exports a colored point cloud, with optional TSDF fusion. Predicted hidden
surfaces must remain distinguishable from observed RGB-D surfaces.

Its depth PNG encoding is uint16 normalized over 0-10 metres, NOT the toolkit's
uint16 millimetres; integration must explicitly convert and test that boundary.
The upstream license restricts use to noncommercial internal research and has
additional modification/distribution conditions. On 2026-10-28, the official
unmodified CLI and `eval_scene` API ran on 77 independently exported simulated
RGB-D/mask inputs. All produced readable nonempty point clouds. Runtime and
weights live outside this toolkit under `~/.local/share/robot-toolkit-tools/rayst3r/`.
Default `n_pred_views=5` exceeded available RTX 5080 memory; the official option
`n_pred_views=3` succeeded (other inference defaults unchanged), with peak PyTorch
allocation 5.50 GiB and warm median inference 0.634 seconds. This establishes
inference execution, not reconstruction accuracy, completeness or multi-frame
fusion. Evidence: `rlinf-workspace/outputs/piper_perception/hemisphere_scan_02/`
`rayst3r_inputs/inference_report.json` in the parent workspace.

## Files and lifecycle

Default external root: `~/.local/share/robot-toolkit-tools`, overridable using
`ROBOT_TOOLKIT_TOOLS_HOME` or `--root`. Nothing creates this directory implicitly.

```text
~/.local/share/robot-toolkit-tools/
  sam3/
    source/                 # independent ENPIRE checkout at the catalogue revision
    environment/            # isolated Python environment, not the robot environment
    models/                 # optional artifact records; not automatically loaded
    cache/huggingface/hub/   # pinned, pre-staged facebook/sam3 HF snapshot
  bundlesdf/                # separate source/environment/models/cache
  anygrasp/
  curobo/
~/.config/robot-toolkit/tools.json   # machine-local endpoints/configuration
```

Local configuration and runtime paths must resolve outside this toolkit's package
directory. Do not vendor checkouts, virtual environments, weights, caches, logs,
licenses, keys or robot calibration into the toolkit. The ignore rules are only
a fallback, not an artifact manager or a security sandbox.

Example machine-local configuration (the `root` may be omitted):

```json
{
  "schema_version": 1,
  "root": "~/.local/share/robot-toolkit-tools",
  "tools": {
    "sam3": {
      "enabled": true,
      "endpoint": "http://127.0.0.1:6767",
      "timeout_s": 30.0
    }
  }
}
```

Enabling a client only permits explicit HTTP calls. It does not start a process,
connect a camera or enable a robot. All tools default to disabled. URL credentials,
query strings and redirects are rejected; environment HTTP proxies are ignored.
Remote endpoints are allowed only when explicitly configured: use trusted private
networks or TLS; do not expose the upstream unauthenticated server publicly.

## Commands

Use your existing lightweight toolkit interpreter. From the workspace parent:

```bash
python -m robot_toolkit.external_tools list
python -m robot_toolkit.external_tools launch-plan sam3
python -m robot_toolkit.external_tools list --config ~/.config/robot-toolkit/tools.json
python -m robot_toolkit.external_tools health sam3 --config ~/.config/robot-toolkit/tools.json
```

`list` inspects configured directories, not running services. Existing directories
do not prove correct versions or dependencies. `ready: null` means not checked.
`launch-plan` emits `cwd`, `argv` and environment overrides; it never executes
them, checks GPU capacity, installs dependencies, or declares deployment valid.
Only loopback `http://127.0.0.1` launch plans are supported. Use another port in
configuration when the chosen port is occupied.

The `health` command also supports `anygrasp` and `bundlesdf`. AnyGrasp reports
model loading; BundleSDF's upstream endpoint only reports `status: ok`, so our
client deliberately returns `ready: null` and the CLI exits nonzero. A live HTTP
endpoint alone does not certify a usable tracking/reconstruction runtime.

The SAM plan binds to loopback, disables ENPIRE's automatic `uv` bootstrap and
sets Hugging Face offline mode. Merge the returned overrides with the external
process environment when launching manually. The dedicated environment must
already provide NumPy, PyTorch, compatible Transformers with `Sam3Model` and
`Sam3Processor`, Pillow, FastAPI, Pydantic and Uvicorn. Resolve and lock exact
versions against the GPU/CUDA runtime before installation; do not copy ENPIRE's
entire environment into the robot-control environment.

The upstream model loader searches `facebook/sam3` HF snapshots. Stage an approved
snapshot and its `refs/main` under the generated `HF_HOME/hub` path, preserving the
HF layout and recording its revision/checksums. The `models/` directory alone is
not sufficient. An alternative distribution is
[ModelScope facebook/sam3](https://modelscope.cn/models/facebook/sam3/files).
The existing Transformers service needs `model.safetensors` and its processor,
tokenizer and model configuration files, not the separate `sam3.pt` checkpoint.
Retain the SAM license and verify each downloaded file against repository SHA-256
metadata. Label ModelScope revisions explicitly rather than presenting them as
Hugging Face revisions; downloading a mirror is not evidence of upstream parity.
The loader can discover a local snapshot without changing `refs/main`.
The plan does not preload the model: health can succeed with
`model_loaded: false`; the first explicit segmentation request loads cached weights.
Missing dependencies/weights must fail rather than download. The initial client
integration did not start a service or validate GPU/model compatibility; the later
offline RGB inference check below is not service deployment acceptance.

### Local ModelScope staging (2026-10-27)

Downloaded the Transformers weights and nine supporting files from ModelScope
revision `f8ee5d16d5c30d0e87b1d22cbd358ce97761df56`. All ten files passed SHA-256
checks against the pinned repository metadata. The 3,439,938,512-byte weight has
SHA-256 `6d06f0a5f84e435071fe6603e61d0b4cc7b40e0d39d487cfd4d67d8cc11cc14a`,
matching the public Hugging Face weight metadata at
`3c879f39826c281e95690f02c7821c4de09afae7` (not a claim of full snapshot parity).

Artifacts, the original license, repository metadata and `provenance.json` live
outside the toolkit under `sam3/models/modelscope-<revision>/`. A symlink at
`sam3/cache/huggingface/hub/models--facebook--sam3/snapshots/modelscope-<revision>`
exposes them to the existing launch plan without duplicating weights or changing
`refs/main`. Executing the upstream cache-resolution functions confirmed both
processor and model resolve to this snapshot with `local_files_only=True`.
At this staging step only file integrity and cache discovery were validated;
GPU inference was checked separately below, without starting the SAM3 service.

### Rendered RGB inference (2026-10-27)

Evaluated SAM3 on the existing Newton dual-camera capture at steps
`0, 30, 60, 120, 180, 240, 300`: 14 RGB images and 56 image/prompt requests.
Only `rgb` was read from the NPZ inputs; no depth, object poses or oracle masks
were supplied to the model. Prompts were `letter block`, `red box`, `green ball`
and absent-target `mug`, with score/mask thresholds both 0.5. All 56 saved mask
artifacts passed shape/count/area checks. No policy or new simulator run occurred.

The dedicated environment installation was interrupted. This check instead used
the existing `.venv-il3/bin/python` read-only (Python 3.12, torch 2.11.0+cu130,
torchvision 0.26.0+cu130, transformers 5.10.4), with `tokenizers==0.22.2` installed
only under external `sam3/compat/`. The existing tokenizers 0.23.0rc0 changes
`RobertaProcessing(cls=...)` to `cls_token=...` and fails with these Transformers.
No packages in the simulation environment were installed or modified.

The executable probe is external `sam3/evaluate_sim_rgb.py`. Reproduce with
`PYTHONPATH=<external-root>/sam3/compat HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`,
the interpreter above, and `--model <external-root>/sam3/models/modelscope-<revision>`
`--input <rlinf-workspace>/outputs/piper_perception/newton_final --output <new-directory>`.
Artifacts are in `<rlinf-workspace>/outputs/piper_perception/sam3_rgb_eval_v2/`:
`report.json`, per-request NPZ masks and 14 overlay PNGs, plus `index.html`.

Warm median latency was 150 ms per image/prompt (P95 166 ms); cold first request
594 ms; model loading 1.41 s. Peak allocated PyTorch tensors were 3.66 GiB,
not total system GPU memory. Inference used float32 weights with BF16 autocast.

Visual review: green-ball predictions were consistent in sampled visible views.
The letter-block prompt also selected the red box and missed the actual block
in some external views. The visible block was missed in wrist frames 0 and 300;
the visible red box was missed in wrist frame 120. No mug was detected in any
of the 14 images. Counts of frames with detections are NOT recall or success
rates. No ground-truth mask IoU, real-camera test, HTTP service acceptance,
tracking, grasp execution or policy validation was performed by this check.

`health` makes one GET request. Its `ready` requires both upstream `status == ok`
and `model_loaded == true`; exit status is nonzero if unready or unreachable.
Even `ready: true` is not proof of segmentation quality, correct model revision,
identity tracking, or safe robot behavior.

## Consume an existing observation

```python
from pathlib import Path
from robot_toolkit.external_tools import ExternalTools

tools = ExternalTools.load(config_path=Path.home() / '.config/robot-toolkit/tools.json')
sam = tools.sam3()
result = sam.segment(
    observation.frames['top'].rgb,
    'red block',
    observation_id=observation.observation_id,
    camera='top',
)
for candidate in result.detections:
    mask = candidate.mask            # bool HxW array
    bbox = candidate.bbox_xywh       # image pixels
    score = candidate.score         # model score, not calibrated certainty
```

The caller supplies an already acquired observation; there is no extra camera
read. Validate its freshness/synchronization before calling and reobserve before
acting if inference latency makes it stale. The result preserves observation ID
and camera name, not a promise that the scene stayed static.

The adapter uses ENPIRE's `/segment_all`, retaining every candidate rather than
silently choosing one. A valid empty list means no detection; HTTP failures,
malformed masks, invalid dimensions, contradictory bounds/areas and nonfinite
scores raise `ExternalToolError`. No retries or fallback detections are generated.
Binary uint8 NPY masks are decoded without pickle, validated, then returned as
NumPy bool arrays. RGB is uint8 HxWx3. Base64 NPY in JSON is an upstream protocol
constraint, not a claim of efficient binary streaming. Bounds: eight million
input pixels, 64 returned candidates and 64 MiB response. The timeout is a socket
operation timeout, not a hard end-to-end cancellation deadline.

Segmentation is not object identity, metric position, 6D pose, grasp selection or
task success. This integration does not force SAM into the existing planar/HSV
`SceneSpec`, change the reset verifier or add motion permission. A generic scene
estimator and calibrated depth/pose fusion remain separate follow-up work.

## Prepared source and inspection workflow

The pinned ENPIRE tree has been extracted once, with licenses preserved, to
`~/.local/share/robot-toolkit-tools/sources/enpire-99ee90acf65b5b18957c8382ad580db999528be3`.
The three tool `source/` directories link to that tree. The sibling
`sources/enpire-source.json` records the revision and service-file SHA256 values.
No upstream code was edited. During initial source staging, no install script,
model download or GPU service was run; subsequent SAM3 weight staging is recorded
above. Machine-local `~/.config/robot-toolkit/tools.json` contains disabled
loopback endpoints: SAM3 6767, BundleSDF 8120, AnyGrasp 8122. These are configured
ports, not a claim that servers exist or are running.

Important deployment gap: ENPIRE imports `BundleSdf` and
`bundlesdf.run_live_bundlesdf` (`SharedSam2Tracker`, `build_configs`,
`has_valid_depth`) from an external runtime. That module is not in the ENPIRE
archive or the searched workspace. The stock public BundleSDF checkout must not
be assumed to be a drop-in implementation. Obtain a compatible runtime or write
and test its compatibility wrapper before deployment. In particular, inspect
its reconstruction configuration and validate mesh/texture export; the ENPIRE
pose endpoint does not establish that a reconstructed mesh exists.

The lightweight toolkit entry points are in `robot_toolkit.perception`:

```python
from pathlib import Path
from robot_toolkit.external_tools import ExternalTools
from robot_toolkit.perception import PerceptionFrame, inspect_object

tools = ExternalTools.load(config_path=Path.home() / '.config/robot-toolkit/tools.json')
frame = PerceptionFrame.from_observation(
  observation, 'top', world_frame='base', calibration_id='your-calibration-version',
  depth_aligned_to_rgb=True,
)
inspection = inspect_object(tools.sam3(), frame, 'drawer',
              part_queries={'handle': 'drawer handle'})
mask = inspection.segmentation.detections[0].mask
candidates = tools.anygrasp().plan(frame, mask)
tracker = tools.bundlesdf()
session = tracker.start(frame, 'drawer', name='drawer-01')
```

This example requires explicitly enabled, deployed services and an existing
observation; it never acquires a camera or commands a robot. Source RGB-D must
already be registered and rectified. The conversion rejects stale, future and
explicitly unmatched camera/robot observations, converts uint16 millimetres to
float32 metres, and preserves calibration/observation identity. Absent timing
evidence still does not prove exposure synchronization. A stored wrist extrinsic
must not be mistaken for a transform valid after the wrist moves.

`inspection.points_world` and `visible_bounds_world` describe observed surfaces
only. A single static view does not reconstruct the unseen back. AnyGrasp returns
vendor-frame candidates (`+X` approach, `+Y` finger opening, `+Z` height), not Piper
TCP targets. `candidates.tcp_targets(T_grasp_tcp)` requires an explicitly calibrated
transform. Reobserve after slow inference, check gripper width, IK and collisions,
and use the robot platform's execution boundary before any movement. The SDK's
object-cloud collision check is not full-scene robot collision checking.

During an independently authorized pick/turn/place experiment, submit subsequent
observations with `tracker.push(session, next_frame)`, query `tracker.pose(session)`
as needed, and end with `tracker.end(session)` after the final placed observations.
Keep the same session through pickup, rotation and placement. These methods do
not create a capture thread: an explicit observation producer must continue
feeding frames during motion. A blocking move followed by one snapshot does not
capture intermediate views. The adapter currently requires a fixed camera and
unchanged calibration; it rejects replayed/nonincreasing frame timestamps.

BundleSDF is designed for joint tracking and reconstruction from moving RGB-D
views. ENPIRE calls the per-object tracker on successive frames and skips bad
masks/occlusion; continuous reconstruction still depends on the external runtime.
The final stationary views can add evidence to the same session but cannot reveal
unobserved surfaces. Repeated calls to `/single_frame_pose` create and destroy
independent trackers and must not be presented as accumulated reconstruction.

The upstream push buffer is latest-frame-only and can drop intermediate frames.
Its pose endpoint lacks the source observation ID, pose timestamp and explicit
occlusion flag; it can retain an old pose. Our result keeps
`source_observation_id=None`, marks freshness and reconstruction as unverified,
and distinguishes `last_submitted_observation_id` from the pose's unknown source.
Remote world poses are ignored because ENPIRE has station-specific optical flips;
our client computes world poses locally from `ob_in_cam`. Identity extrinsics sent
to the service prevent fallback to CAP camera queries; do not consume the service's
world-frame output for this stream. Do not use these diagnostic poses as a closed-loop
motion or reset acceptance signal until frame provenance/occlusion is resolved.

SAM3 supports text-prompted concepts, so a separate `drawer handle` query is a
reasonable part-detection attempt. `inspect_object` requires one unambiguous
parent, crops to its bounding box for each part query, and maps all returned masks
back into full-image coordinates. Missing/multiple parts remain missing/multiple;
the caller declares the parent relationship. This is not automatic hierarchical
part decomposition, a verified joint axis, opening direction or grasp affordance.
Real handle segmentation quality is not yet measured. Do not force independently
moving cabinet/drawer parts into one rigid BundleSDF object model.

Offline demonstration, from the workspace parent:

```bash
python -m robot_toolkit.examples.perception_offline
```

It uses synthetic masks and depth, not SAM inference or a robot. Protocol tests:
`python -m pytest tests/test_perception.py tests/test_external_tools.py -q` from
the toolkit root. The tests cover candidate transforms, invalid responses,
one-session in-hand/placed observations and part-mask coordinates. Physical
pick/turn/place and automatic reset remain unimplemented/unvalidated here.

## RAX real-robot comparison

Inspected RAX `real/franka`, commit
`8858d7f3e34eff382abae738cf0578d2726a9a0a`, not our local Piper simulation adapter.

| Layer | Actual RAX implementation |
| --- | --- |
| Arm/gripper | Franka FR3/FCI through Polymetis, e.g. in DROID; external HTTP bridges isolate libfranka realtime threads |
| Cameras | ZED SDK / `pyzed`, two fixed cameras and a wrist ZED Mini in default skills; separate capture process |
| Geometry | NumPy, SciPy Rotation, OpenCV; stored RGB-D/intrinsics/extrinsics and pixel-to-position projection |
| Motion checks | Own FK, damped-least-squares IK precheck, sampled paths, joint/workspace checks and measured Cartesian servo loop |
| Agent tools | Host-owned MCP robot service; status, observe, pixel-to-position, move-pose, move-joints, gripper, stream, hold |
| Agent isolation | Claude Code, bubblewrap/socat, serialized calls and audit log; optional brokered Gemini |

RAX's checked Franka path is not a packaged SAM/BundleSDF/AnyGrasp/cuRobo tool suite.
Its README explicitly says the Polymetis HTTP bridge implementations are external
and not included in that repository. Own IK and workspace checks are not a general
full-scene collision planner. Learn from its single robot owner, operator arming,
camera isolation and audit boundaries; do not replace our Piper backend with
Franka-specific drivers or reuse Franka kinematics.

Source evidence:

- [ENPIRE SAM service and protocol](https://github.com/NVlabs/ENPIRE/blob/99ee90acf65b5b18957c8382ad580db999528be3/enpire/env/forge/tools/vision/serve_sam3.py)
- [ENPIRE lazy tool registry](https://github.com/NVlabs/ENPIRE/blob/99ee90acf65b5b18957c8382ad580db999528be3/enpire/env/forge/registry.py)
- [RAX real-Franka requirements and external bridge contract](https://github.com/robot-agent-exploration/rax/blob/8858d7f3e34eff382abae738cf0578d2726a9a0a/README.md)
- [RAX camera process](https://github.com/robot-agent-exploration/rax/blob/8858d7f3e34eff382abae738cf0578d2726a9a0a/services/franka/camera_server.py)
- [RAX IK precheck](https://github.com/robot-agent-exploration/rax/blob/8858d7f3e34eff382abae738cf0578d2726a9a0a/services/franka/ik.py)

Offline verification: `python -m pytest tests/test_external_tools.py -q` from the
toolkit repository. Tests use synthetic arrays and a temporary loopback HTTP
server, not a GPU model, external provider, camera or robot.