# Planar scenes and initial-layout checks

This is the first task-scoped scene layer, not a general object tracker or a
training environment. It uses a fixed camera and uniquely color-identifiable
objects on one calibrated plane. No hardware access, model downloads, motion,
automatic resets, rewards, or background workers are introduced by this module.

## Offline demonstration

OpenCV is optional and imported only when estimating a scene. Install the
`vision` extra into the environment used by the toolkit. The current offline
tests use OpenCV 4.10.0 and NumPy 1.26.4.

From the workspace parent directory:

```bash
python -m robot_toolkit.examples.scene_offline
```

The example replays five synthetic RGB frames using a simulated reception clock.
Three distinct stable frames accept the layout; a missing object and a displaced
object reject it. It does not open a camera, connect a robot, or wait. The example
calibration and thresholds are synthetic and MUST NOT be used for real motion.

## Contracts

- `ObjectSpec`: stable configured ID, one or more OpenCV HSV intervals, and pixel
  area limits. Hue is 0..179; saturation/value are 0..255. Split wraparound hue
  ranges, such as red near both 0 and 179, into two intervals.
- `SceneSpec`: camera name, explicit calibration ID, coordinate-frame name,
  calibrated image size, image-to-plane homography, workspace XY limits in
  metres, object definitions, and maximum frame age. Configuration is detached
  and frozen; `spec_id` hashes its contents.
- `SceneState`: source observation ID, frame reception time, estimate time,
  spec/calibration IDs, coordinate frame, and per-object validity/evidence.
  Valid `xy_m` is a NumPy array. Invalid positions are `None`, never stale poses.
- `PlanarColorEstimator`: RGB-to-HSV segmentation, connected-component gates,
  centroid projection, and conservative projected-bounding-box workspace checks.
  Multiple significant components for one ID or overlapping assignments are
  ambiguous. There is no learned confidence score or temporal identity tracker.

Only XY is estimated. The position is the projected image-mask centroid, not a
certified object-frame origin or grasp pose. There is no yaw, full 6D pose, depth
fusion, arbitrary same-color instance association, or contact inference. A
partially occluded or wrongly colored object can still fool the detector; area
checks are not proof of complete visibility or identity.

## Calibration requirements

`image_to_plane` maps UNDISTORTED pixels to physical XY metres. Obtain it from
surveyed plane correspondences using OpenCV homography estimation, then check
independent points across the operating area. Four fit points alone do not
validate calibration accuracy. Choose tolerances using measured localization
error and task requirements, not the synthetic example's values.

Frames must use the same camera, image geometry, rectification, and feature
plane as calibration. Known nonzero distortion coefficients are rejected; pass
properly rectified images and corresponding metadata. Missing coefficients do
not prove rectification. Moving the camera, changing crop/resolution, or using
objects whose visible features are above the calibrated plane invalidates the
mapping and requires revalidation. The estimator cannot detect all such changes.
It does not use `T_world_camera` to silently compensate for a moving wrist camera.
Calibration IDs label supplied evidence; they are not physical calibration proof.

## Platform integration

Build a `SceneSpec` from your measured calibration and approved object definitions.
Then attach the estimator to an existing platform's borrowed camera hub:

```python
from robot_toolkit import RobotPlatform, PlanarColorEstimator

platform = RobotPlatform(
    robot, gripper=gripper, cameras=cameras,
    scene_estimator=PlanarColorEstimator(spec),
)
observation = platform.observe()
scene = observation.scene
item = scene.objects['red_block']
if item.valid:
    print(item.xy_m)
else:
    print(item.reason)
```

Devices must already be opened by their owners. Estimation reuses the single
platform image snapshot, preserving its observation ID and host-time pairing
evidence. The selected camera must be in the hub. Unmatched camera/robot pairing
is rejected by the estimator even with platform `strict=False`.

Historical observations retain detached scene estimates but not image buffers.
`observe(include_images=False)` and all existing motion calls skip scene
estimation and return `scene=None`, rather than reusing an old scene. An invalid
scene does not automatically block raw or platform motion: scene-aware action
authorization and collision checking are separate, not yet implemented here.

## Initial-layout acceptance

```python
from robot_toolkit import PlacementCondition, ResetLayoutVerifier

verifier = ResetLayoutVerifier(
    spec, (PlacementCondition('red_block', target_xy_m=(0.04, 0.03), tolerance_m=0.01),),
    required_frames=3, min_stable_s=0.2, max_step_m=0.003,
)
check = verifier.check(observation.scene)
print(check.layout_ready, check.reason, check.errors_m)
```

Call `check` on successive actual observations from the application's existing
loop. It does not poll, sleep, reset objects, or send actions. All scene objects
must have placement conditions. Acceptance requires distinct increasing frame
timestamps, valid fresh estimates matching the frozen scene contract, all
positions within their target radii, and a stable run lasting the configured
time and frame count. `max_step_m` limits displacement from the run's first
position, not just consecutive-frame displacement; slow drift cannot accumulate
unnoticed. Excessive frame gaps restart progress. Duplicate/out-of-order frames,
missing objects, invalid calibration provenance, and stale images reset progress.

`LayoutCheck` includes spec/layout IDs, observation ID, errors, duration, and
consecutive-frame count. `layout_ready` is true ONLY for the evaluated fresh
sample; it must not be cached indefinitely as a motion permission. `clear()`
clears verifier progress, not the scene or any device. Configuration hashes are
audit identifiers, not a security boundary against untrusted policy code.

## Before autonomous experiments

This implementation verifies XY layout, not orientation, gripper release,
attachment, robot readiness, settled motion, force, or collision clearance.
It is not a `ResetController` and does not certify reset safety. The next gate
is a station-specific acceptance set: annotated normal, missing, same-color,
occluded, displaced, lighting-change and camera-shift frames; independent XY
measurement; and false-accept/reject statistics with frozen thresholds.

Start with manual reset plus this read-only checker. Only after physical
validation should a reviewed reset skill and independent safety supervisor be
added, with bounded attempts and a `needs_operator` outcome. Keep perception,
verification, calibration, reset and safety implementations outside the research
policy's writable scope. No real-station accuracy or reset reliability has been
measured by the synthetic demonstration.