# Ball-Flight Playback Contract

## Scope

Issue #4200 adds deterministic, local playback to the Rate of Closure ball-flight
views. It does not change launch, impact, aerodynamics, or wind physics.

## Canonical Data

- Time is solver trajectory time in seconds.
- Position is the app frame in metres: x downrange on the target line, y up,
  and z right of target.
- Samples must be finite, non-negative in time, and strictly time-ordered.
- Playback linearly interpolates position between adjacent solver samples and
  clamps requests before launch or after landing to the endpoint samples.
- Playback speed changes wall-clock presentation only. It never changes the
  physical timestamp, trajectory, metrics, or paired wind comparison.

## Presentation

PyQt6 uses the existing Matplotlib side, top-down, and mouse-interactive 3D
panels. The moving ball is a mutable artist, so an animation tick does not clear
the figure or reset a user's 3D camera. One `QTimer` is owned by each control
strip and is stopped on pause, landing, or widget close.

The React view uses a local Canvas 2D orthographic projection. Pointer drag
rotates the camera and the wheel adjusts bounded zoom. Projection fitting uses
one scalar pixels-per-metre value for both screen axes, preserving physical
aspect. One `requestAnimationFrame` callback is scheduled while playing and is
cancelled on pause, landing, trajectory replacement, or unmount.

Both interfaces expose play/pause, time scrub, 0.25x through 4x speed, restart,
jump-to-launch, jump-to-apex, and jump-to-landing controls with accessible names and frame/unit
help. Existing static plots and calm-versus-selected-wind overlays remain visible.

The interactive 3D panels also implement the shared
`docs/specs/active/CAMERA_VIEWPORT_CONTROLS.md` contract: canonical Face On,
Down the Line, Overhead, and Reset views; opt-in ball tracking; explicit Auto
Fit; and one-action Recenter after manual orbit suspends tracking.

## Known Boundaries

- “Launch” is the post-club-impact initial flight sample; “Landing” is the
  terminal ground-contact sample. The labels are deliberately not interchangeable.
- The web renderer is orthographic and dependency-free; it does not provide
  terrain occlusion, shadows, video encoding, or GPU/WebGL effects.
- Playback follows the sample horizon returned by the selected physics model.
  It does not extrapolate after the terminal sample or model bounce and roll.
