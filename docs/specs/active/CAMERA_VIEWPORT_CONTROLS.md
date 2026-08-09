# Camera Viewport Controls

Status: implemented locally for Tools issue #4284; protected integration and
UpstreamDrift consumer parity remain open.

## Problem and scope

Moving clubheads and balls can leave a manually framed 3D view. Every Tools
PyQt6 and React swing, impact, and flight 3D viewport therefore consumes one
UI-neutral camera contract. Physics, trajectory, and geometry are unchanged.

## Frame and commands

The app frame is x downrange, y up, z right. Stable command identifiers are
`camera.view.isometric`, `camera.view.face_on`,
`camera.view.down_the_line`, `camera.view.overhead`, `camera.auto_fit`,
`camera.recenter`, and `camera.track_subject`. Down-the-line looks from behind
along +x; overhead looks along -y. Face-on is explicit: right-side looks along
-z and left-side along +z. A snap is idempotent and sets orientation only; it
does not silently change target or zoom.

## Tracking and interaction

Tracking is opt-in and each viewport owns its own state. The target advances
toward the current clubhead or ball by at most the adapter's positive,
finite per-frame step. Zoom is preserved when the subject clearance radius
fits; optional Auto Fit only caps an unsafe zoom with 16% clearance. Manual
orbit suspends tracking predictably. Recenter targets the subject in one action
and resumes tracking. Reset selects the canonical isometric orientation.

All command buttons have visible labels, stable IDs, tooltips, native keyboard
focus, and focus-visible styling. Controls remain available while playback is
paused, playing, looping, restarted, resized, or rendered at high DPI.

## Architecture and contracts

- Python authority: `src/rate_of_closure/application/camera_commands.py`
- PyQt adapter: `src/rate_of_closure/ui/pyqt6/camera_controls.py`
- TypeScript authority: `src/rate_of_closure/web/src/model/cameraCommands.ts`
- Cross-runtime fixture:
  `src/rate_of_closure/web/src/model/__fixtures__/camera_commands_v1.json`

Contracts reject non-finite vectors, non-unit or non-perpendicular camera
bases, invalid zoom, and non-positive tracking steps. No adapter infers a view
convention from an arbitrary club pose.

## Validation and remaining gates

Unit and headless GUI/component tests cover exact orientations, parity,
idempotence, focusable controls, tracking bounds, zoom clearance, manual
suspension, recentering, and complete swing/flight horizons. Final issue
closure still requires protected carrier integration, hosted CI/review,
rendered desktop and constrained-browser review, and UpstreamDrift consumer
parity. Camera preferences are not persisted in this slice.
