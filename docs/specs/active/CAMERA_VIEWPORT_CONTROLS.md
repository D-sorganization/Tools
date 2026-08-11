# Camera Viewport Controls Parity Registry

Status: active. Ready-for-review Tools PR #4358 carries the first matched
clubhead-preset slice for issue #4284. Ready-for-review child PR #4362 is based
exactly on published head `d662b016eceed8cbfbce26c12a42ca2c326a684f` and
adds the bounded tracking slice described below. Protected checks, dependency
landing, and release remain open.

## Problem

The standalone PyQt6 and React clubhead animations previously had independent
orbit defaults and no discoverable engineering camera presets. Moving heads
could also be clipped after zooming. Camera convention drift is unacceptable
because a view label must identify the same physical observation direction in
both renderers.

## Implemented scope

The matched `Club3DView` and `ClubCanvas` surfaces consume equivalent strict
camera contracts pinned by one golden fixture. The application frame is
`x=downrange`, `y=up`, `z=right`. Stable command IDs are:

- `camera.view.isometric`
- `camera.view.face_on`
- `camera.view.down_the_line`
- `camera.view.overhead`
- `camera.reset_view`
- `camera.auto_fit`
- `camera.track_clubhead`
- `camera.recenter`

Down the Line observes from behind exactly along +x with +y vertical.
Overhead observes along -y with +x toward screen-up. Face On never infers
handedness: right-side observes along -z and left-side along +z. Preset and
Reset actions preserve target and zoom. Auto Fit alone may change zoom; it
fits the current complete clubhead and shaft bounding sphere with 16% axis
clearance. Manual orbit remains intact when zoom or Auto Fit changes scale.
After free orbit, both surfaces clear the selected/pressed preset state and
identify the view as custom; the last preset remains internal state only so no
control falsely claims an exact canonical orientation. Selecting a preset or
Reset restores exact canonical state. Changing side while the last preset is
Face On deliberately restores the exact Face-On view for the selected side.

Canonical orthographic-like projections suppress only their collapsed depth
axis in the PyQt renderer: Face On hides display x (`z=right`), Down the Line
hides display y (`x=target line`), and Overhead hides display z (`y=up`).
Isometric and every custom free orbit show all three labeled axes. This fixes
the stacked tick labels found during rendered screenshot review without
discarding the user's manual orbit.

Both control bars expose visible labels, stable machine-readable IDs,
tooltips, keyboard focus, and selected-view pressed state. The React canvas is
keyboard-focusable. Contracts reject unknown view/side IDs, non-finite state,
non-unit directions, and non-perpendicular screen-up vectors with the same
absolute 1e-12 orthonormal tolerance.

Track Clubhead is opt-in and local to each viewport. Enabling it centers the
current clubhead without changing zoom, after which target motion is bounded
to 0.05 m per rendered frame. Start, impact, and end phases are covered. A
playback-loop discontinuity recenters exactly rather than animating a large
false translation. A scenario/replay phase reset also recenters exactly only
when tracking is active; suspended tracking retains its manual target. Manual
orbit places tracking in a visible `suspended` state
in both clients. Native PyQt pan also retains the user's visible target while
suspending; React pan remains open. Re-center resumes tracking at the current
subject without changing zoom or orientation. Turning tracking off retains the
current target.

The existing Auto Fit remains a one-shot command. A separately labeled
`camera.auto_fit_fallback` checkbox opts into continuous clearance protection:
only an unsafe zoom is reduced, never increased, and the complete clubhead and
shaft retain the same 16% clearance. Clearance is corrected before rasterizing
the affected frame, including discontinuous mode, geometry, and target changes;
there is no one-frame clipped transient. Tracking and fallback flags are isolated
per viewport. React now projects every world point relative to `targetM`, so
the shared camera target has real visual effect instead of changing fit math
alone.

## Authorities

- Python: `src/rate_of_closure/application/camera_presets.py`
- PyQt6 controls: `src/rate_of_closure/ui/pyqt6/club_camera_controls.py`
- TypeScript: `src/rate_of_closure/web/src/model/cameraPresets.ts`
- React controls: `src/rate_of_closure/web/src/components/ClubCameraControls.tsx`
- Shared fixture:
  `src/rate_of_closure/web/src/model/__fixtures__/camera_presets_v1.json`
- Tracking fixture:
  `src/rate_of_closure/web/src/model/__fixtures__/camera_tracking_v1.json`

## Surface registry

| Surface | Presets / side / reset | Auto Fit | Tracking | Status |
| --- | --- | --- | --- | --- | --- |
| Tools PyQt6 `Club3DView` | Matched | Matched | Matched in #4362 | Ready PR |
| Tools React `ClubCanvas` | Matched | Matched | Matched in #4362 | Ready PR |
| Tools PyQt6 `SimulationView` | Legacy partial selector | No | No | Open |
| Tools React primary Swing view | No interactive 3D camera | No | No | Open |
| Tools React impact scene | Legacy partial buttons | No | No | Open |
| Tools PyQt6 / React flight | Orbit and zoom only | No shared contract | No | Open |
| UpstreamDrift consumers | Not assessed | Not assessed | Not assessed | Open |

## Non-goals and remaining acceptance

This child does not implement per-viewport workspace persistence, the
principal React 3D swing conversion, other simulation/impact/flight adapters,
Playwright/high-DPI rendered review, or the complete camera/playback matrix.
Those remain explicit acceptance gates for #4284 and epic #4218. Neither ready
PR is protected-complete, dependency-landed, or released, and neither issue may
be represented as complete.

## Validation

Python/PyQt headless and React component tests pin the shared fixture, exact
adapter angles, explicit Face-On sides, strict malformed-input rejection,
idempotence, zoom/target preservation, command discoverability/focus, manual
orbit preservation, and Auto Fit at fixed/moving start, impact, and end phases
for the representative driver geometry. The tracking suite additionally pins
bounded target motion, start/impact/end centering, matched manual-orbit and
native PyQt-pan suspension, exact recenter and playback-wrap behavior,
active-versus-suspended scenario reset, pre-raster reduction-only fallback
clearance, per-viewport isolation, and React target-relative projection.
