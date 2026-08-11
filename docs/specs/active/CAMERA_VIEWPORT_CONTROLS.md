# Camera Viewport Controls

Status: implemented locally for Tools issue #4284; protected integration and
UpstreamDrift consumer parity remain open.

## PR #4331 stack repair

The local publication candidate normally merges exact current PR #4330 parent
`304a069b1777dcf8cf107de26caa3b9fbe96dbb3` after exact live PR #4331 child
`c7bccbccc6cda0c9b938b2862ed660cebdcb7597`. This corrects ancestry only: it
removes the parent's formatting and worktree-pointer files from the effective
child comparison while preserving the camera implementation and every command
contract below. Protected exact-head CI and independent review remain release
gates.

Root `SPEC.md` version `1.14.30` now carries the same orthographic-axis
presentation contract as this active specification. The synchronization is
documentation-only; the previously validated runtime tree is unchanged.

Fresh merged-tree verification covers 71 Python/PyQt camera, compositor,
layout, main-window, and manifest tests; exact-delta Ruff/format, pinned MyPy
1.13, Bandit, Spec Check, version/governance, module-size, assertion,
whitespace, and diff gates;
the complete 114-file / 686-test React suite; TypeScript; zero-warning ESLint;
the 199-module production build; and four serial desktop/constrained-2x-DPR
Playwright camera cases. The dependency audit reports zero vulnerabilities in
337 packages.

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

Exact orthographic presets suppress the complete Matplotlib presentation
(axis container, label, line, pane, and tick artists) only for the display axis
perpendicular to the screen: display x/right for Face On, display y/downrange
for Down the Line, and display z/up for Overhead. This explicit artist-level
contract prevents cached Axes3D depth labels from surviving a snap. The two
visible axes retain their physical labels, ticks, and engineering units.
Reset/isometric and any manual orbit restore all three axes, so hidden
presentation state cannot leak between camera modes. Visible axes retain
Matplotlib's native one-sided tick-artist selection; the adapter never forces
both tick-label sides on and therefore cannot create duplicate labels.

## Tracking and interaction

Tracking is opt-in and each viewport owns its own state. The target advances
toward the current clubhead or ball by at most the adapter's positive,
finite per-frame step. Zoom is preserved when the subject clearance radius
fits; optional Auto Fit only caps an unsafe zoom with 16% clearance. Manual
orbit suspends tracking predictably. Recenter targets the subject in one action
and resumes tracking. Reset selects the canonical isometric orientation.

All command buttons have visible labels, stable IDs, tooltips, native keyboard
focus, and focus-visible styling. Controls remain available while playback is
paused, playing, looping, restarted, stepped by solver-owned sample, resized,
or rendered at high DPI.

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
idempotence, orthographic depth-axis suppression and restoration, focusable
controls, tracking bounds, zoom clearance, manual suspension, recentering, and
complete swing/flight horizons. Playwright covers a bounded camera/playback
interaction matrix in desktop Chromium and at a 520 x 900 viewport with 2x
device scale, including responsive control containment and the canvas backing
store. Final issue closure still requires protected carrier integration,
hosted CI/review, post-polish native rendered review, and UpstreamDrift consumer
parity. Camera preferences are not persisted in this slice.
