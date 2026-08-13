# Rate of Closure Impact Explorer

A clubhead at impact is a rigid body with a full 6-DOF motion state — a
twist. The velocity of any point P on the head is

```
v(P) = v(ref) + omega x (P - ref)
```

Launch monitors report the path of a tracked reference point — the
**geometric center** (GC; the CG lies within ~6 mm of it). The ball only
experiences the path of the **impact point**. When the head is rotating —
and at impact it always is — those two paths differ. This tool quantifies
by how much.

This is the interactive companion to the AffineDrift launch-monitor
research (Launch Monitor Technology Review + closure-rate derivation):
the path gap between two points separated by d is `d / R_ISA`, the offset
over the distance to the instantaneous screw axis, and `1 / R_ISA =
omega / v` — which is why the speed-invariant closure unit the tool
reports is **degrees per foot of travel**.

The Simulation view also renders selectable club and joint screw-motion
glyphs, velocity-component projections, and reconstruction residuals. See the
[Screw-Axis Analysis and Motion Glyphs](../../docs/rate_of_closure/screw_axis_analysis.md)
guide for equations, display semantics, degeneracy rules, and limitations.

## Conventions and Sources

* **Frame** (AffineDrift house convention, `sections/02-parameters.tex`,
  following standard launch-monitor definitions): x along the target line, y up, z right of target.
  Angles positive right and up; club path + = in-to-out. Negative path
  deviation = the impact point travels left of the reported GC path.
  Delivery is referenced at the instant of maximum compression.
* **Rates** (closure-rate literature dossier, verified against the
  Cheetham 2014 paper text): tour-driver horizontal turning velocity
  (HTV, about the shaft) 1,307 ± 304 °/s, range 652–2,432 (n = 94);
  global club closure velocity CCV ≈ 2,100 °/s, reconciling as
  `CCV = HTV·sin(lie) + SPV·cos(lie)`. The model's closure rate *is*
  this CCV by construction.
* **Geometry**: openly published head data cites a 25–50 mm GC-to-face offset for drivers;
  the default 40 mm is the AffineDrift worked-example value.
* **Calibration cross-check**: openly published launch-monitor material puts the GC-path
  vs face-center-path gap at roughly 3° for a driver; the "Published ~3°
  worked example" preset back-solves it (the implied closure exceeds the
  Cheetham range — the R_ISA ≈ 0.77 m tension the derivation documents).

## The Numbers It Answers

| Question | Cheetham tour-median preset (120 mph) |
| --- | --- |
| How fast is the rotation-induced velocity at the face? | ~2–3 mph |
| How far left of the reported path is the impact point moving? | ~1.6° |
| What is the speed-invariant closure? | ~12 °/ft |
| How much does the face close *during* the 450 µs of contact? | ~0.9° |

A "1 percent" lateral velocity is not a 1 percent effect: it is over a
degree of path, which at driver speeds is several yards of curvature —
comparable to the club-path precision claimed by launch monitors, and per
Cheetham a **dispersion** variable (closure rate barely correlates with
outcome bias, r = −.14), not a bias you can simply calibrate away.

## Run It

```bash
# PyQt6 desktop (animated 3D clubhead, closure sweep)
python src/rate_of_closure/launch_pyqt6.py

# Shareable web version through the repository launcher
python src/rate_of_closure/launch_web.py

# Equivalent direct Vite development workflow
cd src/rate_of_closure/web && npm ci && npm run dev
```

The web app builds to a static bundle (`npm run build`) that can be hosted
anywhere as a link. A Tauri desktop wrapper is not currently part of this
tool; the supported desktop release is the PyQt6/PyInstaller application.

Both interfaces open with a generated driver head and its engineering CG
target visible. The Simulation view runs immediately and supports manual,
double-pendulum, and triple-pendulum sources; pendulum modes draw every joint
and link rather than only the clubhead trace. Directional entries include
clickable reference-frame notes. Web numeric fields select the complete value
on focus and accept intermediate signed decimal drafts (for example, `-12.5`
degrees of spin-axis tilt) before committing on Enter or focus loss.

## Build a Standalone Executable

Users can package the explorer and experiment without a Python
environment:

```bash
# Desktop (PyQt6) — requires: pip install pyinstaller
python src/rate_of_closure/build_executable.py            # one-folder app
python src/rate_of_closure/build_executable.py --onefile  # single file

# Static web release — from src/rate_of_closure/web
npm ci
npm run build
```

The PyInstaller output lands in `dist/RateOfClosureExplorer`; the static web
bundle lands in `src/rate_of_closure/web/dist`.

## Structure

```
model.py              # twist physics, DbC contracts, numpy — no Qt
presets.py            # dossier-sourced named scenarios
ui/pyqt6/             # ThemedWindowMixin window, controls, 3D + sweep views
web/                  # React/Vite/TS clone; model mirrored + parity-tested
tests/ (repo level)   # tests/rate_of_closure/: model, contracts, GUI smoke
```

Both implementations pin the same numeric cases (the 2.733 mph / −1.30°
forum example, the −1.70° legacy tour case, the ~2,100 °/s default CCV,
and the −3.0° published worked example), so they cannot drift apart
silently.

## Variation Study Visualizations

The **Variation** view supports delivery, launch, and complete double-pendulum
swing ensembles. Every run retains its sampled inputs and scalar outputs. A
complete swing ensemble additionally retains the common-time-grid positions of
the pivot, wrist, and clubhead reference, including evaluated trials that miss
the ball and explicit invalid rows for numerical failures.

The Impact and Shot-Outcome Scatter view lets either axis select any available
input, contact, impact, or shot scalar. Axis labels always include units. Swing
studies report three disjoint cohorts: evaluated hit, evaluated no impact, and
numerical failure. A value that is physically unavailable is omitted from the
plot and counted as unavailable; it is never replaced by zero. Scalar-only
delivery and launch studies cannot identify a geometric no-impact event and say
so in the view.

The All Swing Arcs view overlays every valid trial plus the pointwise median.
Select the pivot, wrist, or clubhead reference; drag to rotate, use the mouse
wheel or `+`/`-` to zoom, arrow keys to rotate from the keyboard, and Reset View
to restore the engineering view. The plot uses one isotropic spatial scale, so
changing the viewport cannot stretch one physical axis relative to another.
Rendering uses deterministic vertex decimation when the study exceeds its
display budget; exports always retain the full resolution.

Spatial data use the stable application frame
`app_frame:x_target,y_up,z_right`: x points down the target line, y points up,
and z points right of target. Position and time units are metres and seconds.
This is an application frame, not a camera frame; rotating the display does not
change the data coordinates.

Exports are intentionally split by purpose:

* **Dataset CSV/JSON** contains sampled inputs, scalar outputs, success flags,
  and the reproducible plan.
* **Swing Traces CSV** is long-form data with one row per trial, sample, and
  modeled point, including typed status, impact marker, units in column names,
  and coordinate-frame ID.
* **Swing Ensemble JSON** is the lossless document containing the plan, scalar
  dataset, typed outcomes, complete position traces, validity mask, and impact
  sample indices.

One-at-a-time sensitivity reruns each selected input through the same execution
path as the joint study. For swing mode this means the complete simulator, not
the scalar approximation. Misses remain part of contact-level statistics while
impact and shot columns use only their finite hit values.

Current scope: complete trace ensembles require the double-pendulum source and
global perturbations. Local time-window or point-targeted perturbations are
rejected before execution rather than being accepted without a modeled effect.
