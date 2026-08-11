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

# Shareable web version (same math, pinned test-for-test to the Python)
cd src/rate_of_closure/web && npm install && npm run dev
```

The web app builds to a static bundle (`npm run build`) that can be hosted
anywhere as a link, and carries the same Tauri scripts as the other web
tools for desktop packaging.

Both interfaces open with a generated driver head and its engineering CG
target visible. The Simulation view runs immediately and supports manual,
double-pendulum, and triple-pendulum sources; pendulum modes draw every joint
and link rather than only the clubhead trace. Directional entries include
clickable reference-frame notes. Web numeric fields select the complete value
on focus and accept intermediate signed decimal drafts (for example, `-12.5`
degrees of spin-axis tilt) before committing on Enter or focus loss.

Both Club panels can export the selected representative head as deterministic
binary STL. The generator works in metres, while the unitless STL coordinates
are written in conventional millimetres; the fixed header records
`units=mm`, the head frame, and axes `x=target,y=up,z=toe`. Portable filenames
handle Unicode-only and Windows-reserved club names. This visual parametric
mesh is not a density-integrated CAD model and does not imply a measured 3×3
inertia tensor.

The adjacent **Engineering Sidecar JSON** action exports the strict versioned
`rate_of_closure.clubhead_engineering/1` record on both surfaces. It identifies
the exact portable-named companion STL by SHA-256 and byte length, declares its
frame/units/transform and head mass provenance, and fails closed for the
unavailable complete CG, symmetric CG inertia tensor, world attitude, and
assembly properties. Partial CG offsets and the shaft-axis scalar remain
labeled `evidence_only`; they are never emitted as substitute tensor or CG
values.

## Build a Standalone Executable

Users can package the explorer and experiment without a Python
environment:

```bash
# Desktop (PyQt6) — requires: pip install pyinstaller
python src/rate_of_closure/build_executable.py            # one-folder app
python src/rate_of_closure/build_executable.py --onefile  # single file

# Web shell (Tauri) — from src/rate_of_closure/web
npm run tauri build
```

The PyInstaller output lands in `dist/RateOfClosureExplorer`.

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
