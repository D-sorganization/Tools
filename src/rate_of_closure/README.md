# Rate of Closure Impact Explorer

A clubhead at impact is a rigid body with a full 6-DOF motion state — a
twist. The velocity of any point P on the head is

```
v(P) = v(ref) + omega x (P - ref)
```

Launch monitors report the path of a tracked reference point (center of
mass or geometric center). The ball only experiences the path of the
**impact point**. When the head is rotating — and at impact it always is —
those two paths differ. This tool quantifies by how much.

## The Numbers It Answers

| Question | Typical answer (tour-representative delivery) |
| --- | --- |
| How fast is the rotation-induced velocity at the face? | ~2-3 mph |
| How far left of the reported path is the impact point moving? | ~1.3-1.8 deg |
| How much shallower is the impact-point delivery? | ~0.5-0.8 deg |
| How much does the face close *during* the 450 µs of contact? | ~1-1.4 deg |

A 1 percent lateral velocity is not a 1 percent effect: it is a full degree
of path, which at driver speeds is several yards of curvature — larger
than the club-path precision claimed by launch monitors.

## Model

`model.py` decomposes the clubhead angular velocity the way 3-D golf
motion studies report it: rotation about the swing-plane normal
(`omega_plane`) plus rotation about the shaft axis (`omega_shaft`, the
closing/release component). Presets are representative of published
figures (Cheetham's AMM datasets and successors); every value is an input,
not an assumption — enter measured data where you have it.

Frame: +Y target, +Z up, +X trail side. Negative path deviation = left.

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

## Structure

```
model.py              # twist physics, DbC contracts, numpy — no Qt
presets.py            # representative named scenarios
ui/pyqt6/             # ThemedWindowMixin window, controls, 3D + sweep views
web/                  # React/Vite/TS clone; model mirrored + parity-tested
tests/ (repo level)   # tests/rate_of_closure/: model, contracts, GUI smoke
```

Both implementations pin the same numeric cases (the 2.733 mph / -1.30 deg
forum example and the -1.70 deg tour case), so they cannot drift apart
silently.
