# Provider UI Delivery Investigation

Renderer prerequisite [PR #5090](https://github.com/D-sorganization/Tools/pull/5090)
is open at `53f072a58bce42a9c11a639329da01b093e30557`. No baseline approval or
completed CI fix is claimed.

## Evidence Inspected

Tools T2 head `2f975d06ee9e4192a4b35ba5172bb7b3b8950e8a`,
[run 34180114955, job 101931507111](https://github.com/D-sorganization/Tools/actions/runs/34180114955/job/101931507111):
all 23 rendered interaction/accessibility tests pass. The subsequent comparison
fails for **nine** PyQt views: clubhead, plots, calculation_description,
flight_explorer, launch_monitor_analytics, neural_model_lab, variation, putting
and glossary. The earlier turnover count of seven was incomplete.

[Artifact 10040345545](https://github.com/D-sorganization/Tools/actions/runs/34180114955/artifacts/10040345545)
was downloaded to a temporary analysis directory. Its name is
`rate-web-playwright-pr-34180114955-2`, reported archive size 38,823,824 bytes.
The committed and candidate clubhead PNGs were visually inspected. The main
plot/content remain present; differences appear around widget labels and borders.
An RGB-delta >1 region audit found exactly 705 changed left-panel pixels in all
ten views. A complete all-view visual review remains open; no reference image
was replaced.

The candidate manifest declares
`posix-offscreen-qt-6.11.0-pyqt-6.11.0-matplotlib-3.11.1-font-dejavu-sans-dpi-1.0-1440x900`.
The same job's pytest header explicitly reports PyQt6 6.11.0, **Qt runtime
6.11.2**, and Qt compiled 6.11.0. In the inspected source,
`tests/rate_of_closure/pyqt_visualization_tab_probe.py` records
`QT_VERSION_STR`; that identifies the compiled version. The environment checker
in `scripts/check_rate_pyqt_environment.py` verifies distribution versions for
NumPy, SciPy, PyQt6 and Matplotlib, but does not check the separately supplied
Qt runtime. The font-stack check passes in this job.

The latest reference PNGs trace to trusted run 34045862045, job 101522812023
on OGLaptop, **also Qt runtime 6.11.2**. Its FreeType/Fontconfig packages are
2.14.2+dfsg-1ubuntu0.1 / 2.17.1-3ubuntu1; the candidate uses
2.13.2+dfsg-1ubuntu0.1 / 2.15.0-1.1ubuntu2. The old checker accepted both stacks
under one screenshot identity. These observations support a common rendering
difference, not proof that every changed pixel is attributable to fonts.

## Current Remediation and Next Steps

Existing issue #4844 was claimed and updated with the actual runtime/font
provenance. PR #5090 uses one digest-pinned Ubuntu 24.04 container and common
font installation for both paths while retaining the PR-hosted/trusted-fleet
runner boundary. It checks the exact font stack and the Qt runtime/SIP pins.
TDD and SPEC checks pass 49 tests; all nine manual checks and normal commit/push
hooks pass. No image, tolerance or runner-policy guard was weakened.

Continue in `C:/Users/diete/Repositories/Tools-impact-render`, branch
`fix/4844-consistent-pyqt-renderer`, with its
`docs/development/rate-pyqt-renderer-4844.md`. Linux capture, every-candidate
inspection and a protected reference update remain required. The local shell
has no Docker executable, so no local container render is claimed.

The T2 UpstreamDrift consumer job 101951943569 passes after upstream PR #9745
merged. T1's separate UD retry job 101955835753 also passes. The carried-forward Gasification
failure remains a private-repository checkout error; secret-metadata access
returned 403, so credential availability is unknown. These are separate gates.
