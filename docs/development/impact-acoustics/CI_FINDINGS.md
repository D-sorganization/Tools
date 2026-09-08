# Provider UI Delivery Investigation

This is a diagnosis checkpoint, not a baseline approval or completed CI fix.

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
No quantitative image-region classification or complete nine-view visual review
has been performed, and no reference image was replaced.

The candidate manifest declares
`posix-offscreen-qt-6.11.0-pyqt-6.11.0-matplotlib-3.11.1-font-dejavu-sans-dpi-1.0-1440x900`.
The same job's pytest header explicitly reports PyQt6 6.11.0, **Qt runtime
6.11.2**, and Qt compiled 6.11.0. In the inspected source,
`tests/rate_of_closure/pyqt_visualization_tab_probe.py` records
`QT_VERSION_STR`; that identifies the compiled version. The environment checker
in `scripts/check_rate_pyqt_environment.py` verifies distribution versions for
NumPy, SciPy, PyQt6 and Matplotlib, but does not check the separately supplied
Qt runtime. The font-stack check passes in this job.

This proves the reported environment does not fully identify the loaded Qt
runtime. It does **not** prove that the runtime difference alone causes every
pixel difference, or establish which runtime produced the approved baseline.

## Required Next Steps

Establish the baseline capture's actual runtime from its original job evidence.
Check for an existing issue/claim before implementation. Reproduce the runtime
identity discrepancy with TDD, including a compiled/runtime version disagreement.
Ensure the shared runtime probe, dependency constraint and candidate metadata
agree. Preserve the existing font checks and fail closed on a mismatched stack.
Then rerun the rendered comparisons in the intended Linux environment and inspect
any remaining differences. No threshold widening or automatic baseline replacement.

The T2 UpstreamDrift consumer job 101951943569 passes after upstream PR #9745
merged. T1's separate UD retry is pending. The carried-forward Gasification
failure remains a private-repository checkout error; secret-metadata access
returned 403, so credential availability is unknown. These are separate gates.
