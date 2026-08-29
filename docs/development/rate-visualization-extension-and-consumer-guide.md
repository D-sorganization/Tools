# Rate Visualization Extension and Consumer Guide

## Purpose and Evidence Boundary

This guide is the completion map for extending the Rate of Closure React and
PyQt visualization surfaces without duplicating scientific algorithms or
promoting diagnostics into evidence they do not supply. It consolidates the
new-tab checklist, validation commands, parent-issue ownership, and downstream
consumer boundary required by Tools issues
[#4135](https://github.com/D-sorganization/Tools/issues/4135),
[#4142](https://github.com/D-sorganization/Tools/issues/4142),
[#4144](https://github.com/D-sorganization/Tools/issues/4144),
[#4433](https://github.com/D-sorganization/Tools/issues/4433), and
[#4832](https://github.com/D-sorganization/Tools/issues/4832).

Registration, automated rendering, baseline comparison, and human approval are
different evidence tiers. A passing automated probe does not approve an image,
validate a human swing mechanism, establish causal anatomy, or authorize a
coaching recommendation.

## Ownership and Traceability

| Authority | Owns | Does Not Own |
| --- | --- | --- |
| Tools #4135 | Workbench shell, launch, persistence, and cross-surface application contracts | Engine-specific host execution or human validation |
| Tools #4142 | Sampling, ensemble geometry, dispersion, quiet-zone, sensitivity, and evidence schemas | Upstream engine mappings or causal anatomical interpretation |
| Tools #4144 | Universal arc, impact, and shot-outcome visualization semantics | A second numerical implementation in either UI |
| Tools #4433 | Visual hierarchy, visibility, accessibility, responsiveness, performance, and image-governance rules | Automatic human approval |
| Tools #4832 | R14.6 state/case acceptance authority and review packet | Fabricated state execution or unsigned review evidence |
| UpstreamDrift #8358 | Engine execution, canonical parameter mapping, stable marker traces, and consumer orchestration | Tools-owned sampling, statistics, or visualization algorithms |

Tools owns the numerical and evidence authorities under
`shared.python.swing_sim.variation` and `rate_of_closure`. UpstreamDrift owns
its host adapters and engine results. UpstreamDrift must consume the curated
Tools facade from an immutable pin; it must not copy the algorithms into a
parallel package.

The current protected Tools authority for this guide is
`d7a95e2a4024f0f3c1d18f9790143cd766032cd3`. The latest qualified consumer is
[UpstreamDrift #8358](https://github.com/D-sorganization/UpstreamDrift/issues/8358),
closed through [PR #9134](https://github.com/D-sorganization/UpstreamDrift/pull/9134)
at merge commit `f3832b04454c97a7a0999906972f394474107dd7`. That consumer qualifies
Tools R14.3 and its exact vendor pin; R14.6 is a visualization-evidence layer
and does not itself change the numerical provider API. A future provider API
change requires a fresh immutable pin and downstream contract run.

## Five-Manifest Lockstep

Every primary tab is registered in exactly the same order on both surfaces in
five packaged authorities:

1. `visualization_tabs.v1.json` — identity, purpose, prerequisites, visual
   locator, states, counterpart, viewport, and responsive controls.
2. `visualization_accessibility.v1.json` — semantic roles, names, keyboard
   reachability, focus rules, and nonvisual alternatives.
3. `visualization_performance.v1.json` — declared workload and protected
   diagnostic budgets.
4. `visual_baselines.v1.json` — approved reference identity, renderer envelope,
   dimensions, digest, and pixel tolerances.
5. `visualization_acceptance.v1.json` — state-by-reference-case expansion,
   frame, units, provenance, limitations, keyboard path, nonvisual alternative,
   and retained human actions.

Changing only some of these files is a contract failure. The two UIs may adapt
layout and toolkit mechanics, but they must not independently redefine physics,
statistics, frames, units, or evidence categories.

## New-Tab Extension Checklist

### Before Implementation

- Assign one stable snake-case `tab_id`, a reciprocal counterpart, and one
  Tools parent issue.
- Identify the Python authority and any versioned wire/fixture before writing
  React or PyQt presentation code.
- Declare every applicable lifecycle state. Do not register a state that no
  deterministic probe can construct.
- Declare frame, units, provenance, limitations, keyboard path, and a useful
  nonvisual alternative.
- Decide whether the change alters a public provider facade. If so, create the
  downstream issue and protected dependency order before implementation.

### React Surface

- Register the tab and stable test locators without importing Python-only code.
- Preserve the primary visual at 1440x900, 1280x720, and 390x844; advanced
  controls must not displace it.
- Exercise applicable initial, loading, result, no-impact, failure, stale,
  unavailable, and no-result states from deterministic fixtures.
- Retain limitations and provenance at every progressive-disclosure level.
- Add keyboard, focus, responsive, update, layout-shift, and deterministic
  decimation evidence where applicable.

### PyQt Surface

- Construct the widget through the primary-tab registry and store the same
  stable `tab_id` as tab data.
- Provide object names for the primary visual, controls, context, and nonvisual
  result surfaces.
- Exercise every applicable state at 1440x900 under both 1.0 and 1.5 DPI.
- Keep heavy work outside the GUI thread and bound plotted/rendered data without
  mutating the underlying exact samples.
- Record the calibrated Qt, SIP, NumPy, SciPy, Matplotlib, pandas, pyqtgraph,
  pytest, and pytest-qt environment before interpreting image differences.

### Evidence and Publication

- Add or update deterministic fixtures, state probes, semantic assertions, and
  diagnostic captures together.
- Treat candidate PNGs as diagnostic until protected baseline governance and a
  human review bind their exact commit, dataset, environment, digest, findings,
  and disposition.
- Update the #4433 audit and #4142 ledger only for evidence actually executed.
  Keep unmet human or downstream actions partial.
- Update `AGENT_HANDOFF.md`, `src/rate_of_closure/AGENT_HANDOFF.md`, SPEC, and
  module inventory when their governed triggers apply.
- Merge through ordinary protected flow and verify the squash tree and remote
  `main` before downstream pinning or evidence promotion.

## Reproducible Validation Commands

Run Python rendered suites serially and web suites with at most two workers.
The commands below are a minimum map; changed-path governance may require more.

```bash
python -m pytest -n 0 tests/rate_of_closure/test_visualization_tab_manifest.py tests/rate_of_closure/test_visualization_accessibility.py tests/rate_of_closure/test_visualization_performance_manifest.py tests/rate_of_closure/test_visualization_acceptance_manifest.py tests/rate_of_closure/test_visual_baseline_compare.py -q
python -m pytest -n 0 tests/rate_of_closure/test_pyqt_visualization_tab_visibility.py tests/rate_of_closure/test_visual_first_epic_4433_evidence.py -q
python scripts/check_rate_visual_evidence_changes.py --base-ref origin/main
python scripts/check_rate_pyqt_environment.py --constraints requirements-rate-pyqt.txt
cd src/rate_of_closure/web
npm ci
npm test -- --maxWorkers=2
npm run type-check
npm run lint
npm run build
npm run test:e2e
```

For a provider API change, run the UpstreamDrift contract in both editable and
vendored modes after pinning the exact protected Tools commit:

```bash
python -m pytest -n 0 tests/shared_contracts/test_tools_provider_contracts.py --tools-mode local -q
python -m pytest -n 0 tests/shared_contracts/test_tools_provider_contracts.py --tools-mode vendored -q
```

## Completion Rule

The extension is complete only when the registered state/case matrix has exact
executed evidence, responsive and high-DPI findings are dispositioned,
performance and decimation budgets pass on their declared workloads, consumer
contracts pass where applicable, protected merges are verified, and both
retained human actions are signed. Until then, the audit remains partial.
