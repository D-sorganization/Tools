# Triangle-Multiset Lamina Mass Moments — Turnover

## Current Scope

Tools #5503 adds a small numerical kernel for a zero-thickness lamina whose
specified total mass is distributed across triangle instances in proportion to
unsigned area. Repeated faces intentionally retain multiplicity. The calculation
returns a center of mass, central inertia tensor, area and topology diagnostics
through a compatible `InertiaResult` plus `TriangleLaminaResult`.

The combined physical-mass change also closes Tools #5504: the URDF exporter now
preserves the signed off-diagonal entries of the symmetric inertia tensor. A
native MuJoCo URDF loadback verifies the asymmetric tensor and kinetic energy;
this is a focused exporter-consumer check, not all-engine parity.

This is synthetic numerical capability only. It does not load or authorize a
mesh, identify a coordinate frame, infer material/tissue mass distribution,
detect or repair partial overlaps, produce an anatomical asset, or admit a
native model. The included manual chapter is an unregistered note, not a
calculation approval or exemplar.

## Source and Verification

- `src/shared/python/humanoid_character_builder/mesh/triangle_lamina.py` —
  input validation, area weighting, stable centered moments and topology
  diagnostics.
- `src/shared/python/humanoid_character_builder/mesh/inertia_calculator.py` —
  compatible result mode, separate from watertight solid-mesh calculations.
- `src/shared/python/humanoid_character_builder/tests/test_triangle_lamina.py` —
  exact triangle and rectangular-plate checks, cube-shell analytic check,
  rigid-transform/scaling/subdivision invariants, duplicate/degenerate faces,
  small representable-area retention, unrepresentable area/fraction rejection,
  remote-origin stability and invalid-input rejection.
- `src/shared/python/humanoid_character_builder/tests/test_urdf_inertia_export.py` —
  symmetric tensor-entry round-trip and actual MuJoCo URDF tensor/kinetic-energy
  readback.

The implementation owner reported 53 focused tests passing (27 lamina tests,
2 URDF export tests including native MuJoCo 3.3.4, and 24 existing mesh-calculator
tests), with `PYTHONPATH=src;src/shared/python` and `--noconftest`; configured
mypy also passed with its cache disabled. This turnover records that report,
not a rerun by the documentation owner. A legacy humanoid test initially
failed collection when the shared source root was missing from `PYTHONPATH`;
adding the established `src/shared/python` root resolved collection without a
production-code change. The first native URDF test attempt encountered a
Windows DLL load-order issue under the pytest process; isolating the native
readback in a subprocess resolved it, and the complete focused run passed.

## Governance Boundary

`manuals/tools/chapters/06-triangle-lamina-mass.qmd` is included for review but
not listed in the textbook chapter registry. The calculation registry and
exemplar coverage carry explicit blocked entries. The D6 freshness generator
currently computes only the D-plane calculation, so this kernel must not be
marked freshness-verified until that workflow supports independent executable
fixtures for it. Do not add a source-backed or anatomical example until a
governed geometry producer establishes source identity, scale, frame, body
ownership and consumer use.

The pinned manual renderer and complete zero-sampling QA have been refreshed
for the included chapter (17 PDF pages, none uninspected). The canonical module
inventory has been regenerated against the fully materialized governed source
tree and passes `--check`; it includes the new lamina module. The URDF sign
correction resolves the local exporter inconsistency tested by MuJoCo, but is
not a claim of parity across all engines or consumers.

## Publication Handoff

The source/test/SPEC/inventory bootstrap is committed as `14442b774d7f`; the
manual and governance artifacts are committed as `eed47a2707e425431715bf37b7763c23b19be19b`. Publication
projection is regenerated against the exact source commit and current rendered
artifacts. No remote branch or PR was created because the normal pre-push unit
hook did not complete. The new canonical `AGENT_HANDOFF.md` pointer is saved in
Git stash commit `67b0e06f9ad749ea2e0bd34985c54ead1404ba9d`, containing only
`AGENT_HANDOFF.md`. The previous handoff manifest remains unchanged and is
accurate for the base handoff file; regenerate both only after a real draft PR
URL exists. No candidate URL is recorded for this change yet.

The normal pre-push hook did not complete. Its full unit run under Windows
Python 3.13/xdist auto terminated with a MuJoCo DLL-load access violation in
`tests/unit/lower_body_model/conftest.py`; 1,034 tests had passed, 24 skipped,
and 6 xfailed when xdist stopped after two failures. Those two failures were
caused by the sparse checkout omitting `.github/workflows`; after materializing
`.github`, both exact tests passed serially. No full retry or push was made.
The source-only central pre-PR run passed all five gates, including all 53
affected tests; a subsequent normal commit hook also passed. A complete
pre-push unit run and the actual PR-bound handoff manifest remain outstanding.
The publication-projection and handoff manifests need the actual source commit
and PR identity, so they remain stale until that coordinated publication point.
The manual release, human review, physical interpretation, and native-model
integration remain blocked.
