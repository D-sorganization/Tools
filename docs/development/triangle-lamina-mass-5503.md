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

The current focused run passes all 53 tests: 27 lamina tests, 2 URDF export
tests including native MuJoCo 3.3.4, and 24 existing mesh-calculator tests.
The command uses `-o addopts='' -n auto` with
`PYTEST_XDIST_AUTO_NUM_WORKERS=2`; all 53 pass in 8.88 seconds. The normal
RM-6 pre-PR gate passes all five checks, including diff mypy. The explicit
three-coordinate tuple construction in `triangle_lamina.py` satisfies the
`InertiaResult` tuple annotation without changing values. Its LF-normalized
module content hash is
`3fd45937bed011c90b4d4e3fdf8bcf932a8bfd9a0816441ee708cc93cd4080ee`.

A legacy humanoid test initially failed collection when the shared source root
was missing from `PYTHONPATH`; adding the established `src/shared/python` root
resolved collection without a production-code change. The first native URDF
test attempt encountered a Windows DLL load-order issue under the pytest
process; isolating the native readback in a subprocess resolved it. RM-6's
initial uncapped 14-worker run later reproduced a native MuJoCo DLL access
violation (52 passed, 1 failed); the bounded 2-worker run passed all 53.

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
manual and governance artifacts are committed as `eed47a2707e425431715bf37b7763c23b19be19b`. The branch includes a normal merge of current
`origin/main` (`74030f4aaecb04d0839eb2044fd0ab799b360ebb`). Draft PR #5510 is
open at https://github.com/D-sorganization/Tools/pull/5510. Its exact provider
source commit is `920e6fd633e796cf311cfd16f294fd45f5ad8cb1`, tree
`e2ea298055d03302eee1d9e186d1e1e894412137`; the lamina module's LF SHA-256 is
`3fd45937bed011c90b4d4e3fdf8bcf932a8bfd9a0816441ee708cc93cd4080ee`. The
publication projection is regenerated against that source commit and current
rendered artifacts. The handoff manifest is regenerated with the actual PR
URL; the owned root handoff pointer from stash
`67b0e06f9ad749ea2e0bd34985c54ead1404ba9d` is now restored with the PR link.

RM-6 pre-PR passed all five gates. The full normal pre-push passed with
`PYTEST_XDIST_AUTO_NUM_WORKERS=1`, including all unit tests, mypy, Bandit,
pip-audit, and fleet guardrails. A two-worker attempt stopped on MuJoCo DLL
access violations after 1,069 passed, 24 skipped, and 6 xfailed, plus one
sparse-checkout `FileNotFoundError` in
`test_current_user_docs_do_not_claim_root_python_310_support`. The five exact
documentation paths were hydrated; that test passed independently. The earlier
14-worker run had the same native DLL issue and also lacked `.github/workflows`;
both previously failing workflow tests passed after `.github` was materialized.
The draft remains blocked on reviewer/consumer integration. Manual release,
human approval, physical interpretation, and native-model integration remain
blocked.
