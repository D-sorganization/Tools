# Rate of Closure Handoff

## Variation visualization carrier

- Branch: `feat/4144-variation-visualizations`
- PR: `#4167` (draft)
- Base: `feat/investigation-suite`
- Pre-repair head: `edaa56358a9ccf47809533fcab28e6415b336771`

## 2026-08-11 pinned-Ruff repair

Exact published head `3c19aaa9d3e812e4659053735a2955d62a080d34`
inherits the five-file Ruff `0.14.10` format mismatch reported on its immediate
child. The files are now mechanically formatted with that pinned version. No
material handoff or runtime behavior changes: variation, physics, frames, DbC
validation, public contracts, schemas, tests, and UI behavior remain intact.

The branch contains the typed multi-trial visualization foundation described
in `SPEC.md`. Its current in-scope CI repair replaces the Python 3.11-only
`datetime.UTC` import in the torque-profile controller with
`shared.python.compatibility.UTC`. The timestamp remains an ISO-8601 UTC value
ending in `Z`; no persistence schema or user-visible behavior changes.

The AST guard covers direct imports plus unaliased and aliased module-attribute
access. Verification is green: 27 focused controller/history/AST tests and the complete
554-test Rate suite pass on Python 3.13; the shared UTC export is verified using
real CPython 3.10.20; Ruff check/format, focused pinned MyPy 1.13,
detect-secrets, touched-file size, and diff checks pass. Protected checks and
review are still required after publication.

Update this handoff, the repository handoff, the campaign handoff, and
`SPEC.md` in every implementation commit.
