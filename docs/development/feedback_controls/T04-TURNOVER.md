# Native State Artifact and Replay Envelope Turnover

## Development Checkpoint

Tools #5470 is leased by root/codex. The clean merged T01 checkout was reused
initially on `feat/native-state-artifact-5470`, based on main `b1480ff6e`.
The implementation is now merged into `feat/native-state-transport-5470`
for draft PR5471, after main collation commit5dffcd1c4.
No new full clone or native environment was created. Source and tests are
committed locally; the draft initially published a documentation-only staging
checkpoint. The complete implementation is published in ready PR5471, with
protected automatic merge armed. UpstreamDrift #11921 consumes
this seam after publication and exact dependency pinning.

## State and Validation

The artifact descriptor binds complete native restart identity and owned
immutable bytes. The separate native envelope binds inventory model identity,
ordered saved inputs, native clock and execution policy without embedding a
numeric T01 bundle or claiming q/v observables represent hidden state. The
1.0 reader remains unchanged. Shared Tools code performs no native decoding.

Run `python3 -m pytest -o addopts= -q
tests/shared/python/sidekick/lab/mocap/test_native_state_artifact.py
tests/shared/python/sidekick/lab/mocap/test_experiment_replay.py` from the repo.
Latest native contract/schema subset: 77 passed. Full mocap suite: 231 passed.
Manual QA/projection: 39 passed; configured source typing passes. Native artifact/envelope tests include failures
recorded before implementation, source ownership, metadata/input integrity,
nonzero clock, duplicate JSON rejection and experiment-ID binding. Astra review
added three reproduced RED tests for channel-array coercion and large-epoch
interval retiming; all pass after correction. The clock gate permits the larger
of 1e-9 relative interval error or 64 interval ULPs, never epoch ULPs. JUnit
receipts are retained in fleet planning staging, not published as native
physics evidence.

## Next Required Work

The first protected CI pass exposed three omissions: the unchanged type-check
loop reused a type-class variable for the integrity result, the renderer test
still listed six sources, and the public API baseline omitted the three new
modules. The fixes preserve existing signatures and add the intended exports.
PDF extraction with pypdf5.7 and6.19 inserts different whitespace on seven
pages. Page character counts now explicitly count non-whitespace characters;
actual hashes, every page, ordering, lines, fonts, outlines, images and
annotations remain verified. A whitespace perturbation test failed before the
fix; both actual extractor versions verify the regenerated QA ledger. This
does not approve the manual or change the rendered PDF bytes.

The native envelope schema and malformed-input negatives are implemented, with
local references to existing stable definitions. Actual pinned rendering yields
12 inspected PDF pages and 12 resolved native fonts. QA rejects incomplete page
inventories; projection schema admits nonempty dynamic page counts while live
QA comparison remains mandatory. Governance and handoff checks pass. Refresh
module inventory after final source edits, run normal gates and publish a PR. The
canonical manual source is `manuals/tools/chapters/05-native-state-replay.qmd`;
it records pending gates and the native provider responsibilities. Verify
decoded class/time/model compatibility and actual R2025b independent replay
in UpstreamDrift; these pure contracts cannot establish it.
