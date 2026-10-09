# Native State Artifact and Replay Envelope Turnover

## Development Checkpoint

PR #5471 merged the implementation at `89415ee859d6fcfe3384b95e5a53691e68bf9c1b`.
The retained T01 checkout is reused for issue #5472 in PR #5473 on
`fix/canonical-manual-count-5472`. No new clone or native environment was created.
UpstreamDrift #11921 consumes the seam through an exact merged dependency pin.
The follow-up changes the governance test and handoff evidence only.

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

Protected docs job 113769766442 passed governance commands but failed the fixed
six-source test assertion; seven canonical QMD sources exist. The failure was
reproduced locally. The repair enumerates canonical sources, requires the
authority index and retains all release/publication assertions. Its first
publication completed normal hooks after PR5471 had already merged, so #5472
provides a focused main-based follow-up. Generated artifacts and transport code
remain unchanged. The handoff manifest is regenerated through its existing tool.

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

## PR5473 Extractor Regression

The subsequent documentation job failed before validation because its focused
environment omitted `defusedxml`, now imported by the QA contract. The workflow
installs `defusedxml==0.7.1` explicitly alongside its other governance dependencies.
This dependency repair preserves all validation steps and approval blockers.

CI on both Python3.11 and3.12 exposed pypdf6.20 body-line rewrapping on
unchanged PDF pages2,6,7. Byte identity, normalized character counts and every
other page inventory field matched the retained ledger. A body-rewrapping
regression failed before marking line counts as diagnostic observations, then
all29 QA tests passed with both local pypdf5.7 and isolated CI pypdf6.20.
Canonical rendering chapter2 documents this boundary. The pinned toolchain
verified before regenerating its four renderer-owned representations and QA
ledger:12 PDF pages inspected,0 uninspected,12 fonts and77 outlines. Both
release blockers remain. This is a qualified render and automated QA receipt,
not human page/accessibility approval. Normal governance/handoff checks apply.
