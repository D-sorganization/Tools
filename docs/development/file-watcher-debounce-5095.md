# Deterministic Watcher Debounce — #5095

The Rust quality job in Tools run `34217794944`, job `102033732755`,
reported four callback flushes where `debounces_rapid_changes` required at
most three. Ten writes separated by five-millisecond sleeps cannot guarantee
one burst on a loaded runner: scheduling pauses can exceed the configured
quiet interval. Increasing sleeps or the allowed count would hide that problem.

The private `DebounceBatch` owns coalescing and quiet-period state. The watcher
supplies real monotonic timestamps; eight unit tests supply exact timestamps.
They verify latest-value replacement for each path/kind, distinct identities,
restart, exact deadline and no early emission, one-time draining, empty batches,
filtered-notification timing, shutdown, zero delay and backward-time refusal.
No asynchronous test depends on a particular number of scheduler wakeups.

The watcher retains its existing classification, gitignore policy, poll interval,
callback and shutdown semantics. Classified notifications restart the timer even
when all their paths are filtered, matching the original implementation. Four
real filesystem tests remain, moved unchanged into `watcher_tests.rs`; the
former timing-sensitive debounce test alone is replaced. Public exports and
Python bindings do not change. The generic accumulator has no watcher dependency.

## Validation

The first draft had a test-name syntax error, which is not RED evidence. After
correcting it, the contract tests failed on the missing `DebounceBatch` import.
Implementation then passed all twelve tests, including the retained filesystem
tests. After extracting event accumulation and callback dispatch into small
helpers, the same default-feature and Python-feature suites pass.

- `cargo fmt --all -- --check`
- `cargo clippy -p file_watcher --all-targets -- -D warnings`
- `cargo test -p file_watcher`
- `cargo test -p file_watcher --features python`

Local verification uses Windows, Rust 1.95.0 and Python 3.12 for PyO3. Linux
workspace CI remains the platform-wide gate. No checks are skipped, no tolerance
is relaxed, and successful local tests do not establish a passing remote job.

## Delivery Boundary

This is a CI prerequisite for renderer PR #5090 and impact/acoustics #5068.
The renderer's production Linux visual workflow `34217794994` passes. Separate
consumer failures remain: UpstreamDrift #9783 tracks strict reviewed-reference
compatibility; private Gasification checkout returns 404 before tests and awaits
the existing automation credential configuration. None is a shaft/contact or
acoustic model result. Update this file and the root handoff with actual PR/CI
evidence as delivery proceeds.

PR #5097 published implementation `c0f277406c526c4172e3cb15e8fdd6da649ada87`
with every applicable normal commit/push hook passing. Main then advanced to
`86a725c6c2d4765d61aa9e7c59d771c3e075f6bd` with the early IA-T5 ingestion
module. The merge retains that implementation and resolves only the adjacent
root handoff and its digest; both SPEC rows remain. Inventory is regenerated
from the combined tree before revalidation. The initial PR was conflicted and
therefore had no Rust quality result; it is not a failed or passing Rust run.

The first merged-head push was refused by normal hooks: Prettier reformatted
main's incoming swing_sim API JSON and mypy reported an unused ignore in the
incoming ingestion test helper. Parsed JSON before/after is exactly equal.
The helper's adversarial keyword dictionary now explicitly permits invalid
contract inputs via Any and no longer suppresses a nonexistent diagnostic.
The actual mypy hook and all eighteen ingestion tests pass. #5074 was unclaimed
and is leased to codex for this bounded prerequisite. No runtime calibration,
alignment, spectral algorithm or source-provenance gate changed in this repair.

Corrected head `e0369fdda` is published with every normal push hook passing.
Main subsequently merged IA-T6 reporting as `cfca06449`; this second main merge
retains its exact implementation and all existing T5/runtime corrections. The
root epic row now records both early T5 and T6 foundations as landed while the
full impact program remains open. Inventory is regenerated from the final merge.
