# Rate Qt Test Isolation: Tools #5114

This delivery prerequisite addresses native widget ownership between tests.
It changes the Rate test harness, not the impact model or application shutdown
policy. Classifier PR #5103 merged at 33144678cb on 2026-09-09; its final
Python 3.12 Rate shard was cancelled after another Club Tester worker loss.
Neither that merge nor the focused ownership regression qualifies this repair.

## Evidence and limits

The preserved Python 3.12.14 environment uses Qt 6.11.2, PyQt6 6.11.0 and 104
package versions recovered from the installation records of CI job 102338506622.
This recovers package versions, not full system-library or base-image parity.
Exact archived source 32c7b38cb supplied the following controls. Each full run
used four xdist loadscope workers, coverage and the unchanged 60-second thread
timeout; the diagnostic outer cap was 900 seconds.

| Run                                              | Result                                                            | Duration | Interpretation                                                                             |
| ------------------------------------------------ | ----------------------------------------------------------------- | -------- | ------------------------------------------------------------------------------------------ |
| Original full suite                              | 2,096 passed, 28 skipped; worker loss then xdist scheduling error | 680.62 s | Wind-strategy test call had passed before worker loss; cause unresolved                    |
| Separate worker phase traces; stop on first loss | 2,649 passed, 29 skipped; one worker-loss failure                 | 451.39 s | Glossary test setup passed; 40-second trace shows Qt event processing during call          |
| External deferred-deletion experiment            | 2,889 passed, 29 skipped                                          | 514.56 s | Supports testing an explicit cleanup patch; does not identify every earlier native failure |

The full runs collect 2,917 items. The reported skip total also includes a
collection-time optional h5py skip. Smaller controls also pass unchanged:
the wind-strategy closing case alone, its ten-case file, and that file after
its two immediate predecessor files (30 cases). Thus passing an isolated GUI
case did not qualify full-suite behavior.

A separate three-case Qt probe keeps Python wrappers alive intentionally.
After qtbot schedules native widget deletion, the next case still sees the
native object alive: one failure and two passes in 0.13 s. Explicit delivery
of deferred-delete events makes all three pass in 0.18 s. This establishes a
specific isolation gap; the retained wrappers make the check independent of
garbage-collector timing.

Qt documents that repeated `processEvents()` calls without an event loop do
not deliver deferred-delete events and identifies `sendPostedEvents()` as a
way to deliver them. The existing `pyqt_probe_lifecycle.shutdown_probe` already
uses that operation for rendered subprocess probes.
[Qt primary documentation](https://doc.qt.io/qt-6/qcoreapplication.html).

## Repository change and TDD

The Rate directory's `pytest_runtest_teardown` hook runs after fixture
finalizers, inside pytest-qt's exception-capturing wrapper. It delivers already
queued deferred deletions when an application exists. It does not create an
application, close arbitrary windows, disable test assertions, change test
deadlines or introduce a sleep. Missing optional Qt is a no-op for non-GUI
environments. The operation is scoped to Rate tests.

The regression executes a copy of the actual Rate fixture module in a fresh
tiny pytest process. It verifies that non-GUI cases create no application and
that both a registered widget and its owned timer are natively deleted before
the next case, even with retained Python wrappers. Before the hook, that child
suite reports one failure and three passes; the parent regression fails in
47.39 s including cold imports. After the hook it passes. A second regression
raises intentionally in a destruction callback and requires a reported pytest
teardown error, preserving Qt exception capture rather than hiding errors.
Both Windows regressions pass in 13.89 s; scoped Ruff and actual mypy 1.13 pass.

Final review added the usual module-level optional-dependency skips. An
external collection oracle initially failed because the two GUI regressions
were collected with PyQt6 unavailable; it now confirms a reported module skip
when either PyQt6 or pytest-qt is refused separately. The oracle is preserved
as `C:/Users/diete/AppData/Local/Temp/impact-optional-qt-collection-check.py`
with `impact_optional_qt_block.py`. The first oracle expected a skip-count
summary that collect-only mode does not print; its final assertion checks the
explicit skip reason and pytest's no-tests exit instead. Both real Windows
regressions pass again in 17.10 s; scoped Ruff and actual mypy pass. These guards
were added after the full-run snapshot below; the cleanup hook is unchanged.

The repository hook deliberately runs inside pytest-qt's wrapper so deletion
exceptions remain captured. The earlier external experiment delivered deletion
after the complete wrapper. Consequently the external pass is supporting
evidence, not a substitute for validation of the actual repository patch.

## Exact candidate and remaining gates

The candidate worktree is `Tools-impact-rate-isolation`, branch
`fix/5114-qt-deferred-cleanup`, based on then-current main
184e453dbc6cc6999b9309a6076b62604699a46b. Historical `Tools-impact-rate-lifecycle`
and all previous native reproduction sources are preserved.

A fresh native archive of that main commit plus the exact two-file test patch
ran the complete Rate suite with the original four-worker, covered
configuration and no external diagnostic plugin. It failed: 2,880 passed,
29 skipped and three worker-loss failures in 901.23 seconds. The outer
900-second diagnostic cap interrupted the remaining run. Lost workers were
assigned the minimum-window-size, current-tab help and tooltip-completeness
cases; the tooltip call had reported a pass before its worker was lost.
These observations do not distinguish expensive event processing from a
native lifecycle defect. Native profiling controls are recorded below.
The base archive SHA-256 is
70a6cd2f1df480e391ee99a02f365b50f19ae5ef9ea81b7c5359a1a364b468cf.
The exact two-file full-run patch SHA-256 is
04bad20af8768430577b2d115d4a40c382ebcdcfc73f84ada921a001b616d9e5.
The focused Windows regressions, actual mypy, root Ruff and nine manual gates
passed before these evidence updates; all nine manual gates passed again after
the profiling handoff. The full candidate did not qualify.
Normal hooks and protected CI remain pending. Do not close #5114 or call every
prior worker loss explained from the probe or external experiment alone.

Evidence is preserved outside git under
`/home/dieterolson/.cache/codex-impact/rate5114-py312/`: `full-rate-covered.log`,
`full-rate-firstloss.log`, `phase-traces-firstloss/`, `full-rate-flush.log`,
`phase-traces-full-flush/`, and candidate `main184-qt-cleanup.log` with its
`source-184e-qt-cleanup` source snapshot. Issue comments 5597785150 and
5598134846 record the failed controls and cleanup experiment. No physical,
acoustic or perceptual result follows from these test-harness checks.

## Native profiling controls and offline handoff

py-spy 0.4.2 was installed in a separate `profile-venv`; the recovered
104-package test environment is unchanged. The profiler launched its own
Python child, with the existing Linux ptrace policy unchanged. Commands used
`record --native --threads --rate 20 --format raw` and the original covered
test configuration with a 60-second thread deadline. Serial controls use
`-n0`; they do not reproduce four-worker scheduling or resource contention.
[Profiler primary documentation](https://github.com/benfred/py-spy/blob/master/README.md).

- The minimum-window-size case reports one pass in 13.95 s (6.90 s setup,
  2.01 s teardown), with 274 captured samples and zero sampling errors.
- All 14 cases in `test_layout_minsize`, `test_help`, `test_tooltips` and
  `test_club_tester_tab` report passes in 224.51 s, with 41 warnings and
  4,188 captured samples. The profiler reports sampling lag, sometimes over
  11 seconds; these durations and sample counts are not unbiased benchmarks.
- Inclusive stacks contain repaint/flush activity (876 samples) and native
  QWidget destruction (522 samples); the explicit cleanup hook appears in
  542 samples. Counts overlap. This identifies activity worth examining,
  not the cause of full-suite worker losses or evidence of a native deadlock.
- Both serial profilers wrote their raw data but returned exit 1 afterward
  with `No child process (os error 10)`. The pytest pass reports are retained
  separately from the unsuccessful profiler command status.
- The initial native `--subprocesses` profiler itself segfaulted. Its pytest
  child was verified and interrupted explicitly; that partial run reports
  nine passes in 51.80 s and is not qualification evidence.

Logs and raw samples are `profile-layout.{log,raw}` and
`profile-gui-serial.{log,raw}` in the preserved native directory above;
`profile-gui.log` records the interrupted multi-worker attempt. The external
summary script is `C:/Users/diete/AppData/Local/Temp/impact-profile-summary.py`.
No additional application or test deadline changes followed these profiles.

As of 08:40 UTC on 2026-09-09, GitHub CLI returned HTTP 401; the connected GitHub
app also reports that reauthentication is required. An authentication request
is pending. Issue updates, publication and lease renewal are unavailable.
The existing #5114 lease expires at 09:55:42 UTC; the separate #5072 shaft
lease expired at 08:36:26 UTC, so no new shaft implementation was started.
Once access returns, recheck ownership, publish this evidence, integrate
33144678cb where appropriate, and qualify any complete #5114 repair against
the full covered four-worker suite. Physical and blinded evidence remain
separate unanswered requirements of the impact program.
