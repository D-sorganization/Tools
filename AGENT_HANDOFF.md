# AGENT_HANDOFF — Tools (monorepo root)

> **Update this file with every PR and every push to main.**
> Last updated: 2026-08-09

## 2026-08-09 PR #4203 Python 3.10 string-enum compatibility

Current-head child CI exposed a collection boundary that was already present
in PR #4203's parent surface: seven Rate/shared swing modules imported
`enum.StrEnum`, which does not exist on Python 3.10. Runtime imports now use
the repository's existing `shared.python.compatibility.StrEnum`; type checking
retains the native stdlib symbol behind `TYPE_CHECKING` so pinned mypy 1.13
keeps enum-member types rather than weakening them to strings. No enum values,
serialized contracts, physics, or UI behavior changed.

This repair is based directly on published #4203 head `9dbceff76`. Evidence:
64 focused convention, D-plane, manual-delivery, flight-result, inverse,
impact-family, capability, and compatibility tests pass; Ruff and format pass;
pinned mypy 1.13 passes all eight changed Python files; and a real CPython
3.10.20 probe verifies the shared fallback and all seven runtime import paths.
Propagate the new parent normally through #4279, #4280, #4281, and #4282.

## 2026-08-09 PR #4203 Linux collection repair

Draft PR #4203 remains on `feat/4181-launch-monitor-registry`, based on
`feat/4189-dplane`; no branch was rebased, retargeted, force-pushed, or merged
on GitHub. Exact-head CI run `31199764932` passed the quality gate but all
three Python lanes failed while collecting the in-package flight and solver
facade contract tests. Pytest loaded those tests through its `src.shared...`
package namespace, while their absolute dotted aliases requested the editable
`shared...` namespace; Python then reported that `flight`/`solver` could not be
imported from `src.shared.python.swing_sim` before any assertion ran.

The bounded repair uses package-relative facade imports in those two tests, so
collection and the public API assertions stay in one namespace. It does not
change production code or widen either facade. Verify both contract modules
with `--import-mode=importlib`, Ruff/format, and pinned mypy before a normal
push. The run's Rust `-lpython3.11` link error is missing runner toolchain
state, not simulation evidence; do not modify the model to hide it. After the
new #4203 head passes, propagate it through #4279, #4280, #4281, and #4282 in
normal stack order.

Local evidence is now `12 passed` on Windows and `12 passed` under WSL Python
3.11 with importlib collection. Ruff check/format and pinned mypy 1.13 pass
for both changed test modules. The frozen-dataclass assertion casts only its
introspection target to `Any`, matching the later carrier boundary while
retaining the runtime assertion. The minimal WSL environment reports only
unknown-option warnings for intentionally omitted optional pytest plugins.

## Where This Repo Is Headed

Tools is the D-sorganization fleet's shared engineering-tools monorepo (45+
tools: PyQt6 GUIs, FastAPI/React web mirrors, Rust kernels). The current
center of gravity is `src/rate_of_closure`, being grown from a single
closure-rate calculator into a full swing → impact → ball-flight simulation
platform under **Repository_Management#1390** (this handoff rollout) and a
stack of golf-simulation epics:

| Epic                                                                          | Status (one line)                                                                                                                                                                                                   |
| ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| #4103 — Swing–Impact–Ball-Flight Simulation Platform                          | Phases 0-6 implemented on branch `feat/impact-simulation-platform`, consolidated into PR **#4119** (open, auto-merge armed, awaiting review). Phase 7 (WASM web parity swap, Pages CI) still open.                  |
| #4120 — Investigation & Variation Suite (plotting/viewers/Monte Carlo/help)   | V1-V4 implemented, stacked on #4119, consolidated into PR **#4124** (open, draft-for-review, no auto-merge yet — targets `feat/investigation-suite`, itself stacked on #4119).                                      |
| #4125 — Realistic Clubs/Kinetics/Putting/Public Release Mgmt/Showcase Styling | H1-H7 implemented, stacked on #4124, consolidated into PR **#4129** (open, draft-for-review, targets `feat/course-showcase`, stacked on #4124). H5 (public release-management repo) is cross-repo, not yet started. |
| #4130 — Impact-Interval Club Dynamics (contact-interval rigid-body model)     | Foundation epic only (F1 formulation doc not yet started); no PR yet. Next major physics wave after #4125 lands.                                                                                                    |

The separate shared Club Builder epic #4146 is active. Its first dependency
slice, #4147, lives on `feat/4147-club-builder-core` and establishes the
UI-independent assembly mass/CG/inertia, frame, length-datum, and persistence
contracts that the later shaft, CAD, export, fitting, and UI issues consume.

Active infrastructure repair: #4155 hardens the Rust/PyO3 job against
incomplete setup-python cache entries whose interpreter works but whose
declared link library is missing. The repair is isolated on
`fix/4155-rust-libpython-cache` and does not change simulation code.

See `src/rate_of_closure/AGENT_HANDOFF.md` for the detailed stack breakdown
and architecture pointers for this tool specifically.

## Must-Read Architecture Pointers

1. `CLAUDE.md` — repo-wide conventions, CI gate list, cross-repo dependency
   rules (Tools is a leaf dependency; UpstreamDrift and Gasification_Model
   consume it).
2. `docs/architecture/CANONICAL_TOPOLOGY.md` — canonical repo topology policy.
3. `SPEC.md` — living specification; §12 Change Log requires a dated row for
   every PR touching `src/` (enforced by `spec-check.yml`, see gates below).
4. `src/rate_of_closure/AGENT_HANDOFF.md`, `src/pendulum_simulator/AGENT_HANDOFF.md`,
   `src/rotation_converter/AGENT_HANDOFF.md` — per-tool handoff docs.
5. `docs/AGENT_HANDOFF_TEMPLATE.md` — template for adding a handoff doc to a
   new tool.

## In-Flight Branches (what stacks on what)

```
main
 └─ feat/impact-simulation-platform   (PR #4119, epic #4103, auto-merge armed)
     └─ feat/investigation-suite      (PR #4124, epic #4120, stacked on #4119)
         └─ feat/course-showcase      (PR #4129, epic #4125, stacked on #4124)
docs/agent-handoff-1390               (this branch, off origin/main, Repository_Management#1390)
```

Other active non-golf branches worth knowing about: `fix/file-size-budget-bounded-checkout`
(#4096, CI checkout-scope fix), `agent/scada-phase-a-foundation` (#4091, SCADA
epic #4085), several `scada/pr*` branches (SCADA epics #4085-#4089), and a
handful of Bolt/Palette/Sentinel micro-PRs (#4070-#4102) unrelated to the
golf-sim stack.

## Gate Commands (repo-wide)

```bash
python3 -m ruff check .                          # lint
python3 -m ruff format --check .                  # format check
python3 -m pytest -n auto --timeout=60            # full test suite
python3 -m pytest -m contract                     # API contract tests (downstream-facing)
python3 -m pytest -m integration --timeout=60     # cross-repo integration
```

SPEC freshness (CI job `spec-freshness` in `.github/workflows/spec-check.yml`):
any PR touching `src/**`, `tests/**`, `config/**`, `pyproject.toml`,
`Cargo.toml`, `package.json`, or `requirements.txt` must also modify
`SPEC.md` in the same PR, or carry the `spec-exempt` label. Runs on the
`d-sorg-fleet` self-hosted runner.

## Do-Not List

- Do not modify public function signatures in `src/shared/python/**` without
  opening coordinated migration issues in UpstreamDrift and Gasification_Model
  (see `CLAUDE.md` Cross-Repo Dependencies).
- Do not import across package boundaries (e.g. `signal_processing_studio`
  importing from `sidekick.process_calculators`) — LoD is enforced.
- Do not exceed the 500-LOC file budget on new/modified files in the golf-sim
  packages (`rate_of_closure`, `swing_sim`, `swing-core`) — sub-package,
  don't grow monoliths.
- Do not use `git commit --no-verify` / `--push --no-verify` to bypass hooks;
  see `CLAUDE.md` Hook bypass policy.
- Do not regenerate the sidekick API baseline (`tests/sidekick_api_baseline.json`)
  without coordinating a breaking-change migration.
- Do not merge #4124 or #4129 ahead of their base (#4119, #4124 respectively)
  — they are stacked and will conflict/duplicate SPEC.md sections if merged
  out of order.
- Do not hand-roll a GitHub Pages deploy workflow for `rate_of_closure/web`
  yet — Phase 7 of #4103 owns this; today Pages hosting elsewhere in the repo
  (e.g. `unit_converter`) is done via manual branch-folder publish, not CI.

## Short-Term Roadmap (ordered)

1. Land PR #4119 (base platform) — currently the long pole; everything else
   stacks on it.
2. Get #4124 out of draft-for-review and merge into `feat/investigation-suite`
   → cascades onto #4119.
3. Get #4129 out of draft-for-review and merge into `feat/course-showcase`
   → cascades onto #4124.
4. Start #4130 Phase F1 (formulation document) once #4125's stack is in.
5. Phase 7 of #4103: WASM swap for the web mirror + real Pages CI deploy for
   `rate_of_closure/web`.
6. #4125 H5: stand up the public release-management repo (cross-repo, not
   started).
