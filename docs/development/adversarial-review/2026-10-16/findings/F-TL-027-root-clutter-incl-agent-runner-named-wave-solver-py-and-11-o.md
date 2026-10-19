# F-TL-027

**Root clutter incl. agent runner named wave_solver.py and 11 other executables**

| Severity | Category | Confidence | Effort | Status   |
| -------- | -------- | ---------- | ------ | -------- |
| ⬜ Low   | hygiene  | high       | S      | proposed |

**Locations:** `wave_solver.py:1`, `UnifiedToolsLauncher.py:1`, `convert_tools_icon.py:1`, `_bootstrap.py:1`, `setup_dev.py:1`

## Finding

- Severity: Low - Category: hygiene
- Confidence: high
- Location: wave_solver.py:1-13; UnifiedToolsLauncher.py:1-20; convert_tools_icon.py; create_launcher_shortcut.ps1; create_launcher_shortcut_png.ps1; run_impact_explorer.bat; launch.py; launch_signal_toolkit.py; verify_launcher.py; \_bootstrap.py; setup_dev.py; start-gaai-daemon.sh
- Evidence: `wave_solver.py` docstring: "Autonomous issue-solving wave runner … The `--dangerously-skip-permissions` Claude flag is opt-in only." In a repo with `src/ode_solver`, this filename will be imported by someone looking for a wave equation.
- Failure scenario: 12 executables at the root of a library; `_bootstrap.py` and `setup_dev.py` mutate `sys.path`; a new contributor cannot tell which of `launch.py`, `UnifiedToolsLauncher.py`, `launch_signal_toolkit.py` is the entry point (`[project.scripts]` lists none of them).
- Proposed change: `scripts/agents/` for agent tooling, `src/gui_launcher/` for launchers, `deploy/windows/` for shortcut scripts; rename `wave_solver.py` → `issue_wave_runner.py`.
- Verification: root has `conftest.py` (and `setup.py` if kept) only.
- Effort: S

## Draft Pull Request

**Title:** `chore(repo): move root launchers and agent runners out of the wheel root`

### Why This Matters (The Pitch)

Twelve executables sit at the root of a library, and one is an autonomous issue-solving agent runner named wave_solver.py in a repo that also has an ode_solver. Someone looking for a wave equation will import it. Three launchers compete as the entry point and none is in project.scripts. Moving them into scripts/agents, a launcher package and a deploy folder, and renaming wave_solver to issue_wave_runner, leaves less to read at the root and one obvious start point.

### Proposed PR Body

## Problem

Root of the repo holds `wave_solver.py:1-13` (an autonomous issue-solving runner), `UnifiedToolsLauncher.py`, `launch.py`, `launch_signal_toolkit.py`, `verify_launcher.py`, `_bootstrap.py`, `setup_dev.py`, `convert_tools_icon.py`, two `.ps1` shortcut scripts, `run_impact_explorer.bat` and `start-gaai-daemon.sh`.

## Change

- [ ] Move agent tooling to `scripts/agents/` and rename `wave_solver.py` to `issue_wave_runner.py`
- [ ] Move launchers to `src/gui_launcher/`
- [ ] Move shortcut scripts to `deploy/windows/`
- [ ] Update references

## Verification

Root contains `conftest.py` (and `setup.py` if kept) only.

## Risk and Rollback

Path-dependent scripts and shortcuts need reference updates. Rollback is a revert.

## Handoff

- **Branch, commit, and pull request:** `<branch>` @ `<commit>` — PR #<n> (fill in when opened)
- **Validation:** the command(s) under _Verification_ above, run locally and green; plus the repo's fast gates
  (`ruff check`, `ruff format --check`, the relevant `-m unit` subset)
- **Blockers:** none known at drafting time — record any that appear
- **Next step:** open the PR against `main`, arm auto-merge with `scripts/automerge_guard.py`, end the session
- **Follow-ups:** none

Refs: adversarial review 2026-10-16, F-TL-027

---

_Generated 2026-10-16 from an adversarial review of `Tools` @ `b630cd3c`. This is a proposal, not a verdict — verify the cited lines before acting._
