# Putting Launch Monitor — Turnover for Completion (epic #5218)

Written 2026-09-15 by the lead session after every code child of the epic had
merged. Read this first, then `src/putting_launch_monitor/AGENT_HANDOFF.md`
(per-tool current state, gate commands, do-not list) and
`src/putting_launch_monitor/README.md`.

## 1. What exists on `main` (Tools 83ac11cb9 and later)

| piece                                                         | where                                                  | proof                                                                              |
| ------------------------------------------------------------- | ------------------------------------------------------ | ---------------------------------------------------------------------------------- |
| Ground-plane homography, launch fit, HLA                      | `src/putting_launch_monitor/geometry.py`               | synthetic 30 deg camera: speed within 2 %, HLA within 0.5 deg                      |
| GSPro Open Connect v1 codec + client                          | `gspro.py`                                             | spec fields, 200/201/5xx, glued-reply framing, putting mode from the 201           |
| HSV ball detector, search region from the mat                 | `detect.py`                                            | lab frame: mat region yields the real balls; whole frame yields 59 blobs           |
| rest -> armed -> rolling tracker                              | `track.py`                                             | rendered putt recovered end to end; stray balls cannot steal the roll              |
| Calibration document `putting_monitor.calibration/1`          | `calibration.py`                                       | round-trips; fail-closed on unknown fields                                         |
| Orchestration and sinks (`LogSink`, `GsproSink`)              | `monitor.py`                                           | end to end on rendered frames                                                      |
| CLI `calibrate / run / replay / probe-gspro / snapshot`       | `cli.py`                                               | live: 55 fps processed, ball in 240/240 frames, armed at frame 11, 0 false         |
| PyQt6 window: live view, calibration wizard, HSV, GSPro panel | `ui/pyqt6/`                                            | 63 offscreen tests                                                                 |
| Launcher tile **Putting Launch Monitor** (Biomechanics)       | `gui_registration.py`, `launch_pyqt6.py`, `tools.json` | `python scripts/generate_tools_json.py --check`                                    |
| Shared live-camera layer                                      | `src/shared/python/camera/`                            | streams the overhead ELP at 1920x1200@60; UpstreamDrift rig adopted (UD PR #10211) |

Tests: `python -m pytest src/putting_launch_monitor/tests tests/camera -q --timeout=60`
(77 pass on main). Repo venv: `C:\Users\diete\Repositories\Tools\.venv`.

Installed lab calibration:
`%LOCALAPPDATA%\D-sorganization\putting_launch_monitor\calibration.json` —
corners (693,983) (1143,983) (1118,518) (735,518) on camera
`USB\VID_32E4&PID_5234&MI_00\9&2A7EE39F&0&0000`. **The mat size is a
placeholder 1219 x 1524 mm and every speed scales with it.**

## 2. What remains, in order

| #   | issue                                                                    | kind                            | who                          |
| --- | ------------------------------------------------------------------------ | ------------------------------- | ---------------------------- |
| 1   | #5227 lazy `FrameSource` import in `shared/python/camera/__init__.py`    | small code change               | agent                        |
| 2   | #5221 accuracy validation harness (`validate` subcommand, CSV, evidence) | code, then bay time             | agent builds, operator rolls |
| 3   | #5228 shared Open Connect v1 codec (Tools leaf; UD #10208 duplicates it) | refactor                        | agent                        |
| 4   | #5222 replay corpus (clips, manifest, parametrised regression test)      | code, needs recordings from (2) | agent after the operator     |

Each issue body carries the full specification and acceptance criteria. This
document adds only what an agent cannot get from the issue.

### 2.1 #5227 — lazy import (start here)

- `dshow.py` already has no `sidekick` import. Only `ffmpeg_source.py` and
  `video_file_source.py` import `sidekick.lab.mocap.acquisition.FrameSource`.
- Use PEP 562 module `__getattr__` in `camera/__init__.py` for the two source
  classes; keep `__all__` unchanged so
  `from shared.python.camera import FfmpegDirectShowSource` still works.
- Test by blocking the import
  (`monkeypatch.setitem(sys.modules, "sidekick", None)` and the acquisition
  module), reloading the package, and importing the four builders; a second
  test asserts `issubclass(FfmpegDirectShowSource, FrameSource)` when the ABC
  is importable.
- Check `tests/api_baselines/` for a `camera` baseline before changing any
  signature; there was none on 2026-09-15.

### 2.2 #5221 — validation harness, then the operator's session

Build first, no hardware needed:

- A `validate` subcommand (keep `cli.py` under 500 lines — add `validate.py`
  if needed) that runs `PuttingMonitor` with an observer which, on each
  accepted `Putt`, prompts for the operator's reference value(s), appends
  `timestamp, speed_mph, hla_deg, points, r2, span_mm, ref_speed_mph, ref_hla_deg, note`
  to a CSV and prints the running mean and spread of the error. Rejected
  putts are logged with their `reason`.
- Unit-test it with `RenderedPuttSource` from `tests/test_pipeline.py` and a
  scripted stdin.
- Write `docs/putting_launch_monitor/validation.md` with the empty evidence
  table, the ramp / Stimp method from the issue, and the exact commands.

Then give the operator the checklist in section 4 and wait; fill the evidence
page from the CSV they produce.

### 2.3 #5228 — shared codec

- Put the codec under `src/shared/python/launch_monitor/` only if its API
  baseline (`tests/api_baselines/`) can be bumped in the same PR with a `!`
  title; otherwise a new sibling package `src/shared/python/gspro_connect/`
  is acceptable — say which in the PR.
- `putting_launch_monitor/gspro.py` keeps `PuttShot`, `GsproClient` and putter
  detection and delegates `encode_shot / encode_heartbeat / parse_reply /
split_objects`. `src/putting_launch_monitor/tests/test_gspro.py` must keep
  passing with its intent unchanged.
- Pin the exact wire bytes for one putt and one full swing in a contract test.
- File the UpstreamDrift follow-up for `golf_simulator/adapters/gspro` and
  link it from #5228; do not edit UpstreamDrift.

### 2.4 #5222 — corpus

Needs the operator's recordings from the validation session (UpstreamDrift
rig: `rig record --camera cam_b=<id>`). Trim to ~2 s, downscale to the 960 px
decode width, MJPEG, under 5 MB each; `tests/data/manifest.json` with expected
speed, HLA, tolerance and the calibration used; a parametrised test through
`VideoFileSource` + `PuttingMonitor`; mark `slow`.

## 3. Landing recipe (every trap hit so far)

1. One worktree per issue:
   `git worktree add ../_wt_claude_<issue> -b claude/<issue>-<slug> origin/main`.
   Never `git add -A` (untracked recordings and worktrees). Never draft PRs.
2. Post a lease before editing (from Repository_Management:
   `python -m scripts.post_agent_lease --agent claude --session <id> --repo Tools --issue <N>`).
3. Every commit touching `src/` needs: one SPEC.md section 12 row keyed by
   your PR number, staged in that same commit (the guardrail insists — a
   wording touch to your own row is enough on later commits);
   `python -m scripts.build_tools_module_inventory` run with the venv and
   `manuals/tools/manifests/module-inventory.json` staged;
   `docs/development/DEVELOPMENT_LOG.md` entry `DL-#5218` with `Last verified`
   refreshed in place, or the phrase
   `No material development-log change — <reason>` staged.
4. After editing any `AGENT_HANDOFF.md`:
   `python -m scripts.check_tools_handoff --generate` (the manifest is
   hash-pinned); the root handoff stays at or under 150 lines.
5. Fixture-only test modules (no `assert`) go in
   `scripts/test_assertion_allowlist.txt`.
6. mypy: numpy aliases must be declared `Frame: TypeAlias = ...`; `require()`
   needs a Python `bool`, not `numpy.bool_`; CI's OpenCV stubs type
   `cv2.imread` as `Any` — annotate the local before returning it.
7. Pre-push: the pinned mypy hook crashes on every numpy-importing file
   (Tools #5223) — push with `SKIP=mypy git push`; every other hook must run.
   The `language: system` hooks (`tools-module-inventory-freshness`,
   `design-manual-governance`) run under system Python 3.13 and rewrite the
   inventory schema; if they fail on files you did not touch, skip them and
   say so in the PR body.
8. **After merging `origin/main` into your branch, regenerate
   `module-inventory.json` and run
   `tests/architecture/test_phased_production_inventory.py`.** A merge with
   hooks off leaves stale shard digests and CI fails with "shard digest
   differs".
9. Stacked PRs: after the base squash-merges, `git merge origin/main` (never
   rebase) and check `git diff --stat origin/main..<branch>` for unintended
   deletions before pushing.
10. CI status over REST only:
    `gh api repos/D-sorganization/Tools/commits/<sha>/check-runs --jq ...`.
    Required checks are `quality-gate` and `tests (3.11)`; the rest are
    advisory. There is no standalone `jq` on this machine — filter inside
    `gh --jq`. Poll at 5 minute intervals or longer; auto-merge
    (`gh pr merge --auto --squash`) does the merging.
11. CI's shallow runner falls back to a direct diff against `origin/main`
    when it has no merge-base; if main moved after your last merge, files
    added there look "deleted" and the tests lane fails — merge main again.

## 4. Operator checklist (in the bay)

1. Tape-measure the lighter hitting mat: width across the stance line and
   length toward the target, in mm.
2. `python -m putting_launch_monitor snapshot --camera "USB\VID_32E4&PID_5234&MI_00\9&2A7EE39F&0&0000" --out frame.jpg`;
   read the four corners (near-left, near-right, far-right, far-left as the
   player sees them) or use the wizard in the launcher tile; then
   `calibrate --corners ... --mat-mm <W> <L>`. Note the reprojection error.
3. Start GSPro (the game owns port 921; `GSPconnect.exe` alone listens on
   1250). `python -m putting_launch_monitor probe-gspro`; on a putting hole,
   `run --gspro`; record the 201's `Club` value and set it as the putter code
   if it is not `PT`.
4. Roll ten putts at each of three speeds from a fixed ramp height (calibrate
   the ramp once with a stopwatch over a taped 1 m), and along taped lines at
   0, +/-5 and +/-10 degrees, with `validate` running; keep the CSV.
5. Record the same session with the UpstreamDrift rig for #5222.
6. Confirm the rig preview still binds all three cameras at 60 fps after
   UD #10211 (the shared builder adds `-rtbufsize 256M` to the preview
   command, which the old rig path lacked).

## 5. Definition of done for #5218

- #5227 and #5228 merged; #5221's evidence page states a measured tolerance
  (target +/-3 % speed, +/-1 deg HLA at 1-4 mph) or a documented reason with
  a follow-up issue; #5222's corpus test runs in the `slow` lane.
- Tool and root `AGENT_HANDOFF.md` current and `DL-#5218` set to `shipped`;
  the epic closed with a comment linking the evidence page.
