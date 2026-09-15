# Agent Handoff: Putting Launch Monitor

Updated: 2026-09-15 (session claude, epic #5218; GUI #5219; registration #5220)

## Repository and Working Directory

- Repository: `D-sorganization/Tools`, tool at `src/putting_launch_monitor/`,
  shared camera layer at `src/shared/python/camera/`.
- Branch `claude/putting-launch-monitor` (core, PR #5224, merged); GUI on
  `claude/5219-putting-gui` (PR #5225, merged); launcher registration on
  `claude/5220-putting-register` (PR #5226).
- Governing epic: #5218. Objective: a camera-based putting launch monitor
  that measures launch speed and HLA on the ground plane and feeds GSPro.

## What Is Done (and Measured)

| piece                                                     | module                  | evidence                                                                                                     |
| --------------------------------------------------------- | ----------------------- | ------------------------------------------------------------------------------------------------------------ |
| Ground-plane homography, launch fit, HLA                  | `geometry.py`           | synthetic 30° camera, 0.3 px noise: speed within 2%, HLA within 0.5° over 12 speed/angle combinations        |
| GSPro Open Connect v1 codec + client                      | `gspro.py`              | every required field per the published spec; glued 200+201 replies framed; putting mode from the 201         |
| Ball detection (HSV, circularity, size from mat scale)    | `detect.py`             | on the lab frame the three most circular blobs were the three balls nearest the mat                          |
| Rest → armed → rolling state machine, own-ball following  | `track.py`              | rendered putt recovered end to end; a stray resting ball cannot steal the roll                               |
| Calibration document (`putting_monitor.calibration/1`)    | `calibration.py`        | round-trips; refuses unknown schema and fields; default search region = mat + 25%                            |
| Orchestration and sinks                                   | `monitor.py`            | end-to-end on rendered frames; GSPro sink holds putts outside putting mode                                   |
| Shared camera sources                                     | `shared/python/camera/` | `FfmpegDirectShowSource` streams the overhead ELP at 60 fps; `VideoFileSource` replays files                 |
| CLI: calibrate / run / replay / probe-gspro / snapshot    | `cli.py`                | live run: 55 fps processed, ball in 240/240 frames, armed at frame 11, 0 false putts                         |
| PyQt6 window (#5219): live view, wizard, HSV tuner, GSPro | `ui/pyqt6/`             | 22 offscreen tests: wizard equals a hand-built calibration; rendered putt reaches the readout via the worker |
| Launcher tile (#5220): Biomechanics, beta                 | `gui_registration.py`   | `PuttingMonitorWindow` constructs through `make_launcher` offscreen; `generate_tools_json.py --check` fresh  |

Tests: `src/putting_launch_monitor/tests` (55) and `tests/camera` (8), all
passing; ruff, ruff-format and mypy clean; every file under Tools' 500-line
budget.

## GUI (`ui/pyqt6/`, entry `main_window.main`)

`main_window` (window, start/stop, wizard and tuner launch), `worker`
(`MonitorWorker` thread, `EventBridge`, `take_snapshot`, `build_monitor`
using `cli.scaled_calibration`), `live_view` (`ImageCanvas`, `LiveView`,
`CornerCanvas`), `readout`, `calibration_wizard` + `wizard_pages` (camera /
corners with `MatDiagram` / mat / review), `hsv_tuner`, `gspro_panel`,
`painting` (theme pens via `shared.python.theme.Colors`, image conversion).
Camera selection is a text field seeded from the calibration; a picker is a
follow-up. Launcher-registered (#5220): `gui_registration.py` + thin
`launch_pyqt6.py`; no `[project.scripts]` entry because the package imports
OpenCV at module level.

## The Lab Setup, As Calibrated

- Overhead camera: `USB\VID_32E4&PID_5234&MI_00\9&2A7EE39F&0&0000`
  (cam_b of the UpstreamDrift rig), 1920x1200@60 MJPG, facing the player at
  about 30°. Player putts from the bottom of the image toward the net at the
  top, so player-right is image-right.
- Mat corners read off a pixel grid (±10 px): near-left (693, 983),
  near-right (1143, 983), far-right (1118, 518), far-left (735, 518).
- **The mat size is a placeholder (1219 x 1524 mm, a 4x5 ft mat).** Every
  speed scales with it. Measure the lighter hitting mat with a tape and
  re-run `calibrate`. The installed calibration says so in its `notes`.
- Installed at `%LOCALAPPDATA%\D-sorganization\putting_launch_monitor\calibration.json`.
- GSPro is at `C:\GSProV1`. Port 921 is opened by the GSPro game itself, not
  by `GSPconnect.exe` (which listens on 1250 standalone), so `probe-gspro`
  only succeeds with GSPro running. Not yet exercised against real GSPro.

## Gate Commands

```bash
python -m pytest src/putting_launch_monitor/tests tests/camera -q --timeout=60
python -m ruff check src/putting_launch_monitor src/shared/python/camera
python -m ruff format --check src/putting_launch_monitor src/shared/python/camera
python -m mypy src/putting_launch_monitor src/shared/python/camera
python -m scripts.build_tools_module_inventory --check
python scripts/generate_tools_json.py --check
python scripts/check_tools_manifest_layout.py
python shared_scripts/fleet_hooks.py fast
```

## Do Not

- Do not compute HLA in image space or scale distance off the ball's pixel
  radius (the community reference does both; both are wrong at 30°).
- Do not send to GSPro outside putting mode unless the operator overrides.
- Do not open the camera with `cv2.VideoCapture`; use the shared source.
- Do not add colour literals to GUI code; use `shared/python/theme`.
- Do not `git add -A` in this repo.
- Do not detect, track or decide in `ui/`: the window consumes
  `FrameEvent`s (now carrying `candidates` and the decoded `frame`) from
  `PuttingMonitor.run` on a `QThread`; `worker.EventBridge` coalesces frames
  and never drops a putt. Stop the monitor before a snapshot (single-open).
- Do not run the GUI tests without `QT_QPA_PLATFORM=offscreen`; hold the
  `QApplication` at module level (a dropped one aborts pytest silently).

## Ordered Next Steps (agents; each has an issue under #5218)

1. **Accuracy validation on the rig** — measure the mat, re-calibrate, roll
   putts of known speed (a known roll-out on a known Stimp gives launch
   speed; a taped line gives HLA), state the tolerance, write an evidence
   page. Exercise `--gspro` against the running GSPro and record the 201.
2. **Replay corpus** — record putts from the overhead camera with the
   UpstreamDrift rig, keep short clips plus expected results as regression
   tests through `VideoFileSource`.
3. **UpstreamDrift adoption** — the capture rig's `preview_source.py` and
   `recorder.py` device-ref code becomes `shared.python.camera` via the
   `vendor/ud-tools` pin (Tools is the source of truth).
