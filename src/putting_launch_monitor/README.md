# Putting Launch Monitor

A camera-based putting launch monitor for [GSPro](https://gsprogolf.com). An
overhead camera watches the putting surface; a putt's **launch speed** and
**horizontal launch angle** (HLA) are measured on the ground plane and sent to
GSPro over its [Open Connect v1](https://gsprogolf.com/GSProConnectV1.html)
API. Launch monitors such as the Rapsodo MLM2Pro do not see putts; this fills
that gap so putting happens inside the same round as full shots.

Epic: D-sorganization/Tools#5218.

## How It Measures a Putt

1. **Calibrate once.** The hitting mat is a rectangle of known size. Its four
   corners in the image, in the order near-left, near-right, far-right,
   far-left _as the player sees them_, define a homography from pixels to
   millimetres on the green. The world frame is `x` to the player's right,
   `y` toward the target; a camera facing the player needs no flip flag —
   the corner order absorbs it.
2. **Find the ball.** HSV threshold, morphology, the most circular blobs of
   plausible size inside the mat's region. Radius bounds follow the mat's
   scale in the image, so nothing is tuned by hand.
3. **Wait for it to rest, then follow it.** The ball must sit still for a
   dozen frames before it is _armed_; a hand, a club head or a ball rolling
   through never counts. With several balls on the mat the tracker follows
   its own — the one at its resting spot, then the one nearest where it last
   was.
4. **Fit the launch.** Over the first 300 mm of travel, a straight-line fit
   of distance against time gives launch speed (with an `r²` quality
   figure); the principal axis of the track gives direction. Many frames,
   not two gate crossings, so one bad detection cannot decide a shot.
5. **Tell GSPro.** Speed in mph and HLA in degrees (positive to the right),
   with the shot number, heartbeat and 200/201 framing the spec requires.
   Shots are held unless GSPro reports a putter in hand (`Player.Club`),
   so a stray roll during a full-swing hole never reaches the simulator.

## Launching

The tool is a launcher tile, **Putting Launch Monitor**, in the Biomechanics
category of the Tools launcher (registered in `gui_registration.py`). To open
the window on its own, from the repository root:

```bash
python src/putting_launch_monitor/launch_pyqt6.py
```

There is no `[project.scripts]` console entry: the package imports OpenCV at
module level, so the window is reached through the launcher or the command
above with the `putting-monitor` extra installed.

## Command Line

```bash
# Grab a frame to read the mat corners from
python -m putting_launch_monitor snapshot --camera "USB\VID_...&0000" --out frame.jpg

# Calibrate: corners near-left near-right far-right far-left, mat size in mm
python -m putting_launch_monitor calibrate --camera "USB\VID_...&0000" \
    --corners 693,983 1143,983 1118,518 735,518 --mat-mm 1219 1524

# Confirm GSPro is listening (GSPro itself must be running: it owns port 921)
python -m putting_launch_monitor probe-gspro

# Watch the camera; add --gspro to send putts
python -m putting_launch_monitor run --gspro

# Regression: run the pipeline over a recorded putt
python -m putting_launch_monitor replay --video putt.mkv
```

The calibration lives in the per-user config directory
(`platformdirs.user_config_dir("putting_launch_monitor", "D-sorganization")`)
unless `--out` / `--calibration` say otherwise.

## Cameras

Live cameras come through `shared.python.camera.FfmpegDirectShowSource`: the
bundled ffmpeg reading a Windows camera by its DirectShow device reference.
On the lab's ELP AR0234 cameras this is the only path that streams
1920x1200 at 60 fps; OpenCV's backends do not. A DirectShow camera opens once,
so close anything else holding it first. Frame timestamps derive from the
frame index and the mode's rate — the camera's clock is steadier than pipe
scheduling, and a speed fitted over a dozen frames at 60 fps is sensitive to a
millisecond of jitter.

## Status

Measured on the lab rig (overhead camera, six balls and a putter in view):
the live loop processes 55 fps against the camera's 60 fps clock, sees a ball
in every frame, arms on a resting ball after 11 frames and reports no false
putts. See `AGENT_HANDOFF.md` for what is done, what is measured and what
remains.

## Replay Regression Corpus

Recorded putts run through the pipeline as regression tests via `VideoFileSource`.
The test corpus lives in `src/putting_launch_monitor/tests/data/` accompanied by `manifest.json`.

To run the replay corpus regression tests:

```bash
python -m pytest -m slow src/putting_launch_monitor/tests/test_replay_corpus.py
```

### Adding a Recorded Clip to the Corpus

1. **Record the putt**:
   Capture the overhead camera (e.g. from the UpstreamDrift capture rig at 1920x1200@60):
   ```bash
   rig record --camera cam_b=<id> --duration 5 --out raw_putt.mkv
   ```
2. **Trim and downscale**:
   Trim to ~2 s around the roll and scale down to 960 px decode width using ffmpeg (encoded as MJPEG `.avi` to keep file size well under 5 MB):
   ```bash
   ffmpeg -ss 00:01.000 -to 00:03.000 -i raw_putt.mkv -vf "scale=960:-1" -vcodec mjpeg -q:v 3 src/putting_launch_monitor/tests/data/putt_my_test.avi
   ```
3. **Register in `manifest.json`**:
   Add an entry under `"clips"` in `src/putting_launch_monitor/tests/data/manifest.json`:
   ```json
   {
     "file": "putt_my_test.avi",
     "kind": "putt",
     "expected_speed_mph": 4.5,
     "expected_hla_deg": 1.0,
     "tolerance_speed_mph": 0.25,
     "tolerance_hla_deg": 0.8,
     "description": "My test putt ~4.5 mph, HLA +1.0 deg"
   }
   ```
   For negative test controls (e.g. a hand placing a ball, a ball rolling through unarmed), set `"kind": "no_putt"` and `"expected_putts": 0`.
4. **Verify**:
   ```bash
   python -m putting_launch_monitor replay --video src/putting_launch_monitor/tests/data/putt_my_test.avi --calibration src/putting_launch_monitor/tests/data/calibration.json
   python -m pytest -m slow src/putting_launch_monitor/tests/test_replay_corpus.py
   ```

