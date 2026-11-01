---
issue: 5896
summary: "test_cmd_validate_execution patches the camera on the sys.modules entry cmd_validate imports, so a stale shared.python.camera package attribute left by other suite tests no longer lets the real ffmpeg source run (nightly red)"
---
