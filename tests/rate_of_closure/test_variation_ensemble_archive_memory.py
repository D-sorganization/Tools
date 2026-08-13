"""Subprocess RSS budget for the non-materializing result archive sink."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.performance, pytest.mark.headless_safe]

_ROOT = Path(__file__).parents[2]
_SCRIPT = _ROOT / "scripts" / "benchmark_rate_ensemble_archive.py"


def _measure(chunks: int) -> dict[str, float]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(_ROOT / "src")
    completed = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--chunks",
            str(chunks),
            "--rows",
            "2",
            "--samples",
            "128",
        ],
        cwd=_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return json.loads(completed.stdout)


def test_peak_result_sink_rss_does_not_scale_with_archive_chunk_count() -> None:
    small = _measure(16)
    large = _measure(128)

    assert small["peak_delta_bytes"] < 64 * 1024 * 1024
    assert large["peak_delta_bytes"] < 64 * 1024 * 1024
    assert large["peak_delta_bytes"] <= small["peak_delta_bytes"] + 16 * 1024 * 1024
