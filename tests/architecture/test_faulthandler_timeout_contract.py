"""``faulthandler_timeout`` must never fire inside a legitimately long test (#5440).

pytest's faulthandler watchdog dumps every thread's stack once a test runs
longer than ``faulthandler_timeout``. It does so from a C thread, without the
GIL, so a dump taken while the main thread is busy in native code is not safe.
In CI the ``tests-shared`` shard crashed (exit 249) seconds after such a dump,
during a numerical test marked ``timeout(180)``. So the watchdog may be set
only at or above the longest per-test ``timeout`` marker, or not at all.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
_TIMEOUT_MARK = re.compile(r"mark\.timeout\(\s*(\d+)")


def _longest_timeout_marker() -> int:
    longest = 0
    for path in (REPO_ROOT / "tests").rglob("*.py"):
        for match in _TIMEOUT_MARK.finditer(path.read_text(encoding="utf-8")):
            longest = max(longest, int(match.group(1)))
    return longest


def test_faulthandler_watchdog_never_interrupts_a_marked_long_test() -> None:
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text("utf-8"))
    options = config["tool"]["pytest"]["ini_options"]
    watchdog = options.get("faulthandler_timeout")
    if watchdog is None:
        return
    longest = _longest_timeout_marker()
    assert longest > 0, "expected at least one @pytest.mark.timeout in tests/"
    assert float(watchdog) >= longest, (
        f"faulthandler_timeout={watchdog} fires inside tests marked "
        f"timeout({longest}); dumping stacks mid-test crashed CI (#5440)"
    )
