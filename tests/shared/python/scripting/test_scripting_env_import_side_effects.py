"""Regression tests for #3701.

``scripting_env`` previously called ``resource.setrlimit`` at module import
time, capping the *entire* host process to 10s cumulative CPU and 512 MiB.
Importing the module must now be free of process-global side effects; the
limits are only applied via the explicit ``apply_process_resource_limits``
opt-in (or the ``SCRIPTING_ENV_APPLY_RLIMITS`` env flag).
"""

from __future__ import annotations

import subprocess
import sys

import pytest

_PROBE = (
    "import resource;"
    "before=(resource.getrlimit(resource.RLIMIT_CPU),"
    "resource.getrlimit(resource.RLIMIT_AS));"
    "import shared.python.scripting.scripting_env as se;"
    "after=(resource.getrlimit(resource.RLIMIT_CPU),"
    "resource.getrlimit(resource.RLIMIT_AS));"
    "assert before==after, (before, after);"
    "print('OK')"
)


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="resource/setrlimit is POSIX-only; no import-time rlimit on Windows",
)
def test_import_does_not_mutate_process_rlimits() -> None:
    """Importing scripting_env leaves the process RLIMIT_CPU/AS unchanged."""
    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_apply_process_resource_limits_is_explicit() -> None:
    """The opt-in helper exists and is callable.

    We do not invoke it on POSIX here: it would mutate this (the pytest)
    process's rlimits. The subprocess test above proves it is NOT applied at
    import; this test only pins the public opt-in surface.
    """
    from shared.python.scripting import scripting_env as se

    assert callable(se.apply_process_resource_limits)
    if sys.platform == "win32":
        # No-op on Windows: returns False without mutating anything.
        assert se.apply_process_resource_limits() is False
