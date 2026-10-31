"""Real-CAD tests must skip on the tests/vessel_drafter build123d stub (#5874)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_real_cad_tests_skip_when_build123d_is_only_the_stub() -> None:
    # The stub conftest is collected first, as every xdist worker does in the nightly.
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:randomly",
            "-p",
            "no:cacheprovider",
            "-o",
            "addopts=",
            "tests/vessel_drafter/test_dbc_lod.py",
            "src/vessel_drafter/tests/test_cylindrical_bath_layout.py",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
