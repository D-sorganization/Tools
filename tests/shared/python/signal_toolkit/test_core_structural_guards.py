"""Signal shape guards remain active in optimized Python."""

from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.unit
@pytest.mark.parametrize(
    "construction",
    [
        "Signal(time=np.zeros((2, 2)), values=np.zeros(4))",
        "Signal(time=np.zeros(3), values=np.zeros(4))",
        "Signal(time=np.zeros(3), values=np.zeros((4, 2)))",
        "Signal(time=np.zeros(3), values=np.zeros((3, 2, 2)))",
    ],
)
def test_invalid_signal_shape_rejected_with_optimized_python(
    construction: str,
) -> None:
    """Public shape invariants must not depend on optional DbC enforcement."""
    program = (
        "import numpy as np\n"
        "from shared.python.signal_toolkit.core import Signal\n"
        "try:\n"
        f"    {construction}\n"
        "except ValueError:\n"
        "    print('REJECTED')\n"
        "else:\n"
        "    print('ACCEPTED')\n"
    )
    result = subprocess.run(
        [sys.executable, "-O", "-c", program],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "REJECTED"
