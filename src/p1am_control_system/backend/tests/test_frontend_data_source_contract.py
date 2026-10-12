"""The HMI's ``DATA_SOURCES`` must mirror ``models.DataSource`` (issue #5525).

The frontend classifies a scan by the ``data_source`` string the backend puts
in every frame. If the enums drift, a new backend state would render as
"unknown" (fail-safe) -- this test makes the drift loud instead of silent.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

pytest.importorskip("sqlmodel")

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import DataSource  # noqa: E402

_DATA_AGE_TS = Path(__file__).parents[2] / "frontend" / "src" / "lib" / "dataAge.ts"


def _frontend_sources() -> set[str]:
    text = _DATA_AGE_TS.read_text(encoding="utf-8")
    match = re.search(r"DATA_SOURCES\s*=\s*\[([^\]]*)\]\s*as const", text)
    assert match, "DATA_SOURCES tuple not found in dataAge.ts"
    return set(re.findall(r'"(\w+)"', match.group(1)))


def test_frontend_data_sources_match_backend_enum() -> None:
    assert _frontend_sources() == {member.value for member in DataSource}
