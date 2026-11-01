"""Guard the mypy.ini override for PyQt6's broken bundled stubs (issue #5907).

PyQt6 6.12.0 ships ``QtGui.pyi`` with ``*args, /`` (``/`` after ``*``), which
mypy rejects as a syntax error. The error stops the whole run, so the
changed-files mypy step fails on any module that imports ``PyQt6.QtGui``.
"""

from __future__ import annotations

import configparser
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_MYPY_INI = Path(__file__).resolve().parents[1] / "mypy.ini"
_PYQT6_SECTION = "mypy-PyQt6.*"


def _pyqt6_section() -> configparser.SectionProxy:
    parser = configparser.ConfigParser()
    parser.read(_MYPY_INI, encoding="utf-8")
    assert parser.has_section(_PYQT6_SECTION), f"{_MYPY_INI} lacks [{_PYQT6_SECTION}]"
    return parser[_PYQT6_SECTION]


def test_pyqt6_imports_are_skipped() -> None:
    assert _pyqt6_section().get("follow_imports") == "skip"


def test_skip_also_applies_to_pyqt6_stubs() -> None:
    # Without this, mypy still parses the .pyi files and hits the syntax error.
    assert _pyqt6_section().getboolean("follow_imports_for_stubs") is True
