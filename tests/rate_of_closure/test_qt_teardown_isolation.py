"""Check native Qt ownership across actual pytest teardown boundaries."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]
_ROOT = Path(__file__).resolve().parents[2]
_PROBE = """
from PyQt6 import sip
from PyQt6.QtCore import QCoreApplication, QTimer
from PyQt6.QtWidgets import QWidget

retained = []

def test_no_application_is_created_for_non_widget_cases():
    assert QCoreApplication.instance() is None

def test_previous_teardown_did_not_create_an_application():
    assert QCoreApplication.instance() is None

def test_register_widget_and_owned_timer(qtbot):
    widget = QWidget()
    timer = QTimer(widget)
    retained.extend((widget, timer))
    qtbot.addWidget(widget)

def test_native_objects_are_deleted_before_the_next_case():
    assert len(retained) == 2
    assert all(sip.isdeleted(obj) for obj in retained)
"""
_ERROR_PROBE = """
from PyQt6.QtWidgets import QWidget

retained = []

def test_deletion_callback_failure(qtbot):
    widget = QWidget()
    retained.append(widget)
    def fail_on_destroyed(_object):
        raise RuntimeError("intentional deletion failure")
    widget.destroyed.connect(fail_on_destroyed)
    qtbot.addWidget(widget)
"""


def _run_probe(tmp_path: Path, probe: str) -> subprocess.CompletedProcess[str]:
    """Run the repository's actual fixture module in a fresh tiny Qt suite."""
    shutil.copyfile(
        _ROOT / "tests/rate_of_closure/conftest.py",
        tmp_path / "rate_fixture_plugin.py",
    )
    (tmp_path / "test_probe.py").write_text(probe, encoding="utf-8")
    (tmp_path / "pytest.ini").write_text("[pytest]\nqt_api = pyqt6\n", encoding="utf-8")
    environment = {
        **os.environ,
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTEST_ADDOPTS": "",
        "PYTEST_QT_API": "pyqt6",
        "QT_QPA_PLATFORM": "offscreen",
        "PYTHONPATH": os.pathsep.join(
            str(path) for path in (tmp_path, _ROOT / "src", _ROOT / "src/shared/python")
        ),
    }
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "pytestqt.plugin",
            "-p",
            "rate_fixture_plugin",
            "test_probe.py",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )


def test_rate_teardown_releases_native_widgets_between_cases(tmp_path: Path) -> None:
    """Retain wrappers deliberately: Qt deletion must not depend on Python GC."""
    result = _run_probe(tmp_path, _PROBE)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "4 passed" in result.stdout


def test_deferred_deletion_keeps_qt_exception_capture_active(tmp_path: Path) -> None:
    result = _run_probe(tmp_path, _ERROR_PROBE)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "intentional deletion failure" in result.stdout + result.stderr
    assert "1 passed, 1 error" in result.stdout
