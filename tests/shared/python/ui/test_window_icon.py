"""Tests for the shared window-icon / taskbar-identity helpers.

These guard the favicon regression that recurred because earlier fixes only
adjusted the icon file path and asserted ``windowIcon() is not None`` — which
stays true even when the Windows *taskbar* icon is wrong. The critical
assertion here is that the AppUserModelID is actually set (the missing piece),
plus that the icon is applied to BOTH the application and the window.

DbC: each test documents its precondition/postcondition.
LOD: tests use only the public ``shared.python.ui`` surface.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

from shared.python.ui import (
    apply_window_icon,
    resolve_icon_path,
    set_app_user_model_id,
)

pytestmark = pytest.mark.unit


class _FakeIconTarget:
    """Records ``setWindowIcon`` calls without needing a real QWidget."""

    def __init__(self) -> None:
        self.icons: list[Any] = []

    def setWindowIcon(self, icon: Any) -> None:  # noqa: N802 - Qt API name
        self.icons.append(icon)


# ─── resolve_icon_path ───────────────────────────────────────────


def test_resolve_icon_path_returns_first_existing(tmp_path: Path) -> None:
    """Precondition: second candidate exists, first does not.
    Postcondition: the existing path is returned."""
    missing = tmp_path / "missing.ico"
    present = tmp_path / "present.png"
    present.write_bytes(b"icon")
    assert resolve_icon_path([missing, present]) == present


def test_resolve_icon_path_prefers_earlier_candidate(tmp_path: Path) -> None:
    """Precondition: both candidates exist.
    Postcondition: the earliest candidate wins (so .ico can precede .png)."""
    first = tmp_path / "a.ico"
    second = tmp_path / "b.png"
    first.write_bytes(b"ico")
    second.write_bytes(b"png")
    assert resolve_icon_path([first, second]) == first


def test_resolve_icon_path_none_when_no_candidate_exists(tmp_path: Path) -> None:
    """Precondition: no candidate exists.
    Postcondition: returns None rather than raising."""
    assert resolve_icon_path([tmp_path / "nope.ico"]) is None


def test_resolve_icon_path_rejects_none() -> None:
    """Precondition: candidates is None.
    Postcondition: TypeError is raised (DbC)."""
    with pytest.raises(TypeError):
        resolve_icon_path(None)  # type: ignore[arg-type]


# ─── set_app_user_model_id ───────────────────────────────────────


def test_set_app_user_model_id_rejects_non_string() -> None:
    """Precondition: app_id is not a string.
    Postcondition: TypeError is raised."""
    with pytest.raises(TypeError):
        set_app_user_model_id(123)  # type: ignore[arg-type]


def test_set_app_user_model_id_rejects_empty() -> None:
    """Precondition: app_id is whitespace-only.
    Postcondition: ValueError is raised."""
    with pytest.raises(ValueError, match="non-empty"):
        set_app_user_model_id("   ")


def test_set_app_user_model_id_noop_off_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Precondition: platform is not win32.
    Postcondition: returns False without raising."""
    monkeypatch.setattr(sys, "platform", "linux")
    assert set_app_user_model_id("D-sorganization.Test") is False


def test_set_app_user_model_id_calls_windows_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Precondition: platform is win32 (forced) with a stub shell32.
    Postcondition: SetCurrentProcessExplicitAppUserModelID is called with the
    exact app_id, and the function returns True.

    This is the assertion the previous favicon 'fix' lacked — verifying the
    taskbar identity is actually declared, not merely that an icon loaded.
    """
    monkeypatch.setattr(sys, "platform", "win32")
    recorded: list[str] = []

    fake_shell32 = types.SimpleNamespace(
        SetCurrentProcessExplicitAppUserModelID=lambda app_id: recorded.append(app_id)
    )
    fake_ctypes = types.SimpleNamespace(
        windll=types.SimpleNamespace(shell32=fake_shell32)
    )
    monkeypatch.setitem(sys.modules, "ctypes", fake_ctypes)

    result = set_app_user_model_id("D-sorganization.UpstreamDrift")

    assert result is True
    assert recorded == ["D-sorganization.UpstreamDrift"]


# ─── apply_window_icon ───────────────────────────────────────────


def test_apply_window_icon_sets_app_and_window(tmp_path: Path) -> None:
    """Precondition: a valid icon file exists.
    Postcondition: the icon is set on BOTH the app and the window; the
    applied path is returned."""
    icon_file = tmp_path / "app.ico"
    icon_file.write_bytes(b"ico")
    app = _FakeIconTarget()
    window = _FakeIconTarget()

    result = apply_window_icon(
        app=app,
        window=window,
        icon_candidates=[icon_file],
        icon_factory=lambda path: ("ICON", path),
    )

    assert result == icon_file
    assert app.icons == [("ICON", str(icon_file))]
    assert window.icons == [("ICON", str(icon_file))]


def test_apply_window_icon_declares_app_user_model_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Precondition: app_id supplied.
    Postcondition: set_app_user_model_id is invoked with that id before the
    window is shown (taskbar fix)."""
    icon_file = tmp_path / "app.ico"
    icon_file.write_bytes(b"ico")
    called: list[str] = []
    monkeypatch.setattr(
        "shared.python.ui.window_icon.set_app_user_model_id",
        lambda app_id: called.append(app_id) or True,
    )

    apply_window_icon(
        app=_FakeIconTarget(),
        window=_FakeIconTarget(),
        icon_candidates=[icon_file],
        app_id="D-sorganization.MyApp",
        icon_factory=lambda path: path,
    )

    assert called == ["D-sorganization.MyApp"]


def test_apply_window_icon_missing_icon_returns_none(tmp_path: Path) -> None:
    """Precondition: no candidate exists.
    Postcondition: returns None and sets no icon."""
    window = _FakeIconTarget()
    result = apply_window_icon(
        app=_FakeIconTarget(),
        window=window,
        icon_candidates=[tmp_path / "nope.ico"],
        icon_factory=lambda path: path,
    )
    assert result is None
    assert window.icons == []


def test_apply_window_icon_allows_app_none(tmp_path: Path) -> None:
    """Precondition: app is None (e.g. icon set on a child window only).
    Postcondition: only the window icon is set; no crash."""
    icon_file = tmp_path / "app.ico"
    icon_file.write_bytes(b"ico")
    window = _FakeIconTarget()
    result = apply_window_icon(
        app=None,
        window=window,
        icon_candidates=[icon_file],
        icon_factory=lambda path: path,
    )
    assert result == icon_file
    assert window.icons == [str(icon_file)]


def test_apply_window_icon_requires_window() -> None:
    """Precondition: window is None.
    Postcondition: TypeError is raised (DbC)."""
    with pytest.raises(TypeError):
        apply_window_icon(
            app=None,
            window=None,
            icon_candidates=[],
        )
