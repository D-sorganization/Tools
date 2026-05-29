"""Unit tests for ``tools_sidebar.state_profile_actions``.

``StateProfileMixin`` adds save/load/clear of named sidebar state profiles. The
mixin is Qt-free (it delegates to ``SidekickStateProfileStore``), so a tiny
concrete host implementing ``snapshot_state``/``apply_state`` is enough.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from sidekick.ui.tools_sidebar.state import SidebarState
from sidekick.ui.tools_sidebar.state_profile_actions import StateProfileMixin
from sidekick.ui.tools_sidebar.state_profiles import CLEAR_SIDEKICK_DATA_CONFIRMATION


class _Host(StateProfileMixin):
    def __init__(self) -> None:
        self._state: Any = None
        self._snapshot = SidebarState()
        self.applied: SidebarState | None = None

    def snapshot_state(self) -> SidebarState:
        return self._snapshot

    def apply_state(self, state: SidebarState) -> None:
        self.applied = state


def test_save_profile_persists_and_updates_state(tmp_path: Path) -> None:
    host = _Host()
    result = host.save_state_profile(tmp_path, "my_profile")
    assert result.ok is True
    assert result.path is not None and result.path.exists()
    # On success the mixin caches the snapshot as the live state.
    assert host._state is host._snapshot


def test_load_profile_applies_state(tmp_path: Path) -> None:
    host = _Host()
    host.save_state_profile(tmp_path, "p1")

    fresh = _Host()
    result = fresh.load_state_profile(tmp_path, "p1")
    assert result.ok is True
    assert isinstance(fresh.applied, SidebarState)


def test_load_missing_profile_does_not_apply(tmp_path: Path) -> None:
    host = _Host()
    result = host.load_state_profile(tmp_path, "absent")
    assert result.ok is False
    assert host.applied is None


def test_clear_profiles_requires_confirmation(tmp_path: Path) -> None:
    host = _Host()
    host.save_state_profile(tmp_path, "p1")
    denied = host.clear_state_profiles(tmp_path)
    assert denied.ok is False
    assert (tmp_path / "profiles").exists()


def test_clear_profiles_with_confirmation_removes_data(tmp_path: Path) -> None:
    host = _Host()
    host.save_state_profile(tmp_path, "p1")
    cleared = host.clear_state_profiles(
        tmp_path, confirmation=CLEAR_SIDEKICK_DATA_CONFIRMATION
    )
    assert cleared.ok is True
    assert not tmp_path.exists()


def test_invalid_profile_name_raises(tmp_path: Path) -> None:
    host = _Host()
    with pytest.raises(ValueError, match="path-safe"):
        host.save_state_profile(tmp_path, "../escape")


def test_base_mixin_methods_are_abstract() -> None:
    class _Bare(StateProfileMixin):
        pass

    bare = _Bare()
    with pytest.raises(NotImplementedError):
        bare.snapshot_state()
    with pytest.raises(NotImplementedError):
        bare.apply_state(SidebarState())
