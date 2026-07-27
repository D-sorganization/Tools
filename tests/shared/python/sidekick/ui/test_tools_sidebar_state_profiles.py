"""Tests for SidekickStateProfileStore — named state persistence.

DbC: Each test states preconditions and postconditions.
LOD: Tests use the public SidekickStateProfileStore and validate_profile_name API.
TDD: Tests were written before filling in this stub to drive the coverage need.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


class TestValidateProfileName:
    """validate_profile_name() rejects unsafe names and accepts safe ones."""

    @pytest.mark.parametrize(
        "good_name",
        [
            "default",
            "My Profile",
            "profile-1",
            "profile.1.0",
            "A",
            "abc 123",
        ],
    )
    def test_valid_names_accepted(self, good_name: str) -> None:
        """Precondition: name is alphanumeric/safe.
        Postcondition: validate_profile_name returns the stripped name."""
        from sidekick.ui.tools_sidebar.state_profiles import validate_profile_name

        result = validate_profile_name(good_name)
        assert result == good_name.strip()

    @pytest.mark.parametrize(
        "bad_name",
        [
            "",
            "  ",
            ".",
            "..",
            "../../etc",
            "foo/bar",
            "foo\\bar",
            "\x00null",
        ],
    )
    def test_unsafe_names_rejected(self, bad_name: str) -> None:
        """Precondition: name is empty, '.', '..', or contains path separators.
        Postcondition: ValueError raised."""
        from sidekick.ui.tools_sidebar.state_profiles import validate_profile_name

        with pytest.raises(ValueError):
            validate_profile_name(bad_name)


class TestSidekickStateProfileStoreSaveLoad:
    """Save/load round-trip through the filesystem."""

    def test_save_creates_profile_file(self, tmp_path: Path) -> None:
        """Precondition: store is created with a temp root.
        Postcondition: save_profile() creates a .json file."""
        from sidekick.ui.tools_sidebar.state import SidebarState
        from sidekick.ui.tools_sidebar.state_profiles import (
            SidekickStateProfileStore,
        )

        store = SidekickStateProfileStore(tmp_path / "data")
        state = SidebarState()
        result = store.save_profile("test-save", state)

        assert result.ok is True
        assert result.path is not None
        assert result.path.exists()
        assert result.path.suffix == ".json"

    def test_save_returns_correct_profile_name(self, tmp_path: Path) -> None:
        """Precondition: valid profile name is provided.
        Postcondition: result.profile_name matches the provided name."""
        from sidekick.ui.tools_sidebar.state import SidebarState
        from sidekick.ui.tools_sidebar.state_profiles import (
            SidekickStateProfileStore,
        )

        store = SidekickStateProfileStore(tmp_path / "data")
        result = store.save_profile("my-profile", SidebarState())

        assert result.profile_name == "my-profile"

    def test_load_after_save_returns_ok(self, tmp_path: Path) -> None:
        """Precondition: profile was saved.
        Postcondition: load_profile() returns ok=True and state is not None."""
        from sidekick.ui.tools_sidebar.state import SidebarState
        from sidekick.ui.tools_sidebar.state_profiles import (
            SidekickStateProfileStore,
        )

        store = SidekickStateProfileStore(tmp_path / "data")
        store.save_profile("load-test", SidebarState())
        result = store.load_profile("load-test")

        assert result.ok is True
        assert result.state is not None

    def test_embedded_and_standalone_stores_share_profile_format(
        self,
        tmp_path: Path,
    ) -> None:
        """Profiles round-trip directly through both public store APIs."""
        from sidekick.standalone.session_store import StandaloneSessionStore
        from sidekick.ui.tools_sidebar.state import SidebarState
        from sidekick.ui.tools_sidebar.state_profiles import (
            SidekickStateProfileStore,
        )

        storage_root = tmp_path / "shared"
        embedded = SidekickStateProfileStore(storage_root)
        standalone = StandaloneSessionStore(storage_root)
        expected = SidebarState(dock_area="left", active_tab="calculator")

        embedded.save_profile("shared_profile", expected)
        standalone_payload = standalone.load_profile("shared_profile")
        assert standalone_payload.data == expected.to_dict()

        standalone.save_profile("shared_profile", standalone_payload)
        embedded_result = embedded.load_profile("shared_profile")
        assert embedded_result.ok is True
        assert embedded_result.state == expected

    def test_saved_profile_declares_current_schema_version(
        self,
        tmp_path: Path,
    ) -> None:
        """Embedded persistence writes the canonical versioned artifact."""
        from sidekick.persistence import (
            PROFILE_SCHEMA_VERSION,
            PROFILE_SCHEMA_VERSION_KEY,
        )
        from sidekick.ui.tools_sidebar.state import SidebarState
        from sidekick.ui.tools_sidebar.state_profiles import (
            SidekickStateProfileStore,
        )

        store = SidekickStateProfileStore(tmp_path / "data")
        result = store.save_profile("versioned", SidebarState())
        assert result.path is not None

        payload = json.loads(result.path.read_text(encoding="utf-8"))
        assert payload[PROFILE_SCHEMA_VERSION_KEY] == PROFILE_SCHEMA_VERSION

    def test_legacy_profile_is_migrated_on_load(
        self,
        tmp_path: Path,
    ) -> None:
        """A pre-schema embedded profile remains loadable with a clear warning."""
        from sidekick.persistence import SchemaMigration
        from sidekick.ui.tools_sidebar.state import SidebarState
        from sidekick.ui.tools_sidebar.state_profiles import (
            SidekickStateProfileStore,
        )

        store = SidekickStateProfileStore(tmp_path / "data")
        legacy_path = store.profiles_dir / "legacy.json"
        legacy_path.parent.mkdir(parents=True)
        expected = SidebarState(dock_area="left")
        legacy_path.write_text(json.dumps(expected.to_dict()), encoding="utf-8")

        with pytest.warns(SchemaMigration):
            result = store.load_profile("legacy")

        assert result.ok is True
        assert result.state == expected

    def test_load_missing_profile_returns_not_ok(self, tmp_path: Path) -> None:
        """Precondition: profile file does not exist.
        Postcondition: load_profile() returns ok=False with 'not found' in message."""
        from sidekick.ui.tools_sidebar.state_profiles import (
            SidekickStateProfileStore,
        )

        store = SidekickStateProfileStore(tmp_path / "data")
        result = store.load_profile("nonexistent")

        assert result.ok is False
        assert "not found" in result.message.lower()

    def test_save_profile_message_is_saved(self, tmp_path: Path) -> None:
        """Precondition: save_profile called successfully.
        Postcondition: result.message == 'saved'."""
        from sidekick.ui.tools_sidebar.state import SidebarState
        from sidekick.ui.tools_sidebar.state_profiles import (
            SidekickStateProfileStore,
        )

        store = SidekickStateProfileStore(tmp_path / "data")
        result = store.save_profile("msg-test", SidebarState())

        assert result.message == "saved"

    def test_load_profile_message_is_loaded(self, tmp_path: Path) -> None:
        """Precondition: profile was saved.
        Postcondition: load_profile() result.message == 'loaded'."""
        from sidekick.ui.tools_sidebar.state import SidebarState
        from sidekick.ui.tools_sidebar.state_profiles import (
            SidekickStateProfileStore,
        )

        store = SidekickStateProfileStore(tmp_path / "data")
        store.save_profile("load-msg-test", SidebarState())
        result = store.load_profile("load-msg-test")

        assert result.message == "loaded"

    def test_save_with_invalid_profile_name_raises(self, tmp_path: Path) -> None:
        """Precondition: profile name contains path traversal characters.
        Postcondition: ValueError raised by validate_profile_name."""
        from sidekick.ui.tools_sidebar.state import SidebarState
        from sidekick.ui.tools_sidebar.state_profiles import (
            SidekickStateProfileStore,
        )

        store = SidekickStateProfileStore(tmp_path / "data")
        with pytest.raises(ValueError):
            store.save_profile("../../evil", SidebarState())


class TestSidekickStateProfileStoreClear:
    """clear_data() requires confirmation and protects against accidental deletion."""

    def test_clear_without_confirmation_returns_not_ok(self, tmp_path: Path) -> None:
        """Precondition: no confirmation provided.
        Postcondition: ok=False, warning is non-None."""
        from sidekick.ui.tools_sidebar.state_profiles import (
            SidekickStateProfileStore,
        )

        store = SidekickStateProfileStore(tmp_path / "data")
        result = store.clear_data()

        assert result.ok is False
        assert result.warning is not None

    def test_clear_with_wrong_confirmation_returns_not_ok(self, tmp_path: Path) -> None:
        """Precondition: wrong confirmation string provided.
        Postcondition: ok=False."""
        from sidekick.ui.tools_sidebar.state_profiles import (
            SidekickStateProfileStore,
        )

        store = SidekickStateProfileStore(tmp_path / "data")
        result = store.clear_data(confirmation="wrong-string")

        assert result.ok is False

    def test_clear_with_correct_confirmation_returns_ok(self, tmp_path: Path) -> None:
        """Precondition: correct confirmation token provided.
        Postcondition: ok=True, message == 'cleared'."""
        from sidekick.ui.tools_sidebar.state_profiles import (
            CLEAR_SIDEKICK_DATA_CONFIRMATION,
            SidekickStateProfileStore,
        )

        store = SidekickStateProfileStore(tmp_path / "data")
        result = store.clear_data(confirmation=CLEAR_SIDEKICK_DATA_CONFIRMATION)

        assert result.ok is True
        assert result.message == "cleared"

    def test_clear_removes_profile_files(self, tmp_path: Path) -> None:
        """Precondition: a profile was saved to storage root.
        Postcondition: after clear_data() with confirmation, storage root is gone."""
        from sidekick.ui.tools_sidebar.state import SidebarState
        from sidekick.ui.tools_sidebar.state_profiles import (
            CLEAR_SIDEKICK_DATA_CONFIRMATION,
            SidekickStateProfileStore,
        )

        storage_root = tmp_path / "sidekick_data"
        store = SidekickStateProfileStore(storage_root)
        store.save_profile("to-delete", SidebarState())

        assert storage_root.exists()

        store.clear_data(confirmation=CLEAR_SIDEKICK_DATA_CONFIRMATION)

        assert not storage_root.exists()

    def test_clear_data_warning_constant_is_non_empty(self) -> None:
        """Precondition: CLEAR_SIDEKICK_DATA_WARNING is imported.
        Postcondition: it is a non-empty string."""
        from sidekick.ui.tools_sidebar.state_profiles import (
            CLEAR_SIDEKICK_DATA_WARNING,
        )

        assert isinstance(CLEAR_SIDEKICK_DATA_WARNING, str)
        assert len(CLEAR_SIDEKICK_DATA_WARNING) > 0


class TestSidekickStateProfileResultDataclass:
    """SidekickStateProfileResult is a frozen dataclass with expected fields."""

    def test_result_ok_true(self) -> None:
        """Precondition: SidekickStateProfileResult constructed with ok=True.
        Postcondition: ok, message accessible, warning defaults to None."""
        from sidekick.ui.tools_sidebar.state_profiles import (
            SidekickStateProfileResult,
        )

        result = SidekickStateProfileResult(ok=True, message="saved")

        assert result.ok is True
        assert result.message == "saved"
        assert result.warning is None

    def test_result_ok_false_with_warning(self) -> None:
        """Precondition: result created with ok=False and a warning.
        Postcondition: warning accessible."""
        from sidekick.ui.tools_sidebar.state_profiles import (
            SidekickStateProfileResult,
        )

        result = SidekickStateProfileResult(
            ok=False, message="failed", warning="Data will be lost"
        )

        assert result.ok is False
        assert result.warning == "Data will be lost"
