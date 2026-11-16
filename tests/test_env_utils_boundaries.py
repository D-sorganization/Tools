"""Isolated environment-file discovery and optional-dependency boundaries."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest
from utils import env_utils, path_helpers, path_setup

pytestmark = pytest.mark.unit


@pytest.fixture
def isolated_search(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Keep every default search candidate inside a temporary tree."""
    working = tmp_path / "working"
    working.mkdir()
    monkeypatch.chdir(working)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.setattr(env_utils, "_REPO_ROOT_FALLBACK", tmp_path / "fallback")
    monkeypatch.setattr(path_setup, "get_repo_root", lambda: tmp_path / "repo")
    monkeypatch.setattr(
        path_helpers, "get_project_root_from_file", lambda _: tmp_path / "helper"
    )
    return tmp_path


def test_discovery_uses_path_helper_when_primary_resolver_unavailable(
    isolated_search: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = isolated_search / "helper" / ".env"
    candidate.parent.mkdir()
    candidate.write_text("TOOLS_ENV_TEST=helper\n", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "utils.path_setup", None)
    assert env_utils.find_env_file() == candidate


def test_discovery_uses_stable_fallback_when_both_resolvers_unavailable(
    isolated_search: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = isolated_search / "fallback" / ".env"
    candidate.parent.mkdir()
    candidate.write_text("TOOLS_ENV_TEST=fallback\n", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "utils.path_setup", None)
    monkeypatch.setitem(sys.modules, "utils.path_helpers", None)
    assert env_utils.find_env_file() == candidate


def test_absent_repo_root_continues_to_user_config(
    isolated_search: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = isolated_search / "home" / ".pdf_renamer" / ".env"
    candidate.parent.mkdir(parents=True)
    candidate.write_text("TOOLS_ENV_TEST=user\n", encoding="utf-8")
    monkeypatch.setattr(path_setup, "get_repo_root", lambda: None)
    assert env_utils.find_env_file() == candidate


def test_directories_are_skipped_and_current_directory_precedes_repo(
    isolated_search: Path,
) -> None:
    directory_candidate = isolated_search / "not-a-file.env"
    directory_candidate.mkdir()
    local_candidate = isolated_search / "working" / ".env"
    local_candidate.write_text("TOOLS_ENV_TEST=local\n", encoding="utf-8")
    repo_candidate = isolated_search / "repo" / ".env"
    repo_candidate.parent.mkdir()
    repo_candidate.write_text("TOOLS_ENV_TEST=repo\n", encoding="utf-8")
    assert (
        env_utils.find_env_file(search_locations=[directory_candidate])
        == local_candidate
    )


def test_repo_root_candidate_is_found_when_working_directory_has_no_file(
    isolated_search: Path,
) -> None:
    candidate = isolated_search / "repo" / ".env"
    candidate.parent.mkdir()
    candidate.write_text("TOOLS_ENV_TEST=repo\n", encoding="utf-8")
    assert env_utils.find_env_file() == candidate


def test_loading_discovered_file_sets_values_without_overriding_existing_values(
    isolated_search: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(os, "environ", os.environ.copy())
    monkeypatch.delenv("TOOLS_ENV_BOUNDARY_NEW", raising=False)
    monkeypatch.setenv("TOOLS_ENV_BOUNDARY_EXISTING", "operator")
    candidate = isolated_search / "working" / ".env"
    candidate.write_text(
        "TOOLS_ENV_BOUNDARY_NEW=loaded\nTOOLS_ENV_BOUNDARY_EXISTING=file\n",
        encoding="utf-8",
    )
    assert env_utils.load_env_file() is True
    assert env_utils.get_env_var("TOOLS_ENV_BOUNDARY_NEW") == "loaded"
    assert env_utils.get_env_var("TOOLS_ENV_BOUNDARY_EXISTING") == "operator"


def test_missing_dotenv_returns_false_with_diagnostic(
    isolated_search: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setitem(sys.modules, "dotenv", None)
    assert env_utils.load_env_file(env_path=isolated_search / "unused.env") is False
    assert "python-dotenv not installed" in caplog.text


@pytest.mark.parametrize("function", [env_utils.find_env_file, env_utils.load_env_file])
def test_missing_filename_is_rejected(function: Any) -> None:
    missing: Any = None
    with pytest.raises(ValueError, match="filename must be provided"):
        function(filename=missing)


@pytest.mark.parametrize("function", [env_utils.get_env_bool, env_utils.get_env_int])
def test_missing_environment_key_is_rejected(function: Any) -> None:
    missing: Any = None
    with pytest.raises(ValueError, match="key must be provided"):
        function(missing)
