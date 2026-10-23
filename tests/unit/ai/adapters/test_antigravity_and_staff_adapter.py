"""Unit tests for Antigravity (`agy`) and RunnerDashboardStaffAdapter (#5729)."""

from __future__ import annotations

import json
import subprocess
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from shared.python.ai.adapters.antigravity_cli_adapter import (
    DEFAULT_AGY_MODEL,
    SUBSCRIPTION_CLI_ENABLED_BY_DEFAULT,
    AntigravityCliAdapter,
    RunnerDashboardStaffAdapter,
    _resolve_binary,
    normalize_cli_provider_id,
    probe_runner_dashboard_staff,
)
from shared.python.ai.adapters.cli_provider_setup import get_cli_setup_status
from shared.python.ai.adapters.factory import AdapterFactory
from shared.python.ai.exceptions import (
    AIConnectionError,
    AIProviderError,
    AITimeoutError,
)
from shared.python.ai.types import ConversationContext

_RUN = "shared.python.ai.adapters.gemini_cli_adapter.subprocess.run"
_URLOPEN = "shared.python.ai.adapters.antigravity_cli_adapter.urllib.request.urlopen"


def _completed(rc: int = 0, stdout: str = "", stderr: str = "") -> MagicMock:
    return MagicMock(returncode=rc, stdout=stdout, stderr=stderr)


def test_normalize_and_legacy_deprecation_shim() -> None:
    assert SUBSCRIPTION_CLI_ENABLED_BY_DEFAULT is False
    assert AntigravityCliAdapter.ENABLED_BY_DEFAULT is False
    assert normalize_cli_provider_id("AGY") == "agy"
    with pytest.deprecated_call(match="gemini_cli"):
        assert normalize_cli_provider_id("gemini_cli") == "agy"
    with pytest.raises(ValueError, match="non-empty"):
        normalize_cli_provider_id("  ")
    with pytest.deprecated_call(match="gemini_cli"):
        migrated = AntigravityCliAdapter.from_legacy_config({"provider": "gemini_cli"})
    assert migrated.model == DEFAULT_AGY_MODEL
    with pytest.raises(ValueError, match="mapping"):
        AntigravityCliAdapter.from_legacy_config("bad")  # type: ignore[arg-type]


def test_antigravity_cli_adapter_send_and_errors(tmp_path: Path) -> None:
    fake_bin = tmp_path / "agy"
    fake_bin.write_text("")
    assert _resolve_binary(str(fake_bin)) == str(fake_bin)
    adapter = AntigravityCliAdapter(binary=str(fake_bin))
    assert adapter.model == "gemini-3.8-flash-high"
    assert DEFAULT_AGY_MODEL in adapter.list_models()

    with patch(_RUN, return_value=_completed(0, stdout="agy reply\n")) as run_mock:
        resp = adapter.send_message("hello agy", ConversationContext(), [])
    assert resp.content == "agy reply"
    assert resp.metadata["provider"] == "agy"
    assert "--skip-trust" in run_mock.call_args.args[0]

    with pytest.raises(ValueError, match="non-empty"):
        adapter.send_message("   ", ConversationContext(), [])
    with (
        patch(_RUN, side_effect=subprocess.TimeoutExpired(cmd="agy", timeout=5.0)),
        pytest.raises(AITimeoutError),
    ):
        adapter.send_message("slow", ConversationContext(), [])
    with (
        patch(_RUN, return_value=_completed(2, stderr="auth expired")),
        pytest.raises(AIProviderError, match="auth expired"),
    ):
        adapter.send_message("fail", ConversationContext(), [])
    with (
        patch(
            "shared.python.ai.adapters.antigravity_cli_adapter.shutil.which",
            return_value=None,
        ),
        patch(
            "shared.python.ai.adapters.antigravity_cli_adapter.Path.exists",
            return_value=False,
        ),
    ):
        missing = AntigravityCliAdapter(binary=None)
    assert missing.validate_connection()[0] is False
    with pytest.raises(AIConnectionError):
        missing.send_message("hi", ConversationContext(), [])


def test_runner_dashboard_staff_adapter_and_factory() -> None:
    AdapterFactory.clear_cache()
    assert isinstance(
        AdapterFactory.create("agy", host="/usr/bin/agy"), AntigravityCliAdapter
    )
    staff = AdapterFactory.create(
        "runner_dashboard_staff", host="http://127.0.0.1:8321"
    )
    assert isinstance(staff, RunnerDashboardStaffAdapter)
    assert staff.ENABLED_BY_DEFAULT is False
    assert "barb" in staff.list_models()

    fake_resp = MagicMock(status=200)
    fake_resp.__enter__.return_value = fake_resp
    fake_resp.read.return_value = json.dumps({"reply": "Barb ready"}).encode("utf-8")
    with patch(_URLOPEN, return_value=fake_resp):
        assert staff.validate_connection()[0] is True
        chunks = list(staff.stream_response("check fleet", ConversationContext(), []))
    assert chunks[0].content == "Barb ready"

    with (
        patch(_URLOPEN, side_effect=urllib.error.URLError("refused")),
        pytest.raises(AIConnectionError),
    ):
        staff.send_message("offline", ConversationContext(), [])
    with pytest.raises(ValueError, match="positive"):
        probe_runner_dashboard_staff("http://127.0.0.1:8321", timeout=0)

    assert get_cli_setup_status("agy").provider == "agy"
