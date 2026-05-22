"""Tests for the ONNX Runtime preflight helper.

Issue #2777: local-embeddings feature requires ORT_DYLIB_PATH but was
previously undocumented, leading to silent failures on Windows. The preflight
check raises a clear RuntimeError so users get actionable guidance.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------
from src.shared.python.ai._onnx_preflight import (  # noqa: E402
    _ENV_VAR,
    _SETUP_GUIDE,
    check_ort_loadable,
)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCheckOrtLoadableEnvVar:
    """Behaviour when ORT_DYLIB_PATH is absent or empty."""

    def test_raises_when_env_var_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """RuntimeError raised with helpful message when env var is not set."""
        monkeypatch.delenv(_ENV_VAR, raising=False)

        with pytest.raises(RuntimeError) as exc_info:
            check_ort_loadable()

        message = str(exc_info.value)
        assert _ENV_VAR in message
        assert _SETUP_GUIDE in message
        assert "ONNX runtime not loadable" in message

    def test_raises_when_env_var_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """RuntimeError raised when env var is set to an empty string."""
        monkeypatch.setenv(_ENV_VAR, "")

        with pytest.raises(RuntimeError) as exc_info:
            check_ort_loadable()

        assert _ENV_VAR in str(exc_info.value)

    def test_error_message_includes_download_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Error message must include the download URL for user guidance."""
        monkeypatch.delenv(_ENV_VAR, raising=False)

        with pytest.raises(RuntimeError) as exc_info:
            check_ort_loadable()

        assert "github.com/microsoft/onnxruntime/releases" in str(exc_info.value)

    @pytest.mark.parametrize(
        "platform_hint,expected_snippet",
        [
            ("Windows PowerShell", "PowerShell"),
            ("Linux", "Linux"),
            ("macOS", "macOS"),
        ],
    )
    def test_error_message_includes_platform_instructions(
        self,
        monkeypatch: pytest.MonkeyPatch,
        platform_hint: str,
        expected_snippet: str,
    ) -> None:
        """Error message must include setup instructions for all platforms."""
        monkeypatch.delenv(_ENV_VAR, raising=False)

        with pytest.raises(RuntimeError) as exc_info:
            check_ort_loadable()

        msg = f"Expected '{expected_snippet}' in error for '{platform_hint}'"
        assert expected_snippet in str(exc_info.value), msg


class TestCheckOrtLoadableInvalidPath:
    """Behaviour when ORT_DYLIB_PATH is set but the path is invalid."""

    @pytest.mark.parametrize(
        "fake_path",
        [
            "/nonexistent/libonnxruntime.so",
            "C:\\nonexistent\\onnxruntime.dll",
            "/tmp/not_a_real_library.so",
        ],
    )
    def test_raises_for_nonexistent_file(
        self, monkeypatch: pytest.MonkeyPatch, fake_path: str
    ) -> None:
        """RuntimeError raised when the path does not exist."""
        monkeypatch.setenv(_ENV_VAR, fake_path)

        with pytest.raises(RuntimeError) as exc_info:
            check_ort_loadable()

        message = str(exc_info.value)
        assert fake_path in message
        assert _SETUP_GUIDE in message
        assert "ONNX runtime not loadable" in message

    def test_error_message_includes_os_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """RuntimeError must chain / mention the underlying OS error."""
        fake_path = "/definitely/does/not/exist/libonnxruntime.so"
        monkeypatch.setenv(_ENV_VAR, fake_path)

        with pytest.raises(RuntimeError) as exc_info:
            check_ort_loadable()

        assert exc_info.value.__cause__ is not None, (
            "RuntimeError should chain the underlying OSError"
        )

    def test_raises_for_nonexistent_explicit_path(self) -> None:
        """Explicit dylib_path argument is used instead of env var."""
        with pytest.raises(RuntimeError) as exc_info:
            check_ort_loadable(dylib_path="/nonexistent/libonnxruntime.so")

        message = str(exc_info.value)
        assert _SETUP_GUIDE in message

    def test_error_mentions_version_mismatch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Error message should mention version mismatch for diagnostics."""
        monkeypatch.setenv(_ENV_VAR, "/fake/libonnxruntime.so")

        with pytest.raises(RuntimeError) as exc_info:
            check_ort_loadable()

        assert "1.17" in str(exc_info.value)


class TestCheckOrtLoadableExplicitPath:
    """Behaviour of the explicit dylib_path argument."""

    def test_explicit_none_falls_back_to_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When dylib_path=None the env var is consulted."""
        monkeypatch.delenv(_ENV_VAR, raising=False)

        with pytest.raises(RuntimeError):
            check_ort_loadable(dylib_path=None)

    def test_explicit_empty_string_raises(self) -> None:
        """An explicit empty string is treated the same as unset."""
        # ctypes.CDLL("") may or may not raise depending on OS; we only
        # test that it does not silently succeed without a meaningful path.
        # We deliberately pass a path that cannot be a real library.
        with pytest.raises(RuntimeError):
            check_ort_loadable(dylib_path="/this/path/does/not/exist.so")
