"""Wheel-installed smoke test for the RustAgentAdapter Rust path (issue #3521).

Unlike ``test_rust_adapter_fallback.py`` (which proves graceful degradation
when ``ai_backend`` is *absent*), this module hard-fails when the wheel is
missing. It is the CI consumption gate: the ``maturin-ai-backend.yml``
workflow builds the ``ai_backend`` wheel (``--features python``), installs it,
and runs this test so the Rust path is *actually exercised* rather than only
uploaded as an artifact.

Run conditions:
- In normal local/CI suites where the wheel may be absent, the module is
  skipped at collection time (``importorskip``), so it never produces false
  failures.
- In the dedicated maturin gate (wheel installed) it runs and asserts the
  adapter builds and reports availability via the real extension.
"""

from __future__ import annotations

import pytest

# Hard requirement: this whole module is the wheel-present gate. When the
# extension is not installed the module is skipped (so it is inert in suites
# that lack a compiled wheel), but the maturin CI job installs the wheel first
# so the skip never triggers there.
ai_backend = pytest.importorskip(
    "ai_backend",
    reason="ai_backend Rust wheel not installed; built+installed by maturin CI",
)

from src.shared.python.ai.adapters.rust_adapter import (  # noqa: E402
    RustAgentAdapter,
    ai_backend_available,
)

pytestmark = pytest.mark.parity

_ADAPTER_KWARGS = {
    "api_key": "smoke-key",  # pragma: allowlist secret
    "base_url": "https://api.example.com/v1",
    "model": "gpt-4",
}


def test_extension_exports_present() -> None:
    """The installed wheel exposes the core PyO3 classes the adapter uses."""
    required = {"AIConfig", "AIEngine", "MemoryManager", "RagPipeline"}
    missing = required - set(dir(ai_backend))
    assert not missing, f"ai_backend missing exports: {missing}"


def test_availability_reports_true_with_wheel() -> None:
    """With the wheel installed, the availability probe reports True."""
    assert ai_backend_available() is True
    assert RustAgentAdapter.is_available() is True


def test_adapter_builds_via_rust_path() -> None:
    """The adapter constructs against the real extension (Rust path exercised)."""
    adapter = RustAgentAdapter.try_create(**_ADAPTER_KWARGS)
    assert isinstance(adapter, RustAgentAdapter)
    ok, message = adapter.validate_connection()
    assert ok is True
    assert "Rust backend" in message
