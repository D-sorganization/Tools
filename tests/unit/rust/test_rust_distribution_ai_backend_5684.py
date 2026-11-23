"""#5684 / #6023: operator docs must match Linux-only ai_backend wheel qualification."""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
RUST_DIST = _REPO / "docs" / "development" / "rust_distribution.md"
WORKFLOW = _REPO / ".github" / "workflows" / "maturin-ai-backend.yml"


@pytest.mark.unit
def test_rust_distribution_documents_ai_backend_linux_workflow() -> None:
    """Slice 3 truth: qualification exists on fleet Linux, not only manual maturin."""
    text = RUST_DIST.read_text(encoding="utf-8")
    assert RUST_DIST.is_file()
    assert WORKFLOW.is_file()
    assert "maturin-ai-backend.yml" in text
    assert "linux-wheel-qualify" in text
    assert "d-sorg-fleet" in text


@pytest.mark.unit
def test_rust_distribution_does_not_claim_three_os_ai_backend_qualification() -> None:
    """Must not present hosted three-OS matrices as current ai_backend CI."""
    lowered = RUST_DIST.read_text(encoding="utf-8").lower()
    for token in ("ubuntu-latest", "windows-latest", "macos-latest"):
        assert token not in lowered, (
            f"{RUST_DIST.name} must not cite {token} as current ai_backend wheel CI"
        )
    assert "#5684" in lowered or "5684" in lowered
    assert "windows and macos" in lowered or "windows/mac" in lowered
