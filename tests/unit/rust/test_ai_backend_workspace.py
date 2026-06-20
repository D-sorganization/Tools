"""
TDD validation tests for issue #2870:
[Sidekick] Operationalize Rust ai_backend via Maturin CI Pipeline

These tests assert the acceptance criteria BEFORE implementation so that
the green phase proves the pipeline is correctly wired up.

Run with:
    pytest tests/unit/rust/test_ai_backend_workspace.py -v
"""

import pathlib

import pytest

# Resolve repo root relative to this test file (tests/unit/rust/ -> repo root)
REPO_ROOT = pathlib.Path(__file__).parent.parent.parent.parent


@pytest.mark.unit
def test_ai_backend_in_cargo_workspace():
    """ai_backend must be a declared workspace member in the root Cargo.toml."""
    cargo_toml = (REPO_ROOT / "Cargo.toml").read_text(encoding="utf-8")
    assert "ai_backend" in cargo_toml, (
        "rust_core/ai_backend is not listed in the root workspace Cargo.toml members"
    )


@pytest.mark.unit
def test_ai_backend_cargo_toml_exists():
    """The ai_backend crate must have its own Cargo.toml."""
    crate_toml = REPO_ROOT / "rust_core" / "ai_backend" / "Cargo.toml"
    assert crate_toml.exists(), f"Missing crate manifest: {crate_toml}"


@pytest.mark.unit
def test_maturin_ci_workflow_exists():
    """A CI workflow file for maturin / ai_backend builds must exist."""
    workflows_dir = REPO_ROOT / ".github" / "workflows"
    candidates = (
        list(workflows_dir.glob("*maturin*"))
        + list(workflows_dir.glob("*ai_backend*"))
        + list(workflows_dir.glob("*ai-backend*"))
    )
    assert len(candidates) > 0, (
        f"No maturin/ai_backend CI workflow found under {workflows_dir}. "
        "Expected a file matching *maturin* or *ai_backend* or *ai-backend*."
    )


@pytest.mark.unit
def test_maturin_ci_covers_all_platforms():
    """CI workflow must target Windows, macOS, and Linux runners."""
    workflows_dir = REPO_ROOT / ".github" / "workflows"
    candidates = (
        list(workflows_dir.glob("*maturin*"))
        + list(workflows_dir.glob("*ai_backend*"))
        + list(workflows_dir.glob("*ai-backend*"))
    )
    assert candidates, "No maturin CI workflow found — cannot check platform coverage."

    for wf_path in candidates:
        content = wf_path.read_text(encoding="utf-8").lower()
        assert "windows" in content, f"{wf_path.name}: missing Windows runner"
        assert "ubuntu" in content or "linux" in content, (
            f"{wf_path.name}: missing Ubuntu/Linux runner"
        )
        assert "macos" in content or "mac" in content, (
            f"{wf_path.name}: missing macOS runner"
        )


@pytest.mark.unit
def test_maturin_ci_covers_python_versions():
    """Maturin CI must hard-gate supported fleet versions and document 3.13 gaps."""
    workflows_dir = REPO_ROOT / ".github" / "workflows"
    candidates = (
        list(workflows_dir.glob("*maturin*"))
        + list(workflows_dir.glob("*ai_backend*"))
        + list(workflows_dir.glob("*ai-backend*"))
    )
    assert candidates, (
        "No maturin CI workflow found — cannot check Python version coverage."
    )

    fleet_toolcache_limited = {
        "maturin-data-processor-core.yml",
        "maturin-file-watcher.yml",
        "maturin-movement-optimizer.yml",
        "maturin-pendulum-core.yml",
    }
    for wf_path in candidates:
        content = wf_path.read_text(encoding="utf-8")
        for version in ["3.10", "3.11", "3.12"]:
            assert version in content, f"Python {version} not listed in {wf_path.name}"
        if wf_path.name in fleet_toolcache_limited:
            assert "3.13" in content, (
                f"{wf_path.name}: must document why Python 3.13 is not hard-gated"
            )
            assert "toolcache" in content.lower(), (
                f"{wf_path.name}: Python 3.13 deferral must cite runner "
                "toolcache limits"
            )
            continue
        assert "3.13" in content, f"Python 3.13 not listed in {wf_path.name}"


@pytest.mark.unit
def test_ort_dylib_documented_in_claude_md():
    """CLAUDE.md must document ORT_DYLIB_PATH for local-embeddings consumers."""
    claude_md = (REPO_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    assert "ORT_DYLIB_PATH" in claude_md, (
        "CLAUDE.md does not mention ORT_DYLIB_PATH. "
        "Add a section describing how to point ort at the system ONNX Runtime library."
    )


@pytest.mark.unit
def test_ai_backend_cargo_toml_declares_python_feature():
    """ai_backend Cargo.toml must declare a 'python' feature gating PyO3 bindings."""
    crate_toml = (REPO_ROOT / "rust_core" / "ai_backend" / "Cargo.toml").read_text(
        encoding="utf-8"
    )
    assert "python" in crate_toml, (
        "rust_core/ai_backend/Cargo.toml does not declare a 'python' feature. "
        "Maturin needs this feature to activate pyo3/extension-module linkage."
    )


@pytest.mark.unit
def test_ai_backend_cargo_toml_declares_local_embeddings_feature():
    """ai_backend Cargo.toml must declare a 'local-embeddings' feature."""
    crate_toml = (REPO_ROOT / "rust_core" / "ai_backend" / "Cargo.toml").read_text(
        encoding="utf-8"
    )
    assert "local-embeddings" in crate_toml, (
        "rust_core/ai_backend/Cargo.toml does not declare 'local-embeddings' feature."
    )


@pytest.mark.unit
def test_maturin_workflow_references_maturin_action_or_command():
    """CI workflow must actually invoke maturin (action or CLI command)."""
    workflows_dir = REPO_ROOT / ".github" / "workflows"
    candidates = (
        list(workflows_dir.glob("*maturin*"))
        + list(workflows_dir.glob("*ai_backend*"))
        + list(workflows_dir.glob("*ai-backend*"))
    )
    assert candidates, "No maturin CI workflow found."

    for wf_path in candidates:
        content = wf_path.read_text(encoding="utf-8").lower()
        assert "maturin" in content, (
            f"{wf_path.name}: the workflow does not reference 'maturin' — "
            "it must call maturin build or use PyO3/maturin-action."
        )
