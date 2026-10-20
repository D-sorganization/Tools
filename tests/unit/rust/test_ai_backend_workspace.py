"""
TDD validation tests for issue #2870:
[Sidekick] Operationalize Rust ai_backend via Maturin CI Pipeline

These tests assert the acceptance criteria BEFORE implementation so that
the green phase proves the pipeline is correctly wired up.

Run with:
    pytest tests/unit/rust/test_ai_backend_workspace.py -v
"""

import pathlib
from typing import Any, cast

import pytest
import yaml

# Resolve repo root relative to this test file (tests/unit/rust/ -> repo root)
REPO_ROOT = pathlib.Path(__file__).parent.parent.parent.parent
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "maturin-ai-backend.yml"
PYTHON_VERSIONS = ["3.10", "3.11", "3.12", "3.13"]


def _workflow() -> dict[str, Any]:
    """Read the workflow as YAML so comments and cosmetic labels cannot pass."""
    return cast(dict[str, Any], yaml.safe_load(WORKFLOW.read_text(encoding="utf-8")))


def _step(job: dict[str, Any], name: str) -> dict[str, Any]:
    return next(step for step in job["steps"] if step.get("name") == name)


@pytest.mark.unit
def test_primary_build_keeps_clippy_features_fork_guard_and_failure_propagation():
    """Matrix reduction must not weaken the existing native wheel checks."""
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / "maturin-ai-backend.yml").read_text(
            encoding="utf-8"
        )
    )
    build = workflow["jobs"]["build-wheels"]
    steps = build["steps"]
    clippy = next(
        step
        for step in steps
        if step.get("name") == "Clippy - ai_backend (python feature)"
    )
    wheel = next(step for step in steps if step.get("name") == "Build wheel")

    assert build["timeout-minutes"] == 30
    assert "head.repo.full_name == github.repository" in build["if"]
    assert clippy["run"] == "cargo clippy --features python -- -D warnings"
    assert "--release --features python,extension-module" in wheel["run"]
    assert all(step.get("continue-on-error") is not True for step in steps)


@pytest.mark.unit
def test_required_wheel_build_matrix_is_four_python_versions_on_one_native_runner():
    """Platform labels must not multiply Linux executions into fake OS cells."""
    build = _workflow()["jobs"]["build-wheels"]

    assert build["runs-on"] == "d-sorg-fleet"
    assert build["strategy"]["fail-fast"] is False
    assert build["strategy"]["matrix"] == {"python-version": PYTHON_VERSIONS}


@pytest.mark.unit
def test_wheel_cache_output_and_artifact_key_use_actual_runner_identity() -> None:
    """Cache, file and upload identities must use runner OS/architecture."""
    build = _workflow()["jobs"]["build-wheels"]
    steps = build["steps"]
    cache = _step(build, "Cache Cargo registry")["with"]
    build_step = _step(build, "Build wheel")
    upload = _step(build, "Upload wheel artifact")["with"]

    for value in (cache["key"], upload["name"], build_step["run"]):
        assert "${{ runner.os }}" in value
        assert "${{ runner.arch }}" in value
        assert "${{ matrix.python-version }}" in value
    assert upload["path"] == (
        "dist/${{ runner.os }}/${{ runner.arch }}/py${{ matrix.python-version }}/*"
    )
    assert "${{ matrix.platform }}" not in " ".join(
        str(value) for value in (cache["key"], upload["name"], build_step["run"])
    )
    identity = (
        "wheel-${{ runner.os }}-${{ runner.arch }}-py${{ matrix.python-version }}"
    )
    assert upload["name"] == identity
    smoke = _step(build, "Smoke-test wheel import")["run"]
    assert "dist/${{ runner.os }}/${{ runner.arch }}/" in smoke
    assert "py${{ matrix.python-version }}" in smoke
    assert any(step.get("name") == "Record native wheel evidence" for step in steps)


@pytest.mark.unit
def test_native_wheel_evidence_checks_runtime_and_installed_wheel_tag() -> None:
    """Reported qualification must come from the interpreter and built wheel."""
    build = _workflow()["jobs"]["build-wheels"]
    evidence = _step(build, "Record native wheel evidence")["run"]
    smoke = _step(build, "Smoke-test wheel import")["run"]

    for marker in (
        "platform.machine()",
        "sys.version",
        "sys.executable",
        "sys_tags()",
        "parse_wheel_filename",
    ):
        assert marker in evidence
    assert "assert" in evidence
    assert "ai_backend" in smoke
    assert "shopt -s nullglob" in smoke
    assert "${#WHEELS[@]} != 1" in smoke
    assert "Expected exactly one wheel" in smoke
    assert "GITHUB_STEP_SUMMARY" in evidence
    assert (
        "wheel-${{ runner.os }}-${{ runner.arch }}-py${{ matrix.python-version }}"
        in evidence
    )


@pytest.mark.unit
def test_adapter_gate_remains_independent_and_manual_feature_build_is_unchanged() -> (
    None
):
    """Keep the real consumer gate and opt-in local-embeddings path intact."""
    jobs = _workflow()["jobs"]
    adapter = jobs["adapter-rust-path"]
    optional = jobs["build-local-embeddings"]

    assert "needs" not in adapter
    assert adapter["timeout-minutes"] == 30
    adapter_build = _step(adapter, "Build ai_backend wheel (python feature only)")[
        "run"
    ]
    adapter_install = _step(adapter, "Install the built wheel")["run"]
    assert "--features python,extension-module" in adapter_build
    assert "--out ../../dist/adapter-gate" in adapter_build
    assert "dist/adapter-gate/*.whl" in adapter_install
    assert (
        "test_rust_adapter_wheel_smoke.py"
        in _step(
            adapter,
            "Exercise the RustAgentAdapter Rust path (wheel-installed smoke gate)",
        )["run"]
    )
    assert "workflow_dispatch" in optional["if"]
    assert (
        "local-embeddings" in _step(optional, "Build wheel (local-embeddings)")["run"]
    )


@pytest.mark.unit
def test_workflow_keeps_windows_and_macos_qualification_explicitly_open() -> None:
    """Removing duplicate Linux jobs must not imply three-OS qualification."""
    content = WORKFLOW.read_text(encoding="utf-8").lower()

    assert "windows/macos qualification remains open" in content
    assert "issue #5684" in content


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
