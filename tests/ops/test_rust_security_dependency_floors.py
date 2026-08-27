"""Regression contracts for RustSec-remediated workspace dependencies."""

from __future__ import annotations

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _manifest(relative_path: str) -> dict[str, object]:
    with (REPO_ROOT / relative_path).open("rb") as stream:
        return tomllib.load(stream)


def _major_minor(version: str) -> tuple[int, int]:
    major, minor, *_ = version.split(".")
    return int(major), int(minor)


def test_workspace_python_bindings_exclude_advisory_affected_pyo3() -> None:
    """RUSTSEC-2026-0176/0177 require PyO3 0.29 or newer."""
    manifest = _manifest("Cargo.toml")
    workspace = manifest["workspace"]
    assert isinstance(workspace, dict)
    dependencies = workspace["dependencies"]
    assert isinstance(dependencies, dict)
    pyo3 = dependencies["pyo3"]
    numpy = dependencies["numpy"]
    assert isinstance(pyo3, dict)
    assert _major_minor(str(pyo3["version"])) >= (0, 29)
    assert _major_minor(str(numpy)) >= (0, 29)

    math_manifest = _manifest("rust_core/math-primitives/Cargo.toml")
    math_dependencies = math_manifest["dependencies"]
    assert isinstance(math_dependencies, dict)
    math_pyo3 = math_dependencies["pyo3"]
    assert isinstance(math_pyo3, dict)
    assert math_pyo3.get("workspace") is True


def test_http_client_excludes_h2_empty_data_frame_advisory() -> None:
    """Reqwest 0.12 resolves to the fixed h2 0.4 dependency line."""
    manifest = _manifest("rust_core/ai_backend/Cargo.toml")
    dependencies = manifest["dependencies"]
    assert isinstance(dependencies, dict)
    reqwest = dependencies["reqwest"]
    assert isinstance(reqwest, dict)
    assert _major_minor(str(reqwest["version"])) >= (0, 12)


def test_workspace_rust_floor_supports_pyo3_029() -> None:
    """The declared compiler floor must match PyO3 0.29's MSRV."""
    manifest = _manifest("Cargo.toml")
    workspace = manifest["workspace"]
    assert isinstance(workspace, dict)
    package = workspace["package"]
    assert isinstance(package, dict)
    assert tuple(map(int, str(package["rust-version"]).split("."))) >= (1, 83)
