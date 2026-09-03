"""Contracts for scripts/check_wheel_build.py (Tools #4920)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "check_wheel_build.py"

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _load():
    spec = importlib.util.spec_from_file_location("check_wheel_build", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_expected_wheel_name_follows_pep427_normalisation() -> None:
    module = _load()
    assert module.expected_wheel_name("ud-tools", "1.15.0") == (
        "ud_tools-1.15.0-py3-none-any.whl"
    )


def test_project_metadata_reads_pyproject_name_and_version() -> None:
    module = _load()
    name, version = module.project_metadata()
    assert name == "ud-tools"
    assert (REPO_ROOT / "VERSION").read_text(encoding="utf-8").strip() == version


def test_verify_rejects_mismatched_or_empty_wheel(tmp_path: Path) -> None:
    module = _load()
    wrong = tmp_path / "ud_tools-0.0.1-py3-none-any.whl"
    wrong.write_bytes(b"x" * 2048)
    assert module.verify(wrong, "ud-tools", "1.15.0") == [
        "wheel is ud_tools-0.0.1-py3-none-any.whl, pyproject says "
        "ud_tools-1.15.0-py3-none-any.whl"
    ]
    tiny = tmp_path / "ud_tools-1.15.0-py3-none-any.whl"
    tiny.write_bytes(b"x")
    problems = module.verify(tiny, "ud-tools", "1.15.0")
    assert problems and "implausibly small" in problems[0]
    good = tmp_path / "ok" / "ud_tools-1.15.0-py3-none-any.whl"
    good.parent.mkdir()
    good.write_bytes(b"x" * 2048)
    assert module.verify(good, "ud-tools", "1.15.0") == []


def test_release_workflow_builds_wheel_and_sbom_and_attaches_them() -> None:
    import yaml

    workflow = yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(
            encoding="utf-8"
        )
    )
    jobs = workflow["jobs"]
    validate = next(
        step
        for step in jobs["validate"]["steps"]
        if step.get("name") == "Wheel builds with pyproject name/version"
    )
    assert "scripts/check_wheel_build.py --check" in validate["run"]
    build = next(
        step
        for step in jobs["github-release"]["steps"]
        if step.get("name") == "Build source distribution, wheel and SBOM"
    )
    assert "scripts/check_wheel_build.py --outdir dist" in build["run"]
    assert "cyclonedx-py requirements requirements.txt" in build["run"]
    create = next(
        step
        for step in jobs["github-release"]["steps"]
        if step.get("name") == "Create GitHub Release"
    )
    assert "dist/*.whl" in create["run"]
    assert "dist/*.sbom.cdx.json" in create["run"]
    artifact = jobs["wheel-artifact"]
    assert artifact["if"] == "github.event_name == 'push'"
    upload = next(
        step
        for step in artifact["steps"]
        if "upload-artifact" in str(step.get("uses", ""))
    )
    assert upload["with"]["name"] == "tools-wheel-${{ github.sha }}"
    assert upload["with"]["path"] == "dist/*.whl"
