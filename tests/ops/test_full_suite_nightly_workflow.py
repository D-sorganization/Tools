from __future__ import annotations

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "full-suite-nightly.yml"


def _step_script(step_name: str) -> str:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["full-suite"]["steps"]
    return next(step["run"] for step in steps if step.get("name") == step_name)


def _dependency_name(spec: str) -> str:
    return (
        spec.split(";", maxsplit=1)[0]
        .split("[", maxsplit=1)[0]
        .split(">=", maxsplit=1)[0]
        .split("==", maxsplit=1)[0]
        .strip()
    )


def test_full_suite_test_extra_declares_collection_runtime_dependencies() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert 'pip install -e ".[dev,test]"' in workflow
    assert "|| pip install -e ." not in workflow

    test_deps = config["project"]["optional-dependencies"]["test"]
    dep_names = {_dependency_name(dep) for dep in test_deps}

    assert {
        "ezdxf",
        "fastapi",
        "httpx",
        "opencv-python-headless",
        "pymodbus",
        "python-multipart",
        "requests",
        "sqlmodel",
    }.issubset(dep_names)


def test_full_suite_nightly_hard_fails_missing_collection_dependencies() -> None:
    script = _step_script("Verify full-suite collection dependencies")

    assert "missing full-suite dependencies" in script
    assert "silently skip affected tests" in script
    for module_name in ("cv2", "ezdxf", "pymodbus", "requests", "sqlmodel"):
        assert module_name in script


def test_full_suite_nightly_disables_xdist_on_fleet_runners() -> None:
    script = _step_script("Run full suite")

    assert "-n 0" in script
    assert "-n auto" not in script
    assert "--dist no" in script
    assert "--dist loadscope" not in script
    assert "full_suite_results.xml" in script


def test_full_suite_nightly_keeps_collection_floor_guard() -> None:
    script = _step_script("Summarize full-suite results")

    assert "total < 500" in script
    assert "collection broken" in script
    assert "sys.exit(1)" in script
