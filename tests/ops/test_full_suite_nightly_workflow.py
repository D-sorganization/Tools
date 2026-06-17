from __future__ import annotations

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_full_suite_test_extra_declares_collection_runtime_dependencies() -> None:
    workflow = (
        REPO_ROOT / ".github" / "workflows" / "full-suite-nightly.yml"
    ).read_text(encoding="utf-8")
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert 'pip install -e ".[dev,test]"' in workflow

    test_deps = config["project"]["optional-dependencies"]["test"]
    dep_names = {dep.split(">=", maxsplit=1)[0] for dep in test_deps}

    assert {
        "fastapi",
        "httpx",
        "opencv-python-headless",
        "python-multipart",
    }.issubset(dep_names)
