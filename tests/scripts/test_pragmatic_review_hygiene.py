from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_pragmatic_review_has_no_discarded_defaultdict_statement() -> None:
    review_script = REPO_ROOT / "scripts" / "pragmatic_programmer_review.py"
    lines = review_script.read_text(encoding="utf-8").splitlines()

    assert "    defaultdict(list)" not in lines


def test_scripts_do_not_duplicate_ble001_suppressions() -> None:
    duplicated_suppression = "# noqa: BLE001  # noqa: BLE001"
    offenders = [
        path.relative_to(REPO_ROOT)
        for path in (REPO_ROOT / "scripts").rglob("*.py")
        if duplicated_suppression in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


def test_readme_launcher_hierarchy_names_only_existing_launchers() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "UnifiedToolsLauncher.py" in readme
    assert (REPO_ROOT / "UnifiedToolsLauncher.py").is_file()
    assert "`launch_tools_main.py`" not in readme
    assert "`Launcher.py`" not in readme
