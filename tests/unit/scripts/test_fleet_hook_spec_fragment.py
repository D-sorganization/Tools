"""Fleet hook SPEC freshness must accept only valid tracked change fragments."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pytest

from shared_scripts import fleet_hooks

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

REPO_ROOT = Path(__file__).resolve().parents[3]
VALID_FRAGMENT = (
    "---\n"
    "issue: 5686\n"
    'summary: "Honor valid change fragments in the SPEC freshness hook"\n'
    "---\n"
)
INVALID_FRAGMENT = "---\nissue: 5686\n---\n"


@pytest.fixture
def hook_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, list[str]]:
    """Point the real hook at an isolated repo with its shipped validators."""
    helper_dir = tmp_path / "shared_scripts"
    shutil.copytree(REPO_ROOT / "shared_scripts", helper_dir)
    (tmp_path / "SPEC.md").write_text("# Spec\n", encoding="utf-8")
    monkeypatch.setattr(fleet_hooks, "ROOT", tmp_path)
    monkeypatch.setattr(fleet_hooks, "__file__", str(helper_dir / "fleet_hooks.py"))
    changed = ["tests/unit/sample_test.py"]
    monkeypatch.setattr(fleet_hooks, "staged_files", lambda: changed)
    monkeypatch.setattr(fleet_hooks, "changed_files", lambda: changed)
    return tmp_path, changed


def _run_freshness_check() -> int:
    return fleet_hooks.check_spec_freshness(argparse.Namespace(warn_only=False))


def _write_fragment(root: Path, filename: str, contents: str) -> str:
    path = root / "changes" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")
    return path.relative_to(root).as_posix()


def test_valid_fragment_substitutes_for_spec_edit(
    hook_repo: tuple[Path, list[str]],
) -> None:
    """A normal source/test change with a valid fragment does not need SPEC."""
    root, changed = hook_repo
    changed.append(_write_fragment(root, "5686-spec-freshness.md", VALID_FRAGMENT))

    assert _run_freshness_check() == 0


def test_source_change_without_fragment_still_requires_spec_edit(
    hook_repo: tuple[Path, list[str]],
) -> None:
    assert _run_freshness_check() == 1


def test_invalid_fragment_does_not_substitute_for_spec_edit(
    hook_repo: tuple[Path, list[str]],
) -> None:
    root, changed = hook_repo
    changed.append(_write_fragment(root, "5686-invalid.md", INVALID_FRAGMENT))

    assert _run_freshness_check() == 1


def test_deleted_fragment_does_not_substitute_for_spec_edit(
    hook_repo: tuple[Path, list[str]],
) -> None:
    _root, changed = hook_repo
    changed.append("changes/5686-deleted.md")

    assert _run_freshness_check() == 1


def test_invalid_fragment_spoils_a_valid_fragment(
    hook_repo: tuple[Path, list[str]],
) -> None:
    root, changed = hook_repo
    changed.append(_write_fragment(root, "5686-good.md", VALID_FRAGMENT))
    changed.append(_write_fragment(root, "5686-bad.md", INVALID_FRAGMENT))

    assert _run_freshness_check() == 1


def test_missing_fragment_validator_fails_closed(
    hook_repo: tuple[Path, list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, changed = hook_repo
    changed.append(_write_fragment(root, "5686-no-validator.md", VALID_FRAGMENT))
    (root / "shared_scripts" / "changes_fragment.py").unlink()
    monkeypatch.setattr(
        fleet_hooks, "__file__", str(root / "shared_scripts" / "fleet_hooks.py")
    )

    assert _run_freshness_check() == 1


def test_spec_edit_alone_still_satisfies_freshness(
    hook_repo: tuple[Path, list[str]],
) -> None:
    _root, changed = hook_repo
    changed.append("SPEC.md")

    assert _run_freshness_check() == 0
