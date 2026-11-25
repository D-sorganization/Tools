"""Boundary tests for upward path searches."""

from pathlib import Path

import pytest
from utils.path_helpers import find_nearest_dir, find_nearest_file


@pytest.mark.parametrize("kind", ["file", "dir"])
def test_search_reaches_more_than_twenty_parents(tmp_path: Path, kind: str) -> None:
    target = tmp_path / "marker"
    if kind == "file":
        target.write_text("marker", encoding="utf-8")
        find = find_nearest_file
    else:
        target.mkdir()
        find = find_nearest_dir
    leaf = tmp_path.joinpath(*(["d"] * 22))
    leaf.mkdir(parents=True)
    assert find("marker", leaf) == target.resolve()


@pytest.mark.parametrize("kind", ["file", "dir"])
def test_search_checks_root_without_writing_there(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    root = Path(tmp_path.anchor)
    target = root / "codex-test-root-marker-6102"
    real_exists, real_is_file, real_is_dir = Path.exists, Path.is_file, Path.is_dir
    monkeypatch.setattr(
        Path, "exists", lambda p: True if p == target else real_exists(p)
    )
    monkeypatch.setattr(
        Path, "is_file", lambda p: kind == "file" if p == target else real_is_file(p)
    )
    monkeypatch.setattr(
        Path, "is_dir", lambda p: kind == "dir" if p == target else real_is_dir(p)
    )
    find = find_nearest_file if kind == "file" else find_nearest_dir
    assert find(target.name, root) == target


def test_search_prefers_nearest_and_missing_returns_none(tmp_path: Path) -> None:
    (tmp_path / "marker").write_text("outer", encoding="utf-8")
    inner = tmp_path / "inner"
    inner.mkdir()
    (inner / "marker").write_text("inner", encoding="utf-8")
    assert find_nearest_file("marker", inner) == (inner / "marker").resolve()
    assert find_nearest_dir("marker", inner) is None
    assert find_nearest_dir("not-present-6102", inner) is None
