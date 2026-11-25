"""Directory contexts yield paths that still identify the entered directory."""

from pathlib import Path

import pytest
from utils.os_utils import change_directory


@pytest.mark.parametrize("relative", ["child", "./child", Path("child")])
def test_relative_context_yields_usable_absolute_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative: str | Path
) -> None:
    child = tmp_path / "child"
    child.mkdir()
    monkeypatch.chdir(tmp_path)
    with change_directory(relative) as entered:
        assert entered.is_absolute()
        assert entered == Path.cwd() == child
        (entered / "fixture.txt").write_text("fixture")
    assert Path.cwd() == tmp_path
    assert (child / "fixture.txt").read_text() == "fixture"


def test_parent_relative_context_anchors_before_chdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    start = tmp_path / "start"
    sibling = tmp_path / "sibling"
    start.mkdir()
    sibling.mkdir()
    monkeypatch.chdir(start)
    with change_directory("../sibling") as entered:
        assert entered == sibling == Path.cwd()
        assert entered.is_dir()
    assert Path.cwd() == start


def test_nested_relative_contexts_restore_each_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    leaf = tmp_path / "child" / "leaf"
    leaf.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    with change_directory("child") as outer:
        with change_directory("leaf") as inner:
            assert outer == tmp_path / "child"
            assert inner == leaf == Path.cwd()
        assert Path.cwd() == outer
    assert Path.cwd() == tmp_path


def test_nested_failure_restores_outer_then_original_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    leaf = tmp_path / "child" / "leaf"
    leaf.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    error = RuntimeError("fixture failure")
    with change_directory("child") as outer:
        with pytest.raises(RuntimeError) as caught:
            with change_directory("leaf"):
                raise error
        assert caught.value is error
        assert Path.cwd() == outer
    assert Path.cwd() == tmp_path


def test_symlink_context_yields_the_actual_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "linked"
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable on this host")
    monkeypatch.chdir(tmp_path)
    with change_directory("linked") as entered:
        assert entered == target == Path.cwd()
    assert Path.cwd() == tmp_path
