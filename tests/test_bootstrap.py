import sys
from pathlib import Path

import pytest

from _bootstrap import bootstrap


def test_bootstrap_invalid_args() -> None:
    bad_arg: object = 123
    with pytest.raises(TypeError):
        bootstrap(bad_arg)
    with pytest.raises(ValueError):
        bootstrap("")


def test_bootstrap_resolves_paths(tmp_path: Path) -> None:
    # Setup dummy directory structure to mock a repo
    repo_dir = tmp_path / "dummy_repo"
    repo_dir.mkdir()

    # Create marker file
    (repo_dir / "pyproject.toml").touch()

    # Create caller file under a subdirectory
    sub_dir = repo_dir / "src" / "tools"
    sub_dir.mkdir(parents=True)
    caller_file = sub_dir / "launcher.py"
    caller_file.touch()

    # Create dummy shared directory
    shared_dir = repo_dir / "src" / "shared" / "python"
    shared_dir.mkdir(parents=True)

    # Backup sys.path
    original_path = list(sys.path)
    try:
        resolved_root = bootstrap(str(caller_file))
        assert resolved_root.resolve() == repo_dir.resolve()

        # Verify sys.path updates (src and src/python/src roots)
        assert str((repo_dir / "src").resolve()) in sys.path
        assert str((repo_dir / "src" / "python" / "src").resolve()) in sys.path
        # shared/python is no longer directly added to sys.path since #3316
        assert str(shared_dir.resolve()) not in sys.path
    finally:
        # Restore sys.path
        sys.path = original_path
