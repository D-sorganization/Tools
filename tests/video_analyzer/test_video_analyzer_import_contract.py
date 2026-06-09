"""Import-contract tests for the ``video_analyzer`` package.

These tests reproduce the way UpstreamDrift's ``external_tools_adapter``
consumes this repository: the repository **root** (only) is placed on
``sys.path`` and the package is imported under the ``src.`` namespace. The
repo's own ``conftest.py``/``pyproject.toml`` ``pythonpath`` entries and the
editable-install finder (which add ``src``, ``src/shared/python`` and map bare
top-level names) are a convenience the consumer does not get, so the import is
exercised in a *subprocess* via ``tests/_import_contract_bootstrap.py`` which
strips those conveniences before importing.

Companion issues: #3273 (architecture), #3280 (testing).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_BOOTSTRAP = REPO_ROOT / "tests" / "_import_contract_bootstrap.py"


def _import_under_consumer_contract(dotted: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_BOOTSTRAP), str(REPO_ROOT), dotted],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT.parent),  # run from outside the repo, like a consumer
    )


def test_package_imports_under_repo_root_only() -> None:
    """``import src.video_analyzer`` must succeed with only the repo root on path."""
    result = _import_under_consumer_contract("src.video_analyzer")
    assert result.returncode == 0, (
        "import src.video_analyzer failed under repo-root-only sys.path:\n"
        f"{result.stderr}"
    )


def test_metadata_available_without_optional_runtime_deps() -> None:
    """Version/type metadata is reachable without importing cv2/mediapipe.

    Accessing ``__version__`` and a metadata type must not trigger the lazy
    heavy-dependency imports, so this resolves even where cv2 is absent.
    """
    result = _import_under_consumer_contract("src.video_analyzer.types")
    assert result.returncode == 0, result.stderr


def test_console_script_target_is_importable() -> None:
    """The declared console-script target is importable without optional deps.

    ``pyproject.toml`` declares ``video-analyzer =
    video_analyzer.launch_video_analyzer:main``. Importing that module and
    resolving ``main`` must not require cv2/mediapipe — assert it without any
    ``importorskip`` guard.
    """
    from video_analyzer.launch_video_analyzer import main

    assert callable(main)
