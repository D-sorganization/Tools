"""Import-contract tests for top-level ``src/`` packages.

Several ``src/`` packages are consumed cross-repo (e.g. by UpstreamDrift's
``external_tools_adapter``) with only the repository **root** on ``sys.path``
and imported under the ``src.`` namespace. Their ``__init__`` modules must not
rely on the repo's test-only ``pythonpath`` shims (``src``, ``src/shared/python``)
or on the editable-install finder being present, i.e. they must not eagerly
import via bare, ambiguous-root names.

Each case is imported in a *subprocess* driven by
``tests/_import_contract_bootstrap.py``, which strips editable-install /
virtualenv meta-path finders and ``src`` shim path entries so the import is
resolved exactly the way a cross-repo consumer resolves it.

Issue: #3281 (systemic form of #3273).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_BOOTSTRAP = Path(__file__).resolve().parent / "_import_contract_bootstrap.py"

# Packages whose public API must import under the repo-root-only contract.
_CONTRACT_PACKAGES = [
    "video_analyzer",
    "lower_body_model",
    "rotation_converter",
    "pressure_drop_calculator",
]


def _import_under_consumer_contract(dotted: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_BOOTSTRAP), str(REPO_ROOT), dotted],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT.parent),  # run from outside the repo, like a consumer
    )


@pytest.mark.parametrize("package", _CONTRACT_PACKAGES)
def test_top_level_packages_import_under_repo_root_only(package: str) -> None:
    """``import src.<package>`` must succeed with only the repo root on path."""
    result = _import_under_consumer_contract(f"src.{package}")
    assert (
        result.returncode == 0
    ), f"import src.{package} failed under repo-root-only sys.path:\n{result.stderr}"
