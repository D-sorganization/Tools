"""Fresh-process Signal toolkit import identity contracts (#5572)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "first",
    [
        "signal_toolkit",
        "shared.python.signal_toolkit",
        "src.shared.python.signal_toolkit",
    ],
)
def test_signal_import_orders_keep_one_provider_identity(
    first: str, tmp_path: Path
) -> None:
    """Children imported during package initialization must retain one Signal."""
    repo_root = Path(__file__).resolve().parents[4]
    script = """
import importlib
import pathlib
import sys

first = sys.argv[1]
package = importlib.import_module(first)
early = {
    name: module
    for name, module in sys.modules.items()
    if name.startswith(first + ".") and getattr(module, "Signal", None) is not None
}
names = (
    "signal_toolkit",
    "shared.python.signal_toolkit",
    "src.shared.python.signal_toolkit",
)
packages = [importlib.import_module(name) for name in names]
assert all(candidate is packages[0] for candidate in packages), names
core = [importlib.import_module(name + ".core") for name in names]
fitting = [importlib.import_module(name + ".fitting") for name in names]
io = [importlib.import_module(name + ".io") for name in names]
assert all(candidate is core[0] for candidate in core), core
assert all(candidate is fitting[0] for candidate in fitting), fitting
assert all(candidate is io[0] for candidate in io), io
assert all(candidate.Signal is core[0].Signal for candidate in (*fitting, *io))
assert all(candidate.Signal is core[0].Signal for candidate in packages)
assert all(module.Signal is core[0].Signal for module in early.values())
assert pathlib.Path(core[0].__file__).resolve().is_relative_to(
    pathlib.Path(sys.argv[2])
)
assert all(not name.startswith("_pinned_") for name in sys.modules)
"""
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            (
                str(repo_root),
                str(repo_root / "src" / "shared" / "python"),
                str(repo_root / "src"),
            )
        ),
    }
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            first,
            str(repo_root / "src" / "shared" / "python"),
        ],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_bare_first_keeps_downstream_src_namespace(tmp_path: Path) -> None:
    """Early binding must not replace an unrelated downstream ``src`` tree."""
    repo_root = Path(__file__).resolve().parents[4]
    downstream_root = tmp_path / "downstream"
    downstream_src = downstream_root / "src" / "shared" / "python"
    downstream_src.mkdir(parents=True)
    for package in (downstream_root / "src", downstream_src.parent, downstream_src):
        (package / "__init__.py").write_text("DOWNSTREAM = True\n", encoding="utf-8")
    script = """
import importlib
import sys

downstream = importlib.import_module("src.shared.python")
bare = importlib.import_module("signal_toolkit")
early_core = importlib.import_module("signal_toolkit.core")
early_fitting = importlib.import_module("signal_toolkit.fitting")
early_io = importlib.import_module("signal_toolkit.io")
canonical = importlib.import_module("shared.python.signal_toolkit")
assert bare is canonical
assert early_core is importlib.import_module("shared.python.signal_toolkit.core")
assert early_fitting is importlib.import_module("shared.python.signal_toolkit.fitting")
assert early_io is importlib.import_module("shared.python.signal_toolkit.io")
assert early_fitting.Signal is early_io.Signal is early_core.Signal is bare.Signal
assert sys.modules["src.shared.python"] is downstream
assert downstream.DOWNSTREAM is True
"""
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            (
                str(downstream_root),
                str(repo_root / "src" / "shared" / "python"),
                str(repo_root / "src"),
            )
        ),
    }
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
