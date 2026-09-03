"""Pin the `rate_of_closure` surface UpstreamDrift's ADR-0046 drift gates import.

UpstreamDrift's `tests/integration/launch_monitor_drift/` is the safety net for
ADR-0046: one deterministic synthetic session feeds both launch-monitor stacks
and every legitimate divergence is pinned by number. Those gates reach the Tools
stack by **importing `rate_of_closure` module paths and symbol names directly**,
so an ADR-0046 Stage 2 retirement that moves or deletes one of them breaks the
gate from the far side of a vendor pin.

That breakage is unusually expensive to notice. ADR-0048's risk section records
that the drift suite skips itself when `vendor/ud-tools` is not materialised, and
a skipped gate reports green; the module-level `from rate_of_closure... import`
statements below the skip guard would raise `ImportError` at collection in the
one job that does materialise the pin, long after the Tools PR merged.

This test moves the failure to the Tools side. It is deliberately a *static*
pin: it asserts nothing about behaviour or numbers — the drift gates own that —
only that every module path and symbol name they bind still resolves, and that
binding them does not drag in a GUI toolkit the headless gate does not install.

`docs/specs/LAUNCH_MONITOR_ANALYTICS.md` §"ADR-0046 Stage 2 — Canonical-Layer
Mapping" records why none of these modules is retirable yet. When one becomes
retirable, the constraint is not "delete the row" — it is to leave a re-export at
the old path with a deprecation note, so the gate keeps resolving, and to update
this table in the same PR.

Regenerate the expected surface from an UpstreamDrift checkout with::

    grep -rn 'from rate_of_closure' tests/integration/launch_monitor_drift/*.py
"""

from __future__ import annotations

import ast
import importlib
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

# Derived 2026-09-03 from UpstreamDrift `tests/integration/launch_monitor_drift/`
# at vendor pin `e88a334c` (six gate files, 71 gates, 20 bound symbols).
DRIFT_GATE_SURFACE: dict[str, tuple[str, ...]] = {
    "rate_of_closure._launch_monitor_analysis_types": (
        "CONTRACT_VERSION",
        "AnalysisRequest",
    ),
    "rate_of_closure._player_covariation_types": (
        "MIN_FISHER_SAMPLES",
        "CovariationRequest",
        "PairScanRequest",
    ),
    "rate_of_closure.launch_monitor_analysis": (
        "analyze_launch_monitor_data",
        "numeric_columns",
    ),
    "rate_of_closure.launch_monitor_linked_scatter": ("MAX_RETAINED_ROWS",),
    "rate_of_closure.launch_monitor_longitudinal": (
        "LongitudinalRequest",
        "analyze_longitudinal_performance",
    ),
    "rate_of_closure.launch_monitor_performance": (
        "DispersionRequest",
        "analyze_dispersion",
    ),
    "rate_of_closure.launch_monitor_private_corpus": (
        "CORPUS_RELATIVE_PATH",
        "PRIVATE_DATA_ENV",
        "load_private_corpus",
        "resolve_private_corpus_path",
    ),
    "rate_of_closure.launch_monitor_strokes_gained": (
        "SourceBackedStrokesGainedRequest",
        "calculate_source_backed_strokes_gained",
    ),
    "rate_of_closure.launch_monitor_strokes_gained_baseline": (
        "baseline_table_hash",
        "load_strokes_gained_baseline",
    ),
    "rate_of_closure.player_covariation": (
        "analyze_player_covariation",
        "scan_covariation_pairs",
    ),
}

# `require_vendored_tools_stack()` in the drift suite's conftest decides between
# "skip" and "hard fail" by probing exactly this file inside the vendored tree.
# If it ever moves, every gate silently stops running.
VENDOR_MATERIALISATION_SENTINEL = "src/rate_of_closure/launch_monitor_strokes_gained.py"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_STAGE2_DOC = _REPO_ROOT / "docs" / "specs" / "LAUNCH_MONITOR_ANALYTICS.md"
_STAGE2_HEADING = "## ADR-0046 Stage 2 — Canonical-Layer Mapping"


@pytest.mark.parametrize("module_path", sorted(DRIFT_GATE_SURFACE))
def test_drift_gate_module_paths_still_resolve(module_path: str) -> None:
    """Every module UpstreamDrift's gates import by name still imports."""
    importlib.import_module(module_path)


@pytest.mark.parametrize(
    ("module_path", "symbol"),
    [
        (module_path, symbol)
        for module_path, symbols in sorted(DRIFT_GATE_SURFACE.items())
        for symbol in symbols
    ],
)
def test_drift_gate_symbols_still_resolve(module_path: str, symbol: str) -> None:
    """Every symbol UpstreamDrift's gates bind is still exported."""
    module = importlib.import_module(module_path)
    assert hasattr(module, symbol), (
        f"{module_path}.{symbol} is bound by UpstreamDrift's ADR-0046 drift "
        "gates. Retiring it needs a re-export at this path with a deprecation "
        "note, in a PR paired with the UpstreamDrift change."
    )


def test_drift_gate_surface_needs_no_gui_toolkit() -> None:
    """The gates run headless; none of these modules may pull in PyQt6.

    UpstreamDrift's gate job installs the Tools *library* surface, not the
    desktop one. A stray `rate_of_closure.ui` import inside one of these modules
    would turn all 71 gates into a collection error there while every Tools
    suite stayed green here.

    This runs in a subprocess on purpose: `sys.modules` in the pytest process
    already carries whatever the rest of the suite imported, so an in-process
    assertion would measure the suite rather than these modules.
    """
    probe = (
        "import importlib, sys\n"
        f"for name in {sorted(DRIFT_GATE_SURFACE)!r}:\n"
        "    importlib.import_module(name)\n"
        "print('PyQt6' in sys.modules)\n"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        path for path in sys.path if path and Path(path).is_dir()
    )
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        env=environment,
        cwd=_REPO_ROOT,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "False", completed.stdout


def test_vendor_materialisation_sentinel_exists() -> None:
    """The file the drift suite probes to choose skip-vs-fail is still there."""
    assert (_REPO_ROOT / VENDOR_MATERIALISATION_SENTINEL).is_file()


def test_drift_gate_surface_modules_are_import_only_paths() -> None:
    """Each pinned module path names a real file, not a package re-export.

    The gates spell out submodule paths (`rate_of_closure.launch_monitor_x`),
    not the package facade, so a Stage 2 move that keeps only a facade alias
    would still break them.
    """
    for module_path in sorted(DRIFT_GATE_SURFACE):
        relative = Path(*module_path.split(".")).with_suffix(".py")
        assert (_REPO_ROOT / "src" / relative).is_file(), relative


def _stage2_documented_modules() -> set[str]:
    text = _STAGE2_DOC.read_text(encoding="utf-8")
    start = text.index(_STAGE2_HEADING)
    end = text.index("\n## ", start + len(_STAGE2_HEADING))
    return set(
        re.findall(
            r"`(_?(?:launch_monitor|player_covariation)\w*\.py)`", text[start:end]
        )
    )


def test_every_launch_monitor_module_carries_a_stage2_classification() -> None:
    """The Stage 2 mapping table covers the whole application layer.

    ADR-0046 Stage 2's deliverable is a decision per module, so a new
    `launch_monitor_*` module that nobody classified is the failure mode this
    guards. `launch_monitor_workspace*` and `launch_monitor_v2_client` are the
    documented exclusions: ADR-0048 classifies the workspace pair `app-local`
    (two retention postures, not one divergence) and the v2 client is the HTTP
    seam whose fate ADR-0048 leaves to an explicit owner statement.
    """
    excluded = {
        "launch_monitor_v2_client.py",
        "launch_monitor_workspace.py",
        "launch_monitor_workspace_v3.py",
    }
    package = _REPO_ROOT / "src" / "rate_of_closure"
    on_disk = {
        path.name
        for path in package.glob("*.py")
        if re.fullmatch(r"_?(launch_monitor|player_covariation)\w*\.py", path.name)
    } - excluded
    assert on_disk <= _stage2_documented_modules()


def test_stage2_table_claims_no_retirement_without_retiring_anything() -> None:
    """The table's headline and its rows cannot disagree.

    Every row currently reads "No" in the *Retire now* column. If a future PR
    flips one to "Yes" it must also drop the headline sentence, and vice versa —
    an honest classification table is the deliverable, so a stale headline is a
    defect rather than a cosmetic issue.
    """
    text = _STAGE2_DOC.read_text(encoding="utf-8")
    start = text.index(_STAGE2_HEADING)
    end = text.index("\n### What has to happen", start)
    section = text[start:end]
    claims_none = "**Result: zero pure duplicates, zero retirements.**" in section
    rows = [line for line in section.splitlines() if line.startswith("| `")]
    assert rows, "the Stage 2 mapping table lost its rows"
    retires = [row for row in rows if re.search(r"\|\s*Yes\s*\|", row)]
    assert claims_none is (not retires)


def test_canonical_layer_still_refuses_to_import_the_application_layer() -> None:
    """No canonical module may import `rate_of_closure` (ADR-0048 containment).

    Individual canonical modules carry their own AST pin for this; the package
    is what the *mapping* depends on, because a single convenience re-export is
    all it takes to merge the colliding `TrendResult`, `LaunchMonitorProject`,
    `load_private_corpus` and `CONTRACT_VERSION` definitions by accident.
    """
    package = _REPO_ROOT / "src" / "shared" / "python" / "launch_monitor"
    offenders: list[str] = []
    for path in sorted(package.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            else:
                continue
            if any(name.split(".")[0] == "rate_of_closure" for name in names):
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, offenders
