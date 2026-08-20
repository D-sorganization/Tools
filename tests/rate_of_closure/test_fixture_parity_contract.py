"""Cross-runtime fixture symmetry guard.

Ensures that every shared fixture in `src/rate_of_closure/web/src/model/__fixtures__/`
is consumed symmetrically by both Python and TypeScript runtimes, or explicitly
allowlisted with a documented rationale and issue reference (issue #4559).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.contract, pytest.mark.headless_safe]

_REPO_ROOT = Path(__file__).parents[2].resolve()
_THIS_FILE = Path(__file__).resolve()

FIXTURES_DIR = (
    _REPO_ROOT
    / "src"
    / "rate_of_closure"
    / "web"
    / "src"
    / "model"
    / "__fixtures__"
).resolve()

_SEARCH_ROOTS_PY = (
    _REPO_ROOT / "src" / "rate_of_closure",
    _REPO_ROOT / "src" / "shared",
    _REPO_ROOT / "tests" / "rate_of_closure",
    _REPO_ROOT / "tests" / "shared",
    _REPO_ROOT / "rust_core",
)

_SEARCH_ROOTS_TS = (
    _REPO_ROOT / "src" / "rate_of_closure" / "web" / "src",
)

_SEARCH_ROOTS_RS = (
    _REPO_ROOT / "rust_core",
)

_EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    "dist",
    "build",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "target",
    ".tox",
    ".idea",
    ".vscode",
    "replicants",
    "archive",
    "legacy",
    "experimental",
    "htmlcov",
}

# Explicit allowlist of fixtures that currently have fewer than two consuming
# runtimes, each documented with a reason and tracking issue link.
# Stale entries must be cleaned up as companion issues land (e.g. #4558, #4560).
SINGLE_RUNTIME_ALLOWLIST: dict[str, str] = {
    # Python-only fixtures (tracked in #4560, #4558)
    "ground_impact_bounce_golden_v1.json": (
        "Python-only ground impact/bounce golden fixture; "
        "TypeScript consumer tracked in #4560"
    ),
    "ground_reference_conformance_v1.json": (
        "Consumed by Python and Rust core parity suites; "
        "TypeScript consumer tracked in #4560"
    ),
    "ground_skid_roll_golden_v1.json": (
        "Landed as standalone Python slice from #4517; "
        "TypeScript consumer tracked in #4560"
    ),
    "variation_execution_document_edge_floats_v1.json": (
        "Python-only execution metadata fixture ported in #4529; "
        "TypeScript consumer on #4447 tracked in #4558 and #4560"
    ),
    "variation_execution_document_python_v2.json": (
        "Python-only execution metadata fixture ported in #4529; "
        "TypeScript consumer on #4447 tracked in #4558 and #4560"
    ),
    "variation_execution_document_react_v2.json": (
        "Python-only execution metadata fixture ported in #4529; "
        "TypeScript consumer on #4447 tracked in #4558 and #4560"
    ),
    "variation_execution_document_v1.json": (
        "Python-only execution metadata fixture ported in #4529; "
        "TypeScript consumer on #4447 tracked in #4558 and #4560"
    ),
    # TypeScript-only fixtures (tracked in #4560)
    "regional_ground_scalar_ensemble_golden_v1.json": (
        "TypeScript-only regional ground ensemble fixture; "
        "Python consumer tracked in #4560"
    ),
    "runtime_manifest_parity_v1.json": (
        "TypeScript-only runtime manifest parity fixture; "
        "Python consumer tracked in #4560"
    ),
    "torque_profile_parity.json": (
        "TypeScript-only torque profile parity fixture; "
        "Python consumer tracked in #4560"
    ),
}


class CodeCorpus(NamedTuple):
    py_files: dict[Path, str]
    ts_files: dict[Path, str]
    rs_files: dict[Path, str]


def _scan_roots(
    roots: tuple[Path, ...], extensions: tuple[str, ...]
) -> dict[Path, str]:
    corpus: dict[Path, str] = {}
    this_file_str = str(_THIS_FILE)
    for root_dir in roots:
        if not root_dir.is_dir():
            continue
        for root, dirs, files in os.walk(root_dir):
            dirs[:] = [
                d for d in dirs if d not in _EXCLUDED_DIRS and not d.startswith(".")
            ]
            for f in files:
                if any(f.endswith(ext) for ext in extensions):
                    full_path = os.path.join(root, f)
                    if full_path == this_file_str:
                        continue
                    try:
                        with open(full_path, encoding="utf-8", errors="ignore") as fp:
                            corpus[Path(full_path)] = fp.read()
                    except Exception:
                        pass
    return corpus


@pytest.fixture(scope="module")
def corpus() -> CodeCorpus:
    """Index relevant source and test files across the repository."""
    return CodeCorpus(
        py_files=_scan_roots(_SEARCH_ROOTS_PY, (".py",)),
        ts_files=_scan_roots(_SEARCH_ROOTS_TS, (".ts", ".tsx")),
        rs_files=_scan_roots(_SEARCH_ROOTS_RS, (".rs",)),
    )


def _has_reference(fixture: Path, files: dict[Path, str]) -> bool:
    name = fixture.name
    stem = fixture.stem
    return any(name in text or stem in text for text in files.values())


def test_allowlist_is_not_vacuous() -> None:
    """Assert every allowlisted fixture exists on disk and has a valid rationale."""
    assert FIXTURES_DIR.is_dir(), f"Fixtures directory not found: {FIXTURES_DIR}"
    existing_names = {p.name for p in FIXTURES_DIR.glob("*.json")}

    for fixture_name, reason in SINGLE_RUNTIME_ALLOWLIST.items():
        assert fixture_name in existing_names, (
            f"Allowlisted fixture {fixture_name!r} does not exist in {FIXTURES_DIR}. "
            "Remove stale entries from SINGLE_RUNTIME_ALLOWLIST."
        )
        assert (
            reason.strip()
        ), f"Allowlist entry {fixture_name!r} must have a non-empty rationale"
        assert "#" in reason, (
            f"Allowlist entry {fixture_name!r} must reference a tracking issue or "
            "PR (e.g. #4560)"
        )


def test_every_shared_fixture_has_a_consumer_in_both_runtimes(
    corpus: CodeCorpus,
) -> None:
    """Assert shared fixtures are consumed by Python and TypeScript runtimes.

    Excludes fixtures explicitly allowlisted with a documented reason.
    """
    fixtures = sorted(FIXTURES_DIR.glob("*.json"))
    assert len(fixtures) > 0, f"No fixtures found in {FIXTURES_DIR}"

    failures: list[str] = []
    for fixture in fixtures:
        if fixture.name in SINGLE_RUNTIME_ALLOWLIST:
            continue

        has_py = _has_reference(fixture, corpus.py_files)
        has_ts = _has_reference(fixture, corpus.ts_files)

        missing: list[str] = []
        if not has_py:
            missing.append("Python")
        if not has_ts:
            missing.append("TypeScript")

        if missing:
            failures.append(
                f"Fixture {fixture.name} missing consumer in: {', '.join(missing)}"
            )

    msg = (
        f"Found {len(failures)} shared fixtures lacking cross-runtime consumers:\n"
        + "\n".join(failures)
    )
    assert not failures, msg


def test_no_orphaned_fixtures(corpus: CodeCorpus) -> None:
    """Assert that all shared fixtures have at least one consumer."""
    fixtures = sorted(FIXTURES_DIR.glob("*.json"))
    assert len(fixtures) > 0, f"No fixtures found in {FIXTURES_DIR}"

    orphans: list[str] = []
    for fixture in fixtures:
        has_py = _has_reference(fixture, corpus.py_files)
        has_ts = _has_reference(fixture, corpus.ts_files)
        has_rs = _has_reference(fixture, corpus.rs_files)

        if not (has_py or has_ts or has_rs):
            orphans.append(fixture.name)

    assert not orphans, f"Found orphaned fixtures with zero consumers: {orphans}"
