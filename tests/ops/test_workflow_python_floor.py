"""No workflow job may run Python below the floor of the code it exercises.

The root distribution declares ``requires-python = ">=3.11"`` and the root
``conftest.py`` refuses to collect code whose declared floor is above the running
interpreter. A workflow leg on an older interpreter therefore collects zero tests
and pytest exits 5 (issue #5434).

Some jobs legitimately run an older interpreter because they build and test a
sub-package or crate that declares its own, lower ``requires-python`` in its own
``pyproject.toml``. A job's effective floor is the highest floor among the paths it
exercises (pytest targets, ``maturin -m`` manifests, ``working-directory``), each
resolved to its nearest ``pyproject.toml`` exactly as the conftest does. A job that
exercises nothing but root code, or that names no path at all, is held to the root
floor.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
Version = tuple[int, int]

_FLOOR_RE = re.compile(
    r"""^\s*requires-python\s*=\s*["']\s*>=\s*(\d+)\.(\d+)\s*["']""", re.M
)
_REQUIRES_RE = re.compile(r"""^\s*requires-python\s*=""", re.M)
_MANIFEST_RE = re.compile(r"(?:^|\s)-m\s+(\S+Cargo\.toml)")


def parse_floor(pyproject: Path) -> Version:
    """Return the ``>=X.Y`` floor declared by ``pyproject``.

    Precondition: ``requires-python`` is a single ``>=X.Y`` specifier.
    Postcondition: raises ``AssertionError`` naming the file otherwise.
    """
    text = pyproject.read_text(encoding="utf-8")
    assert _REQUIRES_RE.search(text), f"{pyproject} declares no requires-python"
    match = _FLOOR_RE.search(text)
    assert match is not None, (
        f"{pyproject} requires-python has an unsupported shape; "
        "this policy only understands a single '>=X.Y' specifier"
    )
    return int(match.group(1)), int(match.group(2))


def parse_version(raw: object, where: str) -> Version:
    """Parse a matrix/literal version such as ``"3.11"`` into ``(3, 11)``."""
    assert isinstance(raw, str), (
        f"{where}: python-version {raw!r} must be a quoted string "
        "(YAML reads an unquoted 3.10 as the float 3.1)"
    )
    match = re.fullmatch(r"(\d+)\.(\d+)(?:\.\d+)?", raw.strip())
    assert match is not None, f"{where}: unsupported python-version {raw!r}"
    return int(match.group(1)), int(match.group(2))


def nearest_floor(path: Path) -> Version:
    """Floor of the nearest ``pyproject.toml`` at or above ``path`` in the repo."""
    directory = path if path.is_dir() else path.parent
    for parent in (directory, *directory.parents):
        pyproject = parent / "pyproject.toml"
        if pyproject.is_file():
            return parse_floor(pyproject)
        if parent == REPO_ROOT:
            break
    return parse_floor(REPO_ROOT / "pyproject.toml")


def job_versions(job: dict, where: str) -> list[tuple[str, Version]]:
    """Literal interpreter versions a job runs, as ``(origin, version)`` pairs."""
    found: list[tuple[str, Version]] = []
    matrix = (job.get("strategy") or {}).get("matrix") or {}
    if isinstance(matrix, dict):
        entries = list(matrix.get("python-version") or [])
        for include in matrix.get("include") or []:
            if isinstance(include, dict) and "python-version" in include:
                entries.append(include["python-version"])
        for raw in entries:
            found.append(("matrix", parse_version(raw, where)))
    for step in job.get("steps") or []:
        if not str(step.get("uses", "")).startswith("actions/setup-python"):
            continue
        raw = (step.get("with") or {}).get("python-version")
        if isinstance(raw, str) and "${{" not in raw:
            found.append(("setup-python", parse_version(raw, where)))
    return found


def exercised_paths(job: dict) -> Iterator[Path]:
    """Existing repo paths a job tests or builds."""
    candidates: list[str] = []
    for step in job.get("steps") or []:
        if step.get("working-directory"):
            candidates.append(step["working-directory"])
        script = str(step.get("run", "")).replace("\\\n", " ")
        for line in script.splitlines():
            if "pytest" in line:
                after = line.split("pytest", 1)[1].split()
                candidates += [t for t in after if "/" in t and not t.startswith("-")]
            candidates += _MANIFEST_RE.findall(line)
    for raw in candidates:
        path = REPO_ROOT / raw.strip("'\"")
        if path.name == "Cargo.toml":
            path = path.parent
        if path.exists():
            yield path


def effective_floor(job: dict) -> Version:
    """Highest floor among the paths the job exercises (root floor by default)."""
    floors = [nearest_floor(path) for path in exercised_paths(job)]
    return max(floors) if floors else parse_floor(REPO_ROOT / "pyproject.toml")


def violations() -> list[str]:
    """Every ``workflow:job`` leg whose interpreter is below its effective floor."""
    problems: list[str] = []
    for workflow in sorted(WORKFLOWS.glob("*.yml")):
        data = yaml.safe_load(workflow.read_text(encoding="utf-8")) or {}
        for name, job in (data.get("jobs") or {}).items():
            if not isinstance(job, dict):
                continue
            where = f"{workflow.name}:{name}"
            floor = effective_floor(job)
            for origin, version in job_versions(job, where):
                if version < floor:
                    problems.append(
                        f"{where} runs Python {version[0]}.{version[1]} ({origin}) "
                        f"below its floor {floor[0]}.{floor[1]}"
                    )
    return problems


def test_no_workflow_leg_runs_below_the_floor_of_the_code_it_exercises() -> None:
    problems = violations()
    assert not problems, "\n".join(problems)


def test_floor_parser_rejects_unsupported_specifiers(tmp_path: Path) -> None:
    for spec in ('">=3.10,<4"', '"~=3.11"'):
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(f"[project]\nrequires-python = {spec}\n", encoding="utf-8")
        try:
            parse_floor(pyproject)
        except AssertionError as error:
            assert "unsupported shape" in str(error)
        else:
            raise AssertionError(f"{spec} should have been rejected")


def test_unquoted_matrix_version_is_rejected() -> None:
    try:
        parse_version(3.1, "wf:job")
    except AssertionError as error:
        assert "quoted string" in str(error)
    else:
        raise AssertionError("float version should have been rejected")
