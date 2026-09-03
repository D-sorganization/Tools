#!/usr/bin/env python3
"""Prove the ``ud-tools`` wheel builds and carries pyproject's name/version.

Tools #4920: UpstreamDrift consumes Tools as a pip wheel, so a release must
never ship a wheel whose metadata disagrees with ``pyproject.toml`` (or fail to
build at all). ``--check`` builds the wheel into a temporary directory with
``python -m build --wheel`` and asserts the filename is exactly
``<name>-<version>-py3-none-any.whl``. Without ``--check`` it builds into
``--outdir`` (default ``dist/``) and prints the wheel path, which the release
workflow uses to attach the artifact.

Usage::

    python scripts/check_wheel_build.py --check
    python scripts/check_wheel_build.py --outdir dist
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def project_metadata(pyproject: Path = REPO_ROOT / "pyproject.toml") -> tuple[str, str]:
    with pyproject.open("rb") as handle:
        project = tomllib.load(handle)["project"]
    return str(project["name"]), str(project["version"])


def expected_wheel_name(name: str, version: str) -> str:
    # PEP 427 normalises '-' to '_' in the distribution name.
    return f"{name.replace('-', '_')}-{version}-py3-none-any.whl"


def build_wheel(outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(outdir)],
        cwd=REPO_ROOT,
        check=True,
    )
    wheels = sorted(outdir.glob("*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"expected exactly one wheel in {outdir}, found {wheels}")
    return wheels[0]


def verify(wheel: Path, name: str, version: str) -> list[str]:
    problems: list[str] = []
    expected = expected_wheel_name(name, version)
    if wheel.name != expected:
        problems.append(f"wheel is {wheel.name}, pyproject says {expected}")
    if wheel.stat().st_size < 1024:
        problems.append(f"wheel is implausibly small ({wheel.stat().st_size} bytes)")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--check", action="store_true", help="build into a temp dir and only verify"
    )
    parser.add_argument(
        "--outdir", default="dist", help="where to leave the wheel (default dist/)"
    )
    args = parser.parse_args(argv)

    name, version = project_metadata()
    if args.check:
        with tempfile.TemporaryDirectory(prefix="ud-tools-wheel-") as tmp:
            wheel = build_wheel(Path(tmp))
            problems = verify(wheel, name, version)
            if problems:
                for problem in problems:
                    print(f"ERROR: {problem}", file=sys.stderr)
                return 1
            print(f"OK: built {wheel.name} for {name} {version}")
            return 0
    wheel = build_wheel(Path(args.outdir))
    problems = verify(wheel, name, version)
    for problem in problems:
        print(f"ERROR: {problem}", file=sys.stderr)
    print(wheel.as_posix())
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
