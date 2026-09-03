"""Fail closed when the trusted Rate PyQt binary stack is inconsistent."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import subprocess
from importlib.metadata import version as metadata_version
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

LOGGER = logging.getLogger(__name__)
REQUIRED_DISTRIBUTIONS = {
    "numpy": "numpy",
    "scipy": "scipy",
    "pyqt6": "PyQt6",
    "matplotlib": "matplotlib",
}

# The system font stack the trusted PyQt renders depend on (issue #4844: a
# host libfreetype6/libfontconfig1 upgrade re-rasterized every glyph and
# surfaced as opaque pixel drift 4-13x the calibrated envelope). Probed via
# dpkg-query on the Ubuntu runner host; unavailable elsewhere.
_FONT_STACK_PACKAGES = {
    "libfreetype6": "libfreetype6",
    "libfontconfig1": "libfontconfig1",
}
DEFAULT_FONT_STACK_EXPECTATIONS = (
    Path(__file__).resolve().parent / "config" / "rate_pyqt_font_stack.v1.json"
)


def read_expected_versions(constraints_path: Path) -> dict[str, str]:
    """Read exact binary-stack pins from a pip constraints file.

    Args:
        constraints_path: Existing UTF-8 pip constraints file.

    Returns:
        Canonical distribution names mapped to exact expected versions.

    Raises:
        ValueError: If a required distribution is missing, duplicated, or not
            pinned to one concrete version.
    """
    expected: dict[str, str] = {}
    for raw_line in constraints_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        requirement = Requirement(line)
        name = canonicalize_name(requirement.name)
        if name not in REQUIRED_DISTRIBUTIONS:
            continue
        if name in expected:
            raise ValueError(f"duplicate required constraint: {requirement.name}")
        specifiers = list(requirement.specifier)
        if (
            len(specifiers) != 1
            or specifiers[0].operator != "=="
            or "*" in specifiers[0].version
        ):
            raise ValueError(f"{requirement.name} must be exactly pinned")
        expected[name] = specifiers[0].version

    missing = sorted(set(REQUIRED_DISTRIBUTIONS) - set(expected))
    if missing:
        raise ValueError(f"missing required constraints: {', '.join(missing)}")
    return expected


def _import_runtime() -> None:
    """Import the ABI-sensitive modules used by the rendered Rate window."""
    for module_name in (
        "numpy",
        "scipy.integrate",
        "scipy.sparse.linalg",
        "PyQt6.QtWidgets",
    ):
        importlib.import_module(module_name)


def _MATPLOTLIB_FREETYPE_VERSION() -> str:
    """Return matplotlib's compiled freetype version, or ``unknown``."""

    from matplotlib import ft2font

    return str(getattr(ft2font, "__freetype_version__", "unknown"))


def probe_font_stack() -> dict[str, str]:
    """Probe the system font stack the trusted PyQt renders depend on.

    Returns:
        Probed font-stack identifiers: matplotlib's compiled freetype
        version plus the host package versions from
        ``_FONT_STACK_PACKAGES`` (``unavailable`` when not probeable, e.g.
        non-Ubuntu development hosts).
    """

    stack: dict[str, str] = {"matplotlib_freetype": _MATPLOTLIB_FREETYPE_VERSION()}
    for key, package in _FONT_STACK_PACKAGES.items():
        try:
            result = subprocess.run(
                ["dpkg-query", "-W", "-f=${Version}", package],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            stack[key] = "unavailable"
            continue
        stack[key] = result.stdout.strip() if result.returncode == 0 else "unavailable"
    return stack


def verify_font_stack(expectations_path: Path) -> dict[str, str]:
    """Fail with a named cause when the system font stack changed.

    Args:
        expectations_path: Committed JSON mapping font-stack identifiers
            to the versions the approved baselines were captured under.

    Returns:
        The probed font stack when it matches the recorded expectations.

    Raises:
        RuntimeError: Naming every identifier whose live probe differs
            from the recorded expectations — a host font upgrade is an
            environment change (issue #4844), not opaque pixel drift.
    """

    expectations = json.loads(expectations_path.read_text(encoding="utf-8"))
    probed = probe_font_stack()
    mismatches = [
        f"{key} {probed.get(key, 'unavailable')} != expected {expected}"
        for key, expected in expectations.items()
        if probed.get(key) != expected
    ]
    if mismatches:
        raise RuntimeError(
            "system font stack changed (issue #4844): " + "; ".join(mismatches)
        )
    return probed


def verify_runtime(constraints_path: Path) -> dict[str, str]:
    """Verify exact installed versions, then import the binary runtime."""
    expected = read_expected_versions(constraints_path)
    installed = {
        name: metadata_version(distribution)
        for name, distribution in REQUIRED_DISTRIBUTIONS.items()
    }
    mismatches = [
        f"{name} {installed[name]} != constrained {expected[name]}"
        for name in REQUIRED_DISTRIBUTIONS
        if installed[name] != expected[name]
    ]
    if mismatches:
        raise RuntimeError("; ".join(mismatches))
    _import_runtime()
    LOGGER.info("Verified isolated Rate PyQt runtime: %s", installed)
    return installed


def main() -> int:
    """Run the trusted Rate PyQt environment check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--constraints", required=True, type=Path)
    parser.add_argument("--font-stack", type=Path, default=None)
    parser.add_argument(
        "--print-font-stack",
        action="store_true",
        help="print the probed font stack as JSON and exit (for capture)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.print_font_stack:
        print(json.dumps(probe_font_stack(), indent=1, sort_keys=True))
        return 0
    verify_runtime(args.constraints)
    expectations = args.font_stack or DEFAULT_FONT_STACK_EXPECTATIONS
    if expectations.is_file():
        stack = verify_font_stack(expectations)
        LOGGER.info("Verified system font stack: %s", stack)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
