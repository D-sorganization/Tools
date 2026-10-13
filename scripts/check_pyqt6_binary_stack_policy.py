"""Enforce coupled PyQt6 / PyQt6-Qt6 / PyQt6-sip pins (#5508).

PyQt6-sip is the SIP ABI wheel PyQt6 binds against. Dependabot #5486 tried to
bump sip to 13.13.0 without a coordinated PyQt6 upgrade; that deferral is recorded
in ``config/pyqt6_binary_stack_policy.v1.json`` and must stay aligned across the
CI lock files listed there.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from scripts.check_rate_pyqt_environment import read_expected_versions

LOGGER = logging.getLogger(__name__)

DEFAULT_POLICY = Path("config") / "pyqt6_binary_stack_policy.v1.json"
_PYQT_DISTRIBUTIONS = ("pyqt6", "pyqt6-qt6", "pyqt6-sip")
_NAME_BY_CANONICAL = {
    "pyqt6": "PyQt6",
    "pyqt6-qt6": "PyQt6-Qt6",
    "pyqt6-sip": "PyQt6-sip",
}


class PolicyConfigError(Exception):
    """Policy JSON is missing or unusable (fail closed)."""


def load_policy(path: Path) -> dict[str, object]:
    """Load and minimally validate the PyQt6 binary-stack policy."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PolicyConfigError(f"cannot read {path}: {exc!r}") from exc
    if not isinstance(data, dict):
        raise PolicyConfigError(f"{path}: policy must be a JSON object")
    coupled = data.get("coupled_pins")
    lock_files = data.get("ci_lock_files")
    if not isinstance(coupled, dict) or not coupled:
        raise PolicyConfigError(f"{path}: coupled_pins must be a non-empty object")
    if not isinstance(lock_files, list) or not lock_files:
        raise PolicyConfigError(f"{path}: ci_lock_files must be a non-empty list")
    for key in _PYQT_DISTRIBUTIONS:
        display = _NAME_BY_CANONICAL[key]
        value = coupled.get(display)
        if not isinstance(value, str) or not value.strip():
            raise PolicyConfigError(f"{path}: coupled_pins missing {display}")
    for rel in lock_files:
        if not isinstance(rel, str) or not rel.strip():
            raise PolicyConfigError(f"{path}: ci_lock_files entries must be strings")
    blocked = data.get("blocked_standalone_sip_versions", [])
    if blocked is not None and not (
        isinstance(blocked, list)
        and all(isinstance(item, str) for item in blocked)
    ):
        raise PolicyConfigError(
            f"{path}: blocked_standalone_sip_versions must be a list of strings"
        )
    return data


def _expected_pyqt_pins(policy: dict[str, object]) -> dict[str, str]:
    coupled = policy["coupled_pins"]
    assert isinstance(coupled, dict)
    return {
        key: str(coupled[_NAME_BY_CANONICAL[key]]) for key in _PYQT_DISTRIBUTIONS
    }


def collect_violations(
    root: Path,
    *,
    policy_path: Path | None = None,
) -> list[str]:
    """Return human-readable violations; empty when the repo matches policy."""
    path = policy_path or (root / DEFAULT_POLICY)
    policy = load_policy(path)
    expected = _expected_pyqt_pins(policy)
    violations: list[str] = []

    lock_files = policy["ci_lock_files"]
    assert isinstance(lock_files, list)
    for rel in lock_files:
        lock_path = root / str(rel)
        if not lock_path.is_file():
            violations.append(f"{rel}: missing lock file")
            continue
        try:
            pinned = read_expected_versions(lock_path)
        except ValueError as exc:
            violations.append(f"{rel}: {exc}")
            continue
        for key in _PYQT_DISTRIBUTIONS:
            actual = pinned.get(key)
            if actual != expected[key]:
                violations.append(
                    f"{rel}: {_NAME_BY_CANONICAL[key]} {actual} != policy {expected[key]}"
                )

    blocked = policy.get("blocked_standalone_sip_versions") or []
    assert isinstance(blocked, list)
    sip_pin = expected["pyqt6-sip"]
    if sip_pin in blocked:
        violations.append(
            f"policy couples PyQt6-sip to {sip_pin}, which is listed as blocked"
        )

    for rel in lock_files:
        lock_path = root / str(rel)
        if not lock_path.is_file():
            continue
        for raw_line in lock_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            requirement = Requirement(line)
            if canonicalize_name(requirement.name) != "pyqt6-sip":
                continue
            for spec in requirement.specifier:
                if spec.version in blocked:
                    violations.append(
                        f"{rel}: PyQt6-sip must stay at {expected['pyqt6-sip']} "
                        f"(blocked standalone {spec.version}, issue #5508)"
                    )
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="repository root (default: current directory)",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=None,
        help="override policy JSON path",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    root = args.repo_root.resolve()
    try:
        violations = collect_violations(root, policy_path=args.policy)
    except PolicyConfigError as exc:
        LOGGER.error("%s", exc)
        return 1
    if violations:
        for item in violations:
            LOGGER.error("%s", item)
        return 1
    LOGGER.info("PyQt6 binary stack matches policy (#5508)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
