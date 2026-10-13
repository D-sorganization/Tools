"""PyQt6 / PyQt6-sip must stay a coupled CI lock (#5508, Dependabot #5486)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import check_pyqt6_binary_stack_policy as policy_check

ROOT = Path(__file__).resolve().parents[2]


def test_policy_pins_match_repository_lock_files() -> None:
    violations = policy_check.collect_violations(ROOT)
    assert violations == []


def test_isolated_sip_bump_is_rejected(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.json"
    lock_path = tmp_path / "lock.txt"
    policy_path.write_text(
        json.dumps(
            {
                "ci_lock_files": ["lock.txt"],
                "coupled_pins": {
                    "PyQt6": "6.11.0",
                    "PyQt6-Qt6": "6.11.2",
                    "PyQt6-sip": "13.12.0",
                },
            }
        ),
        encoding="utf-8",
    )
    lock_path.write_text(
        "\n".join(
            [
                "numpy==2.3.5",
                "scipy==1.17.1",
                "matplotlib==3.11.2",
                "PyQt6==6.11.0",
                "PyQt6-Qt6==6.11.2",
                "PyQt6-sip==13.13.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    violations = policy_check.collect_violations(tmp_path, policy_path=policy_path)
    assert any("PyQt6-sip" in item and "13.13.0" in item for item in violations)


def test_dependabot_ignores_deferred_sip_release() -> None:
    body = (ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8")
    assert "pyqt6-sip" in body.lower()
    assert "13.13.0" in body


def test_malformed_policy_fails_closed(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.json"
    policy_path.write_text("{}", encoding="utf-8")
    with pytest.raises(policy_check.PolicyConfigError, match="coupled_pins"):
        policy_check.load_policy(policy_path)
