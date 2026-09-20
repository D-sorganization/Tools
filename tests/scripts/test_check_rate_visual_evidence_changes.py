"""Fail-closed changed-path governance for Rate-of-Closure visual evidence."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.check_rate_visual_evidence_changes import (
    ACCEPTANCE_MANIFEST,
    PYQT_FIRST_VIEWPORT_TEST,
    REACT_FIRST_VIEWPORT_TEST,
    SHARED_AUDIT,
    SHARED_MANIFEST,
    extract_exemption_reason,
    main,
    validate_visual_evidence_changes,
)


def test_unrelated_change_requires_no_visual_evidence() -> None:
    assert validate_visual_evidence_changes(["README.md"]) == ()


def test_react_visual_change_requires_manifest_audit_and_first_viewport_test() -> None:
    errors = validate_visual_evidence_changes(
        ["src/rate_of_closure/web/src/components/SimulationDisplay.tsx"]
    )

    assert errors == (
        f"react visual changes require {SHARED_MANIFEST}",
        f"react visual changes require {ACCEPTANCE_MANIFEST}",
        f"react visual changes require {SHARED_AUDIT}",
        f"react visual changes require {REACT_FIRST_VIEWPORT_TEST}",
    )


def test_pyqt_visual_change_requires_manifest_audit_and_first_viewport_test() -> None:
    errors = validate_visual_evidence_changes(
        ["src/rate_of_closure/ui/pyqt6/variation_tab.py"]
    )

    assert errors == (
        f"pyqt visual changes require {SHARED_MANIFEST}",
        f"pyqt visual changes require {ACCEPTANCE_MANIFEST}",
        f"pyqt visual changes require {SHARED_AUDIT}",
        f"pyqt visual changes require {PYQT_FIRST_VIEWPORT_TEST}",
    )


def test_cross_surface_change_passes_with_complete_matched_evidence() -> None:
    changed = [
        "src/rate_of_closure/web/src/components/PrimaryViewTabs.tsx",
        "src/rate_of_closure/ui/pyqt6/main_window_layout.py",
        SHARED_MANIFEST,
        ACCEPTANCE_MANIFEST,
        SHARED_AUDIT,
        REACT_FIRST_VIEWPORT_TEST,
        PYQT_FIRST_VIEWPORT_TEST,
    ]

    assert validate_visual_evidence_changes(changed) == ()


def test_windows_paths_are_normalized_before_matching() -> None:
    errors = validate_visual_evidence_changes(
        [r"src\rate_of_closure\ui\pyqt6\plots_tab.py"]
    )

    assert errors[-1] == (f"pyqt visual changes require {PYQT_FIRST_VIEWPORT_TEST}")


def test_test_only_change_does_not_create_recursive_evidence_requirement() -> None:
    assert validate_visual_evidence_changes([REACT_FIRST_VIEWPORT_TEST]) == ()


def test_react_test_modules_do_not_claim_the_shipped_visual_surface_changed() -> None:
    for path in (
        "src/rate_of_closure/web/src/components/VariationPanel.test.tsx",
        "src/rate_of_closure/web/src/components/VariationPanel.spec.tsx",
    ):
        assert validate_visual_evidence_changes([path]) == ()


def test_extract_exemption_reason_from_trailers() -> None:
    assert (
        extract_exemption_reason("Rate-Visual-Exemption: perf refactor")
        == "perf refactor"
    )
    assert (
        extract_exemption_reason("Rate-Visual-Exempt: replaced spread with loop")
        == "replaced spread with loop"
    )
    assert (
        extract_exemption_reason("rate_visual_exempt: O(N) rewrite") == "O(N) rewrite"
    )
    assert (
        extract_exemption_reason("no-visual-change: algorithmic loop")
        == "algorithmic loop"
    )
    assert (
        extract_exemption_reason("rate-visual-exempt")
        == "declared no-visual-change marker"
    )
    assert extract_exemption_reason("Just a normal commit message") is None
    assert extract_exemption_reason("") is None


def test_exemption_reason_permits_unmanifested_visual_changes() -> None:
    errors = validate_visual_evidence_changes(
        ["src/rate_of_closure/web/src/components/SimulationDisplay.tsx"],
        exemption_reason="pure perf refactor in data processing loop",
    )
    assert errors == ()


def test_whitespace_exemption_reason_fails_closed() -> None:
    with pytest.raises(ValueError, match="exemption_reason must be nonempty"):
        validate_visual_evidence_changes(
            ["src/rate_of_closure/web/src/components/SimulationDisplay.tsx"],
            exemption_reason="   ",
        )


def test_cli_changed_file_fixture_fails_closed_on_incomplete_evidence(
    tmp_path: Path,
) -> None:
    changed = tmp_path / "changed.txt"
    changed.write_text(
        "src/rate_of_closure/web/src/components/SimulationDisplay.tsx\n",
        encoding="utf-8",
    )

    assert main(["--changed-files", str(changed)]) == 1


def test_cli_changed_file_fixture_accepts_complete_evidence(tmp_path: Path) -> None:
    changed = tmp_path / "changed.txt"
    changed.write_text(
        "\n".join(
            (
                "src/rate_of_closure/web/src/components/SimulationDisplay.tsx",
                SHARED_MANIFEST,
                ACCEPTANCE_MANIFEST,
                SHARED_AUDIT,
                REACT_FIRST_VIEWPORT_TEST,
            )
        ),
        encoding="utf-8",
    )

    assert main(["--changed-files", str(changed)]) == 0


def test_cli_missing_changed_file_fixture_returns_evaluation_error(
    tmp_path: Path,
) -> None:
    assert main(["--changed-files", str(tmp_path / "absent.txt")]) == 2


def test_cli_exemption_reason_flag_passes(tmp_path: Path) -> None:
    changed = tmp_path / "changed.txt"
    changed.write_text(
        "src/rate_of_closure/web/src/components/SimulationDisplay.tsx\n",
        encoding="utf-8",
    )

    assert (
        main(
            [
                "--changed-files",
                str(changed),
                "--exemption-reason",
                "perf refactor loop optimization",
            ]
        )
        == 0
    )


def test_cli_whitespace_exemption_reason_flag_fails_closed(tmp_path: Path) -> None:
    changed = tmp_path / "changed.txt"
    changed.write_text(
        "src/rate_of_closure/web/src/components/SimulationDisplay.tsx\n",
        encoding="utf-8",
    )

    assert (
        main(
            [
                "--changed-files",
                str(changed),
                "--exemption-reason",
                "   ",
            ]
        )
        == 2
    )


def test_cli_env_exemption_reason_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    changed = tmp_path / "changed.txt"
    changed.write_text(
        "src/rate_of_closure/web/src/components/SimulationDisplay.tsx\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("RATE_VISUAL_EXEMPTION", "perf refactor without visual change")

    assert main(["--changed-files", str(changed)]) == 0


def test_cli_pr_body_env_exemption_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    changed = tmp_path / "changed.txt"
    changed.write_text(
        "src/rate_of_closure/web/src/components/SimulationDisplay.tsx\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "PR_BODY",
        "## Summary\n\nRate-Visual-Exemption: replaced O(N^2) spread with O(N) loop\n",
    )

    assert main(["--changed-files", str(changed)]) == 0


def test_cli_git_commit_trailer_exemption_passes() -> None:
    with (
        patch(
            "scripts.check_rate_visual_evidence_changes._git_changed_files",
            return_value=(
                "src/rate_of_closure/web/src/components/SimulationDisplay.tsx",
            ),
        ),
        patch(
            "scripts.check_rate_visual_evidence_changes._git_commit_messages",
            return_value=(
                "feat: perf refactor\n\nRate-Visual-Exemption: linear min/max\n"
            ),
        ),
    ):
        assert main(["--base-ref", "origin/main"]) == 0
