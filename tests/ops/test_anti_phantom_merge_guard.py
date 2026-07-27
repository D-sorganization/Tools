"""Tests for scripts/anti_phantom_merge_guard.py (TDD — red phase first).

Verifies the anti-phantom-merge guard logic:
- Uses deterministic refs (pinned SHA from GitHub event, not dynamic resolution)
- Correctly counts changed files for known-good fixture data
- Rule 1: blocks empty-diff PRs (0 changed files)
- Rule 1 escape: allows "chore: empty PR" title
- Rule 2: blocks feature-claim titles with no src/ changes
- Rule 4: blocks bot-only commit histories
- Regression: non-zero file count for a known-good diff output

Race-condition fix: the guard must accept BASE_SHA and HEAD_SHA as
*explicit* parameters, never resolve them dynamically from an unstable
ref (e.g. origin/main which may still be propagating).
"""

from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent
from unittest.mock import patch

import pytest
import yaml

# Ensure scripts/ is importable
_SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"
_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_SCRIPTS_DIR))

import anti_phantom_merge_guard as guard  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_HEAD_SHA = "aaaa1111" * 5  # 40 chars
SAMPLE_BASE_SHA = "bbbb2222" * 5


def _make_diff(files: list[str]) -> str:
    """Return a fake ``git diff --name-only`` output for *files*."""
    return "\n".join(files)


def _workflow_uses_pull_request_target(workflow: dict[str, object]) -> bool:
    """Return whether a workflow is triggered by ``pull_request_target``."""
    triggers = workflow.get("on", {})
    if isinstance(triggers, str):
        return triggers == "pull_request_target"
    if isinstance(triggers, list):
        return "pull_request_target" in triggers
    if isinstance(triggers, dict):
        return "pull_request_target" in triggers
    return False


def _iter_workflow_steps(workflow: dict[str, object]) -> list[dict[str, object]]:
    """Return all step dictionaries from a GitHub Actions workflow."""
    jobs = workflow.get("jobs", {})
    if not isinstance(jobs, dict):
        return []
    steps: list[dict[str, object]] = []
    for job in jobs.values():
        if not isinstance(job, dict):
            continue
        job_steps = job.get("steps", [])
        if not isinstance(job_steps, list):
            continue
        steps.extend(step for step in job_steps if isinstance(step, dict))
    return steps


# ---------------------------------------------------------------------------
# count_changed_files
# ---------------------------------------------------------------------------


class TestCountChangedFiles:
    """count_changed_files must parse git diff output deterministically."""

    def test_returns_int(self) -> None:
        diff_output = _make_diff(["src/foo.py", "tests/test_foo.py"])
        n = guard.count_changed_files(diff_output)
        assert isinstance(n, int)

    def test_counts_two_files(self) -> None:
        diff_output = _make_diff(["src/foo.py", "tests/test_foo.py"])
        assert guard.count_changed_files(diff_output) == 2

    def test_counts_zero_for_empty_string(self) -> None:
        assert guard.count_changed_files("") == 0

    def test_counts_zero_for_whitespace_only(self) -> None:
        assert guard.count_changed_files("   \n  \n") == 0

    def test_ignores_blank_lines(self) -> None:
        diff_output = "src/a.py\n\nsrc/b.py\n\n"
        assert guard.count_changed_files(diff_output) == 2

    def test_single_file(self) -> None:
        assert guard.count_changed_files("src/single.py") == 1

    def test_known_good_fixture_nonzero(self) -> None:
        """Regression: a realistic diff must never return 0."""
        fixture = dedent("""\
            scripts/bump_vendor_pin.py
            tests/scripts/test_bump_vendor_pin.py
            docs/ops/vendor_pins.md
        """)
        assert guard.count_changed_files(fixture) == 3


# ---------------------------------------------------------------------------
# get_diff_via_git (mocked subprocess)
# ---------------------------------------------------------------------------


class TestGetDiffViaGit:
    """get_diff_via_git must call git with the pinned SHAs, not dynamic refs."""

    def test_calls_subprocess_with_shas(self) -> None:
        expected = "src/foo.py\n"
        with patch("subprocess.check_output", return_value=expected.encode()) as mock:
            guard.get_diff_via_git(
                base_sha=SAMPLE_BASE_SHA,
                head_sha=SAMPLE_HEAD_SHA,
            )
        # Must have called git with the exact SHAs across all subprocess calls
        all_args = [arg for call in mock.call_args_list for arg in call[0][0]]
        assert SAMPLE_BASE_SHA in all_args
        assert SAMPLE_HEAD_SHA in all_args

    def test_returns_string(self) -> None:
        with patch("subprocess.check_output", return_value=b"src/foo.py\n"):
            result = guard.get_diff_via_git(
                base_sha=SAMPLE_BASE_SHA,
                head_sha=SAMPLE_HEAD_SHA,
            )
        assert isinstance(result, str)

    def test_strips_trailing_whitespace(self) -> None:
        with patch("subprocess.check_output", return_value=b"src/foo.py\n  \n\n"):
            result = guard.get_diff_via_git(
                base_sha=SAMPLE_BASE_SHA,
                head_sha=SAMPLE_HEAD_SHA,
            )
        # The raw string is returned; count_changed_files handles filtering
        assert "src/foo.py" in result

    def test_rejects_empty_base_sha(self) -> None:
        with pytest.raises((ValueError, AssertionError)):
            guard.get_diff_via_git(base_sha="", head_sha=SAMPLE_HEAD_SHA)

    def test_rejects_empty_head_sha(self) -> None:
        with pytest.raises((ValueError, AssertionError)):
            guard.get_diff_via_git(base_sha=SAMPLE_BASE_SHA, head_sha="")


# ---------------------------------------------------------------------------
# Workflow hardening
# ---------------------------------------------------------------------------


class TestPullRequestTargetWorkflowHardening:
    """Privileged label workflows must never check out untrusted PR code."""

    def test_pull_request_target_workflows_guard_untrusted_head_checkout(self) -> None:
        failures: list[str] = []
        for path in sorted((_REPO_ROOT / ".github" / "workflows").glob("*.yml")):
            # BaseLoader preserves GitHub Actions keys like "on" as strings.
            workflow = yaml.load(  # nosec B506
                path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader
            )
            if not isinstance(workflow, dict):
                continue
            if not _workflow_uses_pull_request_target(workflow):
                continue
            for step in _iter_workflow_steps(workflow):
                uses = str(step.get("uses", ""))
                if not uses.startswith("actions/checkout"):
                    continue
                step_with = step.get("with", {})
                if not isinstance(step_with, dict):
                    continue
                checkout_ref = str(step_with.get("ref", ""))
                if "github.event.pull_request.head.sha" not in checkout_ref:
                    continue
                condition = str(step.get("if", ""))
                if "github.event_name == 'pull_request'" not in condition:
                    failures.append(f"{path}: {step.get('name', 'checkout')}")

        assert failures == []

    def test_anti_phantom_documents_privileged_checkout_invariant(self) -> None:
        workflow_text = (
            _REPO_ROOT / ".github" / "workflows" / "anti-phantom-merge.yml"
        ).read_text(encoding="utf-8")

        assert "Never check out PR head code on pull_request_target" in workflow_text

    def test_anti_phantom_checkout_is_bounded_and_api_fallback_keeps_paths(
        self,
    ) -> None:
        workflow_path = _REPO_ROOT / ".github" / "workflows" / "anti-phantom-merge.yml"
        workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
        guard_job = workflow["jobs"]["guard"]
        checkout = next(
            step
            for step in guard_job["steps"]
            if step.get("name") == "Check out PR head with bounded history"
        )
        guard_script = next(
            step
            for step in guard_job["steps"]
            if step.get("name") == "Run phantom guard checks"
        )["run"]

        assert checkout["with"]["fetch-depth"] == 50
        assert "gh api --paginate" in guard_script
        assert "/pulls/$PR_NUMBER/files" in guard_script
        assert 'CHANGED_FILES="$API_CHANGED_FILES"' in guard_script


# ---------------------------------------------------------------------------
# Rule 1 — empty diff
# ---------------------------------------------------------------------------


class TestRule1EmptyDiff:
    """Rule 1 must block PRs with 0 changed files."""

    def test_raises_on_zero_changed_files(self) -> None:
        result = guard.check_rule_1(
            num_changed=0,
            pr_title="feat: add awesome feature",
        )
        assert result is not None  # error message returned
        assert "0 changed" in result.lower() or "empty" in result.lower()

    def test_passes_on_nonzero_changed_files(self) -> None:
        result = guard.check_rule_1(
            num_changed=3,
            pr_title="feat: add awesome feature",
        )
        assert result is None  # no error

    def test_escape_hatch_allows_empty_pr_with_correct_title(self) -> None:
        result = guard.check_rule_1(
            num_changed=0,
            pr_title="chore: empty PR",
        )
        assert result is None  # escape hatch honored


# ---------------------------------------------------------------------------
# Rule 2 — feature claim vs no implementation
# ---------------------------------------------------------------------------


class TestRule2FeatureClaim:
    """Rule 2 must block feat: titles with no src/ changes."""

    def test_blocks_feat_title_with_no_src_changes(self) -> None:
        diff_output = _make_diff(["docs/README.md", "tests/test_foo.py"])
        result = guard.check_rule_2(
            pr_title="feat: add awesome feature",
            changed_files_output=diff_output,
            num_changed=2,
        )
        assert result is not None

    def test_passes_feat_title_with_src_changes(self) -> None:
        diff_output = _make_diff(["src/foo.py", "tests/test_foo.py"])
        result = guard.check_rule_2(
            pr_title="feat: add awesome feature",
            changed_files_output=diff_output,
            num_changed=2,
        )
        assert result is None

    def test_passes_chore_title_with_no_src_changes(self) -> None:
        diff_output = _make_diff(["docs/README.md"])
        result = guard.check_rule_2(
            pr_title="chore: update readme",
            changed_files_output=diff_output,
            num_changed=1,
        )
        assert result is None

    def test_blocks_implement_in_title(self) -> None:
        diff_output = _make_diff(["docs/design.md"])
        result = guard.check_rule_2(
            pr_title="Implement new Engine subsystem",
            changed_files_output=diff_output,
            num_changed=1,
        )
        assert result is not None

    def test_skips_rule_when_num_changed_is_zero(self) -> None:
        """Rule 2 must not double-fire when Rule 1 already caught empty diff."""
        result = guard.check_rule_2(
            pr_title="feat: add something",
            changed_files_output="",
            num_changed=0,
        )
        assert result is None


# ---------------------------------------------------------------------------
# Rule 4 — bot-only commits
# ---------------------------------------------------------------------------


class TestRule4BotOnlyCommits:
    """Rule 4 must block PRs where only bot merge commits exist."""

    def test_blocks_bot_only_history(self) -> None:
        commits = [
            "aaa|github-actions[bot]|Merge branch 'main' into feature",
            "bbb|github-actions[bot]|Merge branch 'main'",
        ]
        result = guard.check_rule_4(commits=commits)
        assert result is not None

    def test_passes_when_author_commit_exists(self) -> None:
        commits = [
            "aaa|github-actions[bot]|Merge branch 'main'",
            "bbb|alice|feat: implement foo",
        ]
        result = guard.check_rule_4(commits=commits)
        assert result is None

    def test_passes_empty_commit_list(self) -> None:
        result = guard.check_rule_4(commits=[])
        assert result is None

    def test_passes_single_human_commit(self) -> None:
        commits = ["aaa|bob|fix: correct typo"]
        result = guard.check_rule_4(commits=commits)
        assert result is None


# ---------------------------------------------------------------------------
# run_all_checks — integration
# ---------------------------------------------------------------------------


class TestRunAllChecks:
    """run_all_checks returns a list of failure messages (empty == all passed)."""

    def test_clean_pr_returns_empty_failures(self) -> None:
        failures = guard.run_all_checks(
            base_sha=SAMPLE_BASE_SHA,
            head_sha=SAMPLE_HEAD_SHA,
            pr_title="feat: add new calculator",
            pr_body="Closes #123",
            diff_output=_make_diff(["src/calculators/new.py", "tests/test_new.py"]),
            commits=["aaa|alice|feat: add new calculator"],
        )
        assert failures == []

    def test_empty_diff_returns_failure(self) -> None:
        failures = guard.run_all_checks(
            base_sha=SAMPLE_BASE_SHA,
            head_sha=SAMPLE_HEAD_SHA,
            pr_title="feat: add new calculator",
            pr_body="",
            diff_output="",
            commits=["aaa|alice|feat: add new calculator"],
        )
        assert len(failures) >= 1

    def test_returns_list(self) -> None:
        failures = guard.run_all_checks(
            base_sha=SAMPLE_BASE_SHA,
            head_sha=SAMPLE_HEAD_SHA,
            pr_title="chore: update readme",
            pr_body="",
            diff_output=_make_diff(["README.md"]),
            commits=["aaa|alice|chore: update readme"],
        )
        assert isinstance(failures, list)
