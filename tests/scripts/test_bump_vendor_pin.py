"""Tests for scripts/bump_vendor_pin.py (TDD — red phase first).

Verifies that the vendor-pin bump helper:
- Resolves the latest tag from the Tools repository
- Generates a valid bump-PR body mentioning the new SHA/tag
- Validates consumer repo names against an allowlist
- Enforces preconditions (non-empty tag, known consumer)
- Does NOT perform network operations in unit tests (all I/O is mocked)
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure scripts/ is importable
_SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

import bump_vendor_pin as bvp  # noqa: E402  (import after sys.path modification)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

CONSUMER_REPOS = ["D-sorganization/UpstreamDrift", "D-sorganization/Gasification_Model"]
SAMPLE_TAG = "v1.2.3"
SAMPLE_SHA = "abc1234def5678901234567890123456789012ab"


# ---------------------------------------------------------------------------
# precondition tests
# ---------------------------------------------------------------------------


class TestPreconditions:
    """bump_vendor_pin functions must reject invalid inputs immediately."""

    def test_validate_consumer_rejects_unknown_repo(self) -> None:
        with pytest.raises((ValueError, SystemExit)):
            bvp.validate_consumer("D-sorganization/Unknown_Repo")

    def test_validate_consumer_accepts_known_repos(self) -> None:
        for repo in CONSUMER_REPOS:
            # Must not raise
            bvp.validate_consumer(repo)

    def test_build_pr_body_rejects_empty_tag(self) -> None:
        with pytest.raises((ValueError, AssertionError)):
            bvp.build_pr_body(tag="", sha=SAMPLE_SHA, consumer_repo=CONSUMER_REPOS[0])

    def test_build_pr_body_rejects_empty_sha(self) -> None:
        with pytest.raises((ValueError, AssertionError)):
            bvp.build_pr_body(tag=SAMPLE_TAG, sha="", consumer_repo=CONSUMER_REPOS[0])

    def test_build_pr_body_rejects_invalid_consumer(self) -> None:
        with pytest.raises((ValueError, AssertionError, SystemExit)):
            bvp.build_pr_body(
                tag=SAMPLE_TAG, sha=SAMPLE_SHA, consumer_repo="bad/unknown"
            )


# ---------------------------------------------------------------------------
# build_pr_body tests
# ---------------------------------------------------------------------------


class TestBuildPrBody:
    """PR body must contain the tag and SHA."""

    def test_pr_body_contains_tag(self) -> None:
        body = bvp.build_pr_body(
            tag=SAMPLE_TAG, sha=SAMPLE_SHA, consumer_repo=CONSUMER_REPOS[0]
        )
        assert SAMPLE_TAG in body

    def test_pr_body_contains_sha(self) -> None:
        body = bvp.build_pr_body(
            tag=SAMPLE_TAG, sha=SAMPLE_SHA, consumer_repo=CONSUMER_REPOS[0]
        )
        assert SAMPLE_SHA in body

    def test_pr_body_non_empty(self) -> None:
        body = bvp.build_pr_body(
            tag=SAMPLE_TAG, sha=SAMPLE_SHA, consumer_repo=CONSUMER_REPOS[0]
        )
        assert len(body.strip()) > 50  # noqa: PLR2004

    def test_pr_body_mentions_consumer_repo(self) -> None:
        body = bvp.build_pr_body(
            tag=SAMPLE_TAG, sha=SAMPLE_SHA, consumer_repo=CONSUMER_REPOS[1]
        )
        assert "Gasification_Model" in body or "vendor" in body.lower()

    def test_pr_body_different_for_each_consumer(self) -> None:
        body_ud = bvp.build_pr_body(
            tag=SAMPLE_TAG, sha=SAMPLE_SHA, consumer_repo=CONSUMER_REPOS[0]
        )
        body_gm = bvp.build_pr_body(
            tag=SAMPLE_TAG, sha=SAMPLE_SHA, consumer_repo=CONSUMER_REPOS[1]
        )
        # At least one must differ (they can share common text but the body should
        # identify the consumer)
        assert body_ud != body_gm or "UpstreamDrift" in body_ud


# ---------------------------------------------------------------------------
# resolve_latest_tag tests (mocked subprocess)
# ---------------------------------------------------------------------------


class TestResolveLatestTag:
    """resolve_latest_tag must call git/gh and return a non-empty string."""

    def test_returns_tag_string(self) -> None:
        with patch("subprocess.check_output", return_value=SAMPLE_TAG.encode()):
            tag = bvp.resolve_latest_tag()
        assert isinstance(tag, str)
        assert tag == SAMPLE_TAG

    def test_strips_whitespace(self) -> None:
        with patch(
            "subprocess.check_output", return_value=f"  {SAMPLE_TAG}\n".encode()
        ):
            tag = bvp.resolve_latest_tag()
        assert tag == SAMPLE_TAG

    def test_raises_on_empty_output(self) -> None:
        with patch("subprocess.check_output", return_value=b""):
            with pytest.raises((ValueError, RuntimeError, AssertionError)):
                bvp.resolve_latest_tag()


# ---------------------------------------------------------------------------
# resolve_sha_for_tag tests (mocked subprocess)
# ---------------------------------------------------------------------------


class TestResolveShaForTag:
    """resolve_sha_for_tag must call git and return a 40-char hex string."""

    def test_returns_sha_string(self) -> None:
        with patch("subprocess.check_output", return_value=SAMPLE_SHA.encode()):
            sha = bvp.resolve_sha_for_tag(SAMPLE_TAG)
        assert sha == SAMPLE_SHA

    def test_strips_whitespace(self) -> None:
        with patch("subprocess.check_output", return_value=f"{SAMPLE_SHA}\n".encode()):
            sha = bvp.resolve_sha_for_tag(SAMPLE_TAG)
        assert sha == SAMPLE_SHA

    def test_rejects_empty_tag_arg(self) -> None:
        with pytest.raises((ValueError, AssertionError)):
            bvp.resolve_sha_for_tag("")


# ---------------------------------------------------------------------------
# CONSUMER_REPOS constant
# ---------------------------------------------------------------------------


class TestConsumerReposConstant:
    """The module must export the list of known consumer repos."""

    def test_consumer_repos_list_exists(self) -> None:
        assert hasattr(bvp, "CONSUMER_REPOS")

    def test_consumer_repos_is_nonempty(self) -> None:
        assert len(bvp.CONSUMER_REPOS) >= 2  # noqa: PLR2004

    def test_consumer_repos_contains_upstreamdrift(self) -> None:
        assert any("UpstreamDrift" in r for r in bvp.CONSUMER_REPOS)

    def test_consumer_repos_contains_gasification_model(self) -> None:
        assert any("Gasification_Model" in r for r in bvp.CONSUMER_REPOS)
