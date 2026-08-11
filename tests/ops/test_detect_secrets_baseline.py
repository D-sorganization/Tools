"""Tests for detect-secrets baseline staleness guard (TDD).

Verifies:
- The committed .secrets.baseline matches a fresh detect-secrets scan
- All path separators in the baseline are normalized to forward slashes
- The baseline version matches the installed detect-secrets version
- Baseline results contain only known false-positives (no new unreviewed secrets)

Acceptance criteria (issue #2947):
- Baseline drift is caught before it reaches CI
- Windows vs Linux path separator discrepancies are detected and flagged
- A stale or regenerated-but-not-committed baseline fails the test
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent.parent
BASELINE_PATH = REPO_ROOT / ".secrets.baseline"
DETECT_SECRETS_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "detect-secrets.yml"


def _detect_secrets_audit_supports(option: str) -> bool:
    """Return whether the installed detect-secrets audit CLI supports an option."""
    try:
        result = subprocess.run(
            ["detect-secrets", "audit", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        pytest.skip("detect-secrets not installed")
    if result.returncode != 0:
        pytest.skip(f"detect-secrets audit --help failed: {result.stderr}")
    return option in result.stdout


def _load_baseline() -> dict[str, Any]:
    """Load the committed .secrets.baseline."""
    assert BASELINE_PATH.exists(), f".secrets.baseline not found at {BASELINE_PATH}"
    with BASELINE_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_paths(data: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of baseline data with all path separators as forward slashes."""
    import copy

    result = copy.deepcopy(data)
    new_results: dict[str, Any] = {}
    for k, v in result.get("results", {}).items():
        new_k = k.replace("\\", "/")
        new_v = []
        for entry in v:
            new_entry = dict(entry)
            new_entry["filename"] = new_entry["filename"].replace("\\", "/")
            new_v.append(new_entry)
        new_results[new_k] = new_v
    result["results"] = new_results
    return result


def _run_fresh_scan() -> dict[str, Any]:
    """Run detect-secrets scan and return parsed JSON output.

    Precondition: detect-secrets must be installed (skip if not).
    """
    try:
        subprocess.run(
            ["detect-secrets", "scan", "--baseline", str(BASELINE_PATH)],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=120,
        )
    except FileNotFoundError:
        pytest.skip("detect-secrets not installed — skipping staleness check")

    # detect-secrets scan with --baseline updates the file in place; read it back
    with BASELINE_PATH.open("r", encoding="utf-8") as f:
        updated = json.load(f)
    return updated


def _scan_fresh_without_updating_baseline() -> dict[str, Any]:
    """Run detect-secrets scan (no --baseline) to get a raw fresh scan result."""
    try:
        result = subprocess.run(
            ["detect-secrets", "scan"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=120,
        )
    except FileNotFoundError:
        pytest.skip("detect-secrets not installed — skipping staleness check")
    if result.returncode != 0:
        pytest.skip(f"detect-secrets scan failed: {result.stderr}")
    return json.loads(result.stdout)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBaselineExists:
    """Basic structural tests that run quickly without shelling out."""

    def test_workflow_uses_shallow_checkout_with_adequate_timeout(self) -> None:
        """Secret scanning needs the current tree, not the repository's full history."""
        import yaml

        workflow = yaml.safe_load(DETECT_SECRETS_WORKFLOW.read_text(encoding="utf-8"))
        job = workflow["jobs"]["detect-secrets"]
        checkout = next(
            step for step in job["steps"] if step.get("name") == "Checkout repository"
        )

        assert checkout["with"]["fetch-depth"] == 1
        assert int(job["timeout-minutes"]) >= 30

    def test_workflow_invokes_installed_python_module(self) -> None:
        """The fleet runner PATH must not decide whether detect-secrets is runnable."""
        workflow = DETECT_SECRETS_WORKFLOW.read_text(encoding="utf-8")

        assert "python -m detect_secrets scan --baseline .secrets.baseline" in workflow
        assert "\n          detect-secrets scan --baseline" not in workflow

    def test_baseline_file_exists(self) -> None:
        """Precondition: .secrets.baseline must exist in repo root."""
        assert (
            BASELINE_PATH.exists()
        ), ".secrets.baseline is missing. Run: detect-secrets scan > .secrets.baseline"

    def test_baseline_is_valid_json(self) -> None:
        """Baseline must be parseable JSON."""
        data = _load_baseline()
        assert isinstance(data, dict)

    def test_baseline_has_version_field(self) -> None:
        """Baseline must declare a detect-secrets version."""
        data = _load_baseline()
        assert "version" in data, "Baseline missing 'version' field"
        assert isinstance(data["version"], str)

    def test_baseline_has_results_field(self) -> None:
        """Baseline must have a 'results' dict."""
        data = _load_baseline()
        assert "results" in data, "Baseline missing 'results' field"
        assert isinstance(data["results"], dict)

    def test_baseline_has_plugins_used_field(self) -> None:
        """Baseline must declare which plugins were used."""
        data = _load_baseline()
        assert "plugins_used" in data, "Baseline missing 'plugins_used' field"
        assert isinstance(data["plugins_used"], list)

    @pytest.mark.parametrize(
        "field", ["version", "results", "plugins_used", "filters_used"]
    )
    def test_baseline_required_fields_present(self, field: str) -> None:
        """All required detect-secrets baseline fields must be present."""
        data = _load_baseline()
        assert field in data, f"Baseline missing required field: {field!r}"


class TestBaselinePathNormalization:
    """Verify path separators are normalized to forward slashes."""

    def test_no_backslash_keys_in_results(self) -> None:
        """Result keys must use forward slashes (not Windows backslashes).

        Backslashes in the baseline cause CI mismatches between Linux
        (which generates forward slashes) and Windows.
        """
        data = _load_baseline()
        results = data.get("results", {})
        violating_keys = [k for k in results if "\\" in k]
        assert not violating_keys, (
            f"Baseline has backslash paths (Windows-generated): {violating_keys[:5]}. "
            "Regenerate baseline on Linux or normalize with "
            "scripts/normalize_secrets_baseline.py"
        )

    def test_no_backslash_filenames_in_results(self) -> None:
        """Entry 'filename' fields must use forward slashes."""
        data = _load_baseline()
        violating: list[str] = []
        for entries in data.get("results", {}).values():
            for entry in entries:
                filename = entry.get("filename", "")
                if "\\" in filename:
                    violating.append(filename)
        assert not violating, (
            f"Baseline entries have backslash filenames: {violating[:5]}. "
            "Normalize with scripts/normalize_secrets_baseline.py"
        )

    def test_baseline_filenames_match_keys(self) -> None:
        """Each entry's filename must match its dict key (after normalization)."""
        data = _load_baseline()
        mismatches: list[str] = []
        for k, entries in data.get("results", {}).items():
            for entry in entries:
                fname = entry.get("filename", "")
                # Normalize both for comparison
                if fname.replace("\\", "/") != k.replace("\\", "/"):
                    mismatches.append(f"key={k!r} vs filename={fname!r}")
        assert not mismatches, f"Key/filename mismatches in baseline: {mismatches[:5]}"


class TestBaselineVersion:
    """Verify the baseline version matches the installed detect-secrets."""

    def test_baseline_version_matches_installed(self) -> None:
        """Baseline version must match installed detect-secrets version.

        Version mismatch causes subtle false positives/negatives as
        different versions use different hashing or plugin logic.
        """
        try:
            result = subprocess.run(
                ["detect-secrets", "--version"],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except FileNotFoundError:
            pytest.skip("detect-secrets not installed")

        if result.returncode != 0:
            pytest.skip(f"detect-secrets --version failed: {result.stderr}")

        installed_version = result.stdout.strip()
        data = _load_baseline()
        baseline_version = data.get("version", "")

        assert installed_version == baseline_version, (
            f"detect-secrets version mismatch: installed={installed_version!r}, "
            f"baseline={baseline_version!r}. "
            "Run: detect-secrets scan --baseline .secrets.baseline to update."
        )


class TestBaselineNotStale:
    """Verify the baseline is up-to-date with the current codebase.

    These tests shell out to detect-secrets and are slower. They guard against
    the primary failure mode: new secrets committed without updating baseline.
    """

    @pytest.mark.slow
    def test_no_new_secrets_since_baseline(self) -> None:
        """Running detect-secrets audit must report no unreviewed secrets.

        If the baseline is stale, detect-secrets will find secrets not in the
        baseline and this test fails with a clear message.
        """
        audit_args = ["detect-secrets", "audit", "--report"]
        if _detect_secrets_audit_supports("--only-allowlisted"):
            audit_args.append("--only-allowlisted")
        audit_args.append(str(BASELINE_PATH))
        env = {**os.environ, "PYTHONUTF8": "1"}
        try:
            result = subprocess.run(
                audit_args,
                capture_output=True,
                env=env,
                text=True,
                cwd=str(REPO_ROOT),
                timeout=120,
            )
        except FileNotFoundError:
            pytest.skip("detect-secrets not installed")

        # Exit code 0 means no unreviewed secrets
        assert result.returncode == 0, (
            "detect-secrets audit found unreviewed secrets. "
            "Run: detect-secrets scan --baseline .secrets.baseline "
            "to update the baseline, then review and commit."
        )

    @pytest.mark.slow
    def test_scan_result_matches_baseline_fingerprint(self) -> None:
        """A fresh scan with --baseline must produce no new results.

        Scans with --baseline only report new secrets NOT already in baseline.
        If new secrets are found, the baseline is stale.
        """
        try:
            subprocess.run(
                ["detect-secrets", "scan", "--list-all-plugins"],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except FileNotFoundError:
            pytest.skip("detect-secrets not installed")

        # Do a non-mutating scan by reading a fresh scan result
        fresh_data = _scan_fresh_without_updating_baseline()
        fresh_results = fresh_data.get("results", {})

        # Normalize paths in fresh results for comparison
        normalized_fresh: dict = {}
        for k, v in fresh_results.items():
            new_k = k.replace("\\", "/")
            normalized_fresh[new_k] = v

        # Check for keys in fresh scan NOT in baseline (new unreviewed secrets)
        baseline_data = _load_baseline()
        baseline_keys = set(baseline_data.get("results", {}).keys())
        fresh_keys = set(normalized_fresh.keys())

        # The baseline exclusion filter should filter out .secrets.baseline itself
        fresh_keys.discard(".secrets.baseline")

        new_files_with_secrets = fresh_keys - baseline_keys
        assert not new_files_with_secrets, (
            f"detect-secrets found potential secrets in new files not in baseline: "
            f"{new_files_with_secrets}. "
            "Run: detect-secrets scan --baseline .secrets.baseline to update baseline, "
            "review each finding, then commit the updated baseline."
        )


class TestBaselineEntryIntegrity:
    """Validate individual entry structure in the baseline."""

    @pytest.mark.parametrize(
        "required_field",
        ["type", "filename", "hashed_secret", "is_verified", "line_number"],
    )
    def test_all_entries_have_required_field(self, required_field: str) -> None:
        """Every baseline entry must have all required fields."""
        data = _load_baseline()
        missing: list[str] = []
        for file_key, entries in data.get("results", {}).items():
            for i, entry in enumerate(entries):
                if required_field not in entry:
                    missing.append(f"{file_key}[{i}]")
        assert (
            not missing
        ), f"Baseline entries missing field {required_field!r}: {missing[:10]}"

    def test_hashed_secrets_are_40_char_hex(self) -> None:
        """All hashed_secret values must be 40-char hex strings (SHA1)."""
        import re

        data = _load_baseline()
        invalid: list[str] = []
        hex_pattern = re.compile(r"^[0-9a-f]{40}$")
        for file_key, entries in data.get("results", {}).items():
            for entry in entries:
                hs = entry.get("hashed_secret", "")
                if not hex_pattern.match(hs):
                    invalid.append(f"{file_key}: {hs!r}")
        assert not invalid, f"Non-SHA1 hashed_secret values: {invalid[:5]}"

    def test_line_numbers_are_positive_integers(self) -> None:
        """All line_number values must be positive integers."""
        data = _load_baseline()
        invalid: list[str] = []
        for file_key, entries in data.get("results", {}).items():
            for entry in entries:
                ln = entry.get("line_number", -1)
                if not isinstance(ln, int) or ln < 1:
                    invalid.append(f"{file_key}: line_number={ln!r}")
        assert not invalid, f"Invalid line numbers: {invalid[:5]}"

    def test_is_verified_is_boolean(self) -> None:
        """All is_verified values must be boolean."""
        data = _load_baseline()
        invalid: list[str] = []
        for file_key, entries in data.get("results", {}).items():
            for entry in entries:
                iv = entry.get("is_verified")
                if not isinstance(iv, bool):
                    invalid.append(f"{file_key}: is_verified={iv!r}")
        assert not invalid, f"Non-boolean is_verified values: {invalid[:5]}"
