"""Atomic command-line bundle publication for development-log migration."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from shared_scripts import development_log_migration
from shared_scripts.development_log import validate_devlog_content
from tests.test_development_log_migration import SOURCE, _plan, _repo


def test_cli_writes_complete_bundle_and_never_modifies_source(tmp_path: Path) -> None:
    root, commit = _repo(tmp_path)
    source = root / "DEVELOPMENT_LOG.md"
    source.write_bytes(SOURCE.encode())
    plan = root / "plan.json"
    plan.write_text(json.dumps(_plan()), encoding="utf-8")
    bundle = root / "review-bundle"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "shared_scripts.development_log_migration",
            "--repo-root",
            str(root),
            "--source",
            str(source),
            "--plan",
            str(plan),
            "--bundle-dir",
            str(bundle),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert source.read_bytes() == SOURCE.encode()
    assert {item.name for item in bundle.iterdir()} == {
        "DEVELOPMENT_LOG.candidate.md",
        "migration-receipt.json",
    }
    assert (
        validate_devlog_content(
            (bundle / "DEVELOPMENT_LOG.candidate.md").read_text(), source
        )
        == []
    )
    again = subprocess.run(
        [
            sys.executable,
            "-m",
            "shared_scripts.development_log_migration",
            "--repo-root",
            str(root),
            "--source",
            str(source),
            "--plan",
            str(plan),
            "--bundle-dir",
            str(bundle),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert again.returncode != 0
    assert "must not already exist" in again.stderr


def _assert_cli_failure_leaves_no_partial_bundle_or_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    root, _ = _repo(tmp_path)
    source = root / "DEVELOPMENT_LOG.md"
    source.write_bytes(SOURCE.encode())
    plan = root / "plan.json"
    plan.write_text(json.dumps(_plan()), encoding="utf-8")
    bundle = root / "review-bundle"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "development_log_migration.py",
            "--repo-root",
            str(root),
            "--source",
            str(source),
            "--plan",
            str(plan),
            "--bundle-dir",
            str(bundle),
        ],
    )
    if failure == "receipt_write":
        original_write = Path.write_text

        def fail_receipt(path: Path, *args: Any, **kwargs: Any) -> int:
            if path.name == "migration-receipt.json":
                raise OSError("injected receipt write failure")
            return original_write(path, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", fail_receipt)
    else:

        def fail_rename(*_args: Any) -> None:
            raise OSError("injected publish failure")

        monkeypatch.setattr(development_log_migration.os, "replace", fail_rename)
    with pytest.raises(OSError, match="injected"):
        development_log_migration.main()
    assert source.read_bytes() == SOURCE.encode()
    assert not bundle.exists()
    assert list(root.glob(".devlog-migration-*")) == []


def test_cli_receipt_write_failure_is_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _assert_cli_failure_leaves_no_partial_bundle_or_staging(
        tmp_path, monkeypatch, "receipt_write"
    )


def test_cli_publish_rename_failure_is_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _assert_cli_failure_leaves_no_partial_bundle_or_staging(
        tmp_path, monkeypatch, "publish_rename"
    )
