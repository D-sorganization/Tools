"""Tests for generated baselines inventory and bounded query views (#5995, #5998)."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from src.tools import generated_baselines as gb
from src.tools.generated_baselines import (
    build_spec_changelog_baseline,
    load_inventory,
    query_baseline,
    summarize_baseline,
    verify_baseline_integrity,
)
from src.tools.spec_baseline_integrity import _load_shared

ROOT = Path(__file__).resolve().parents[2]
PR_5997_SHA = "2a404a33cb610edaa1cbe0dfa07735c2c80ff5515bb1d461ddebe94a867f9a5a"  # pragma: allowlist secret  # noqa: E501
_SEP = "| ---------- | ---------- | ---------- |\n"


def _write_frag(repo: Path, name: str, issue: int, summary: str) -> Path:
    (repo / "changes").mkdir(parents=True, exist_ok=True)
    path = repo / "changes" / name
    path.write_text(f"---\nissue: {issue}\nsummary: {summary}\n---\n", encoding="utf-8")
    return path


def _make_repo(repo_dir: Path) -> None:
    (repo_dir / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    spec = (
        "# Tools Specification\n\n| Field | Value |\n| --- | --- |\n"
        "| **Spec Version** | 1.2.0 |\n\n## 12. Change Log\n\n"
        f"| Date       | PR         | Changes    |\n{_SEP}"
        "| 2026-11-01 | #5991 | reviewed baseline row one |\n"
        "| 2026-10-30 | #5980 | reviewed baseline row two |\n"
        "---\n\n## 13. Appendix\n\nReviewed footer.\n"
    )
    (repo_dir / "SPEC.md").write_text(spec, encoding="utf-8", newline="\n")
    (repo_dir / "docs/development").mkdir(parents=True, exist_ok=True)
    devlog = repo_dir / "docs/development/DEVELOPMENT_LOG.md"
    devlog.write_text(
        "# Dev\n\n## Active\n\n## Shipped\n", encoding="utf-8", newline="\n"
    )


def _replace_spec(repo_dir: Path, old: str, new: str) -> None:
    p = repo_dir / "SPEC.md"
    p.write_text(
        p.read_text("utf-8").replace(old, new, 1), encoding="utf-8", newline="\n"
    )


@pytest.mark.unit
def test_inventory_and_all_baselines_match_hashes() -> None:
    inv = load_inventory(ROOT)
    assert inv["schema_version"] == "1.0.0"
    b_map = {b["id"]: b for b in inv["baselines"]}
    exp = {
        "benchmarks_baseline",
        "sidekick_api_baseline",
        "swing_sim_api_baseline",
        "spec_changelog",
    }
    assert exp.issubset(b_map.keys())
    for entry in b_map.values():
        assert entry["generator"] and entry["test_consumers"] and entry["review_owner"]
        assert entry["determinism"] in {
            "deterministic",
            "environment_dependent_benchmark",
        }
        assert len(entry["sha256"]) == 64
    assert b_map["spec_changelog"] == build_spec_changelog_baseline(ROOT)
    assert verify_baseline_integrity(ROOT, check_hashes=True) == []


@pytest.mark.unit
def test_real_fragment_collation_reproduces_5997_and_passes(tmp_path: Path) -> None:
    collate = _load_shared(ROOT, "changes_fragment_collate").collate
    archive = _load_shared(ROOT, "development_log").archive_repository_devlog
    (tmp_path / "pyproject.toml").write_text("[project]\nname='t'\n", encoding="utf-8")
    (tmp_path / "SPEC.md").write_bytes((ROOT / "SPEC.md").read_bytes())
    (tmp_path / "docs/development").mkdir(parents=True, exist_ok=True)
    dl_dst = tmp_path / "docs/development/DEVELOPMENT_LOG.md"
    dl_dst.write_bytes((ROOT / "docs/development/DEVELOPMENT_LOG.md").read_bytes())

    summary_5995 = (
        "inventory generated baselines, define reproducibility contract "
        "and bounded query utility"
    )
    f5995 = _write_frag(tmp_path, "5995-baselines.md", 5995, summary_5995)
    sha0 = "0f907c2ffbbc64bf94058adbdb00422a980c9712"
    collate(tmp_path, [f5995], pr=5996, today=date(2026, 11, 8), sha=sha0)
    raw_5997 = (tmp_path / "SPEC.md").read_bytes()
    assert hashlib.sha256(raw_5997).hexdigest() == PR_5997_SHA

    inv = load_inventory(ROOT)
    b_spec = [b for b in inv["baselines"] if b["id"] == "spec_changelog"]
    spec_inv: dict[str, Any] = {"baselines": b_spec}
    assert verify_baseline_integrity(tmp_path, inventory=spec_inv) == []

    fa = _write_frag(tmp_path, "5998-a.md", 5998, "first collated fix")
    fb = _write_frag(tmp_path, "5999-b.md", 5999, "second collated fix")
    collate(tmp_path, [fa, fb], pr=6001, today=date(2026, 11, 9), sha="1234567890ab")
    assert verify_baseline_integrity(tmp_path, inventory=spec_inv) == []

    fc = _write_frag(tmp_path, "5995-c.md", 5995, "follow-up slice")
    collate(tmp_path, [fc], pr=6002, today=date(2026, 11, 10), sha="abcdef123456")
    archive(tmp_path, today=date(2026, 12, 1), days=14)
    assert verify_baseline_integrity(tmp_path, inventory=spec_inv) == []


@pytest.mark.unit
@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("1.2.0", "9.9.9"),
        ("Reviewed footer.", "Tampered footer."),
        ("| Date       |", "| Date  |"),
        ("| 2026-10-30 | #5980 | reviewed baseline row two |\n", ""),
        ("reviewed baseline row one", "mutated baseline row one"),
        ("| 2026-11-01 | #5991 |", "|  2026-11-01 | #5991 |"),
        (_SEP, f"{_SEP}| 2026-11-09 | #5991 | duplicate pr key |\n"),
        (_SEP, f"{_SEP}| 2026-11-09 | #7777 | manual uncollated edit |\n"),
    ],
)
def test_spec_integrity_rejects_direct_corruptions(
    tmp_path: Path, old: str, new: str
) -> None:
    _make_repo(tmp_path)
    inv = {"baselines": [build_spec_changelog_baseline(tmp_path)]}
    _replace_spec(tmp_path, old, new)
    assert verify_baseline_integrity(tmp_path, inventory=inv) != []


@pytest.mark.unit
@pytest.mark.parametrize(
    "kind", ["fake_devlog", "tampered_summary", "tampered_date", "crlf"]
)
def test_spec_integrity_rejects_provenance_tampering(tmp_path: Path, kind: str) -> None:
    _make_repo(tmp_path)
    inv = {"baselines": [build_spec_changelog_baseline(tmp_path)]}
    spec_path = tmp_path / "SPEC.md"
    if kind == "crlf":
        spec_path.write_bytes(spec_path.read_bytes().replace(b"\n", b"\r\n", 1))
    elif kind == "fake_devlog":
        _replace_spec(tmp_path, _SEP, f"{_SEP}| 2026-11-09 | #7777 | manual edit |\n")
        dl = tmp_path / "docs/development/DEVELOPMENT_LOG.md"
        fake = (
            "\n### DL-#7777 · Fake\n\n- **State:** shipped\n- **PR:** #7777\n"
            "- **Started:** 2026-11-09\n- **Last verified:** 2026-11-09 manual\n"
            "- **Summary:** manual edit\n"
        )
        dl.write_text(dl.read_text("utf-8") + fake, encoding="utf-8", newline="\n")
    else:
        collate = _load_shared(ROOT, "changes_fragment_collate").collate
        frag = _write_frag(tmp_path, "6100-x.md", 6100, "valid text")
        collate(tmp_path, [frag], pr=6101, today=date(2026, 11, 9), sha="deadbeef1234")
        assert verify_baseline_integrity(tmp_path, inventory=inv) == []
        if kind == "tampered_summary":
            _replace_spec(tmp_path, "valid text", "tampered text")
        else:
            _replace_spec(tmp_path, "| 2026-11-09 | #6101 |", "| 2026-11-08 | #6101 |")
    assert verify_baseline_integrity(tmp_path, inventory=inv) != []


@pytest.mark.unit
def test_spec_refresh_cli_and_non_spec_exact_hashes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _make_repo(tmp_path)
    jfile = tmp_path / "sample.json"
    jfile.write_text('{"ok": true}\n', encoding="utf-8", newline="\n")
    sha = hashlib.sha256(jfile.read_bytes()).hexdigest()

    assert gb.main(["--repo-root", str(tmp_path), "--print-spec-baseline"]) == 0
    emitted = json.loads(capsys.readouterr().out)
    inv = {"baselines": [{"id": "s", "path": "sample.json", "sha256": sha}, emitted]}
    assert verify_baseline_integrity(tmp_path, inventory=inv) == []

    jfile.write_text('{"ok": false}\n', encoding="utf-8", newline="\n")
    errs = verify_baseline_integrity(tmp_path, inventory=inv)
    assert len(errs) == 1 and "Hash mismatch for sample.json" in errs[0]

    bad = {"baselines": [{"path": "sample.json", "integrity_mode": "unknown"}]}
    errs_bad = verify_baseline_integrity(tmp_path, inventory=bad)
    assert "Unsupported integrity_mode" in errs_bad[0]


@pytest.mark.unit
def test_integrity_check_detects_missing_file(tmp_path: Path) -> None:
    inv = {
        "schema_version": "1.0.0",
        "baselines": [{"path": "missing.json", "sha256": "0" * 64}],
    }
    errs = verify_baseline_integrity(tmp_path, check_hashes=True, inventory=inv)
    assert len(errs) == 1 and "File not found: missing.json" in errs[0]


@pytest.mark.unit
def test_bounded_summary_and_query_views() -> None:
    for b_id in [
        "benchmarks_baseline",
        "sidekick_api_baseline",
        "swing_sim_api_baseline",
    ]:
        s = summarize_baseline(b_id, ROOT)
        assert (
            len(json.dumps(s)) < 10000
            and s["id"] == b_id
            and s["size_bytes"] > 1000000
            and "entry_count" in s
        )
    sk = query_baseline("sidekick_api_baseline", "standalone", ROOT)
    assert sk["match_count"] > 0 and all("standalone" in m for m in sk["matches"])
    bench = query_baseline("benchmarks_baseline", "euler", ROOT)
    assert bench["match_count"] >= 1 and any(
        "euler" in b["name"] for b in bench["matches"]
    )
    assert all("data" not in b.get("stats", {}) for b in bench["matches"])
