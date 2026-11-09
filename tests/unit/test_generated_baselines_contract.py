"""Tests for generated baselines inventory and bounded query views (#5995-#5999)."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from src.tools import generated_baselines as gb
from src.tools.generated_baselines import (
    DEFAULT_MAX_QUERY_BYTES,
    QueryOptions,
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


def _cli_bytes(payload: dict[str, Any]) -> int:
    return len((json.dumps(payload, indent=2) + "\n").encode("utf-8"))


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
    txt = p.read_text("utf-8").replace(old, new, 1)
    p.write_text(txt, encoding="utf-8", newline="\n")


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
    live = build_spec_changelog_baseline(ROOT)
    spec_b = b_map["spec_changelog"]
    for k in ("spec_prefix_sha256", "spec_suffix_sha256", "changelog_header_sha256"):
        assert live[k] == spec_b[k]
    assert live["baseline_row_count"] >= spec_b["baseline_row_count"]
    assert verify_baseline_integrity(ROOT, check_hashes=True) == []


@pytest.mark.unit
def test_real_fragment_collation_reproduces_5997_and_passes(tmp_path: Path) -> None:
    sc = _load_shared(ROOT, "spec_changelog")
    collate = _load_shared(ROOT, "changes_fragment_collate").collate
    archive = _load_shared(ROOT, "development_log").archive_repository_devlog
    inv = load_inventory(ROOT)
    b_spec = next(b for b in inv["baselines"] if b["id"] == "spec_changelog")
    spec_inv: dict[str, Any] = {"baselines": [b_spec]}
    live_text = (ROOT / "SPEC.md").read_text("utf-8")
    cl = sc.parse_changelog(live_text)
    pinned = sc.replace_rows(
        live_text, cl, cl.rows[len(cl.rows) - b_spec["baseline_row_count"] :]
    )
    _make_repo(tmp_path)
    (tmp_path / "SPEC.md").write_text(pinned, encoding="utf-8", newline="\n")
    assert build_spec_changelog_baseline(tmp_path) == b_spec
    summary_5995 = (
        "inventory generated baselines, define reproducibility contract "
        "and bounded query utility"
    )
    f5995 = _write_frag(tmp_path, "5995-baselines.md", 5995, summary_5995)
    collate(tmp_path, [f5995], pr=5996, today=date(2026, 11, 8), sha="0f907c2")
    if b_spec["baseline_row_count"] == 1293 and b_spec["size_bytes"] == 1048482:
        raw_5997 = (tmp_path / "SPEC.md").read_bytes()
        assert hashlib.sha256(raw_5997).hexdigest() == PR_5997_SHA
    assert verify_baseline_integrity(tmp_path, inventory=spec_inv) == []
    fa = _write_frag(tmp_path, "990001-a.md", 990001, "first collated fix")
    fb = _write_frag(tmp_path, "990002-b.md", 990002, "second collated fix")
    collate(tmp_path, [fa, fb], pr=990010, today=date(2026, 11, 9), sha="1111111")
    assert verify_baseline_integrity(tmp_path, inventory=spec_inv) == []
    fc = _write_frag(tmp_path, "990001-c.md", 990001, "follow-up slice")
    collate(tmp_path, [fc], pr=990011, today=date(2026, 11, 10), sha="2222222")
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
def test_spec_integrity_rejects_corruptions(tmp_path: Path, old: str, new: str) -> None:
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
        collate(tmp_path, [frag], pr=6101, today=date(2026, 11, 9), sha="3333333")
        assert verify_baseline_integrity(tmp_path, inventory=inv) == []
        if kind == "tampered_summary":
            _replace_spec(tmp_path, "valid text", "tampered text")
        else:
            _replace_spec(tmp_path, "| 2026-11-09 | #6101 |", "| 2026-11-08 | #6101 |")
    assert verify_baseline_integrity(tmp_path, inventory=inv) != []


@pytest.mark.unit
def test_spec_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
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


_SK, _SW = "sidekick_api_baseline", "swing_sim_api_baseline"
_BM, _SP = "benchmarks_baseline", "spec_changelog"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("b_id", "query", "min_total"),
    [
        (_SK, "", 253),
        (_SK, "py", 253),
        (_SK, "calculator", 97),
        (_SK, "standalone", 6),
        (_SW, "", 268),
        (_SW, "py", 268),
        (_SW, "flight", 53),
        (_SW, "ImpactSolverAPI", 1),
        (_BM, "", 9),
        (_BM, "euler", 1),
        (_SP, "", 1000),
    ],
)
def test_bounded_summary_and_broad_queries(
    b_id: str, query: str, min_total: int
) -> None:
    s = summarize_baseline(b_id, ROOT)
    assert _cli_bytes(s) <= DEFAULT_MAX_QUERY_BYTES and s["id"] == b_id
    res = query_baseline(b_id, query, ROOT)
    assert _cli_bytes(res) <= DEFAULT_MAX_QUERY_BYTES
    assert res["match_count"] == res["total_matches"] >= min_total
    assert res["returned_matches"] == len(res["matches"]) >= 1
    if res["returned_matches"] < res["total_matches"]:
        assert res["truncated"] is True and res["has_more"] is True
        assert res["next_offset"] == res["returned_matches"]
    if b_id == _SW:
        first_val = next(iter(res["matches"].values()))
        assert isinstance(first_val.get("explicit_all"), bool)
    if b_id == _BM:
        assert all("data" not in b.get("stats", {}) for b in res["matches"])


@pytest.mark.unit
def test_large_single_entries_bounded_across_formats(tmp_path: Path) -> None:
    for b_id, q, off in [
        (_SK, "lab/mocap/__init__.py", 0),
        (_SK, "ui/tools_sidebar/sidebar.py", 0),
        (_SK, "UnifiedToolsSidebar", 1),
        (_SW, "flight/__init__.py", 0),
    ]:
        res = query_baseline(b_id, q, ROOT, QueryOptions(offset=off))
        assert _cli_bytes(res) <= DEFAULT_MAX_QUERY_BYTES
        assert res["returned_matches"] == 1 and res["truncated"] is True
        assert res["item_truncated"] is True
    _make_repo(tmp_path)
    (tmp_path / "SPEC.md").write_text("huge_row " + ("w" * 5000) + "\n", "utf-8")
    (tmp_path / ".benchmarks").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tests/api_baselines").mkdir(parents=True, exist_ok=True)
    big = {f"Sym{i}": {"type": "class", "doc": "x" * 900} for i in range(25)}
    sk_obj = {"big_sk.py": {"__all__": list(big), "symbols": big}}
    sw_obj = {"big_sw.py": {"__all__": list(big), "explicit_all": True, "symbols": big}}
    (tmp_path / "tests/sidekick_api_baseline.json").write_text(json.dumps(sk_obj))
    sw_p = tmp_path / "tests/api_baselines/swing_sim_api_baseline.json"
    sw_p.write_text(json.dumps(sw_obj))
    b_entry = {
        "name": "huge_bench",
        "fullname": "pkg::" + ("f" * 4000),
        "extra_info": {"blob": "z" * 9000},
        "stats": {"mean": 1.25, "min": 1.0, "max": 1.5, "data": [1.1] * 500},
    }
    (tmp_path / ".benchmarks/baseline.json").write_text(
        json.dumps({"benchmarks": [b_entry]})
    )
    tight = QueryOptions(max_bytes=800)
    for b_id in (_SK, _SW, _BM, _SP):
        r_t = query_baseline(b_id, "", tmp_path, options=tight)
        assert _cli_bytes(r_t) <= 800 and r_t["returned_matches"] == 1
        assert r_t["truncated"] is True and r_t["item_truncated"] is True
        assert r_t["next_offset"] is None


def _walk_pages(b_id: str, q: str, opts: QueryOptions) -> list[Any]:
    collected: list[Any] = []
    off: int | None = 0
    while off is not None:
        p_opt = QueryOptions(off, opts.limit, opts.max_bytes)
        page = query_baseline(b_id, q, ROOT, p_opt)
        assert _cli_bytes(page) <= opts.max_bytes and page["returned_matches"] >= 1
        m = page["matches"]
        items = list(m) if isinstance(m, dict) else [r["name"] for r in m]
        assert set(items).isdisjoint(collected)
        collected.extend(items)
        off = page["next_offset"]
    return collected


@pytest.mark.unit
def test_pagination_completeness_and_full_data_route() -> None:
    sk_keys = _walk_pages(_SK, "standalone", QueryOptions())
    assert len(sk_keys) == 6 and sk_keys == sorted(sk_keys)
    sw_keys = _walk_pages(_SW, "flight", QueryOptions(limit=15, max_bytes=6000))
    assert len(sw_keys) == 53 and sw_keys == sorted(sw_keys)
    assert len(_walk_pages(_BM, "", QueryOptions(0, 3, 2500))) == 9
    full_opt = QueryOptions(limit=1, full_data=True)
    full_sk = query_baseline(_SK, "lab/mocap/__init__.py", ROOT, full_opt)
    assert _cli_bytes(full_sk) > 90000 and full_sk["item_truncated"] is False
    full_sw = query_baseline(_SW, "flight/__init__.py", ROOT, full_opt)
    sw_mod = full_sw["matches"]["flight/__init__.py"]
    assert _cli_bytes(full_sw) > 70000 and sw_mod["explicit_all"] is True
    full_b = query_baseline(_BM, "euler", ROOT, full_opt)
    b_stats = full_b["matches"][0]["stats"]
    assert len(b_stats["data"]) > 100
    past = query_baseline(_BM, "", ROOT, QueryOptions(offset=999))
    assert past["returned_matches"] == 0 and past["next_offset"] is None
    for off_v, lim_v, mb_v in [(-1, 1, 1000), (0, 0, 1000), (0, 1, 100)]:
        with pytest.raises(ValueError):
            QueryOptions(off_v, lim_v, mb_v)


@pytest.mark.unit
def test_query_cli_sizing_empty_query_and_flags(
    capsys: pytest.CaptureFixture[str],
) -> None:
    base = ["--repo-root", str(ROOT)]
    for b_id, q in [(_SK, ""), (_SK, "py"), (_SW, ""), (_BM, "")]:
        rc = gb.main([*base, "--query", q, "--baseline", b_id])
        raw_out = capsys.readouterr().out
        assert rc == 0 and len(raw_out.encode("utf-8")) <= DEFAULT_MAX_QUERY_BYTES
        assert json.loads(raw_out)["returned_matches"] >= 1
    p_flags = "--query=standalone --offset=1 --limit=2 --max-bytes=4000".split()
    assert gb.main([*base, f"--baseline={_SK}", *p_flags]) == 0
    out_p = capsys.readouterr().out
    assert len(out_p.encode("utf-8")) <= 4000 and json.loads(out_p)["offset"] == 1
    f_args = [*base, "--query=lab/mocap/__init__.py", f"--baseline={_SK}", "--full"]
    assert gb.main(f_args) == 0 and len(capsys.readouterr().out.encode("utf-8")) > 90000
    assert gb.main(["--query", "py"]) == 1
    assert gb.main(["--query=py", f"--baseline={_SK}", "--offset=-1"]) == 1
    assert gb.main(["--query=py", "--baseline=unknown_id"]) == 1
    capsys.readouterr()
