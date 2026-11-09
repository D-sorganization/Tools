"""SPEC.md structural-anchor computation and collated-changelog verification (#5998)."""

from __future__ import annotations

import hashlib
import importlib.util
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

_hex64_re = re.compile(r"^[0-9a-f]{64}$")
_pr_key_re = re.compile(r"^#\d+$")
_collated_stamp_re = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})\s+\(`[0-9a-fA-F]{7,40}`;\s+"
    r"collated from changes/[A-Za-z0-9._-]+\.md\)$"
)
_spec_hash_keys = (
    "sha256",
    "spec_prefix_sha256",
    "spec_suffix_sha256",
    "changelog_header_sha256",
    "baseline_rows_sha256",
)


@dataclass(frozen=True)
class _SpecSnapshot:
    raw_sha: str
    prefix_sha: str
    suffix_sha: str
    header_sha: str
    rows: list[Any]
    size_bytes: int


def _sha(data: str | bytes) -> str:
    raw = data.encode("utf-8") if isinstance(data, str) else data
    return hashlib.sha256(raw).hexdigest()


def _render_rows(rows: list[Any]) -> str:
    return "".join(f"{row.render()}\n" for row in rows)


def _load_shared(repo_root: Path, stem: str) -> ModuleType:
    candidate = repo_root / "shared_scripts" / f"{stem}.py"
    if not candidate.is_file():
        here = Path(__file__).resolve()
        candidate = here.parents[2] / "shared_scripts" / f"{stem}.py"
    if not candidate.is_file():
        raise FileNotFoundError(f"Shared script not found: {stem}.py")
    key = f"_tools_shared_{stem}_{_sha(str(candidate))[:12]}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, candidate)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {candidate}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    loader = spec.loader
    loader.exec_module(mod)
    return mod


def _parse_spec(repo_root: Path, rel: str) -> tuple[_SpecSnapshot | None, list[str]]:
    rel_path = Path(rel)
    if not rel or rel_path.is_absolute() or ".." in rel_path.parts:
        raise ValueError(f"spec_rel_path must stay within repo_root: {rel!r}")
    raw = (repo_root / rel_path).read_bytes()
    if b"\r" in raw:
        return None, [f"Unexpected CR byte in {rel}; LF line endings required"]
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        return None, [f"Invalid UTF-8 in {rel}: {exc}"]
    sc = _load_shared(repo_root, "spec_changelog")
    try:
        cl = sc.parse_changelog(text)
    except Exception as exc:
        return None, [f"Invalid SPEC change log in {rel}: {exc}"]
    table = cl.header_block + _render_rows(cl.rows)
    if text[cl.start : cl.end] != table or sc.replace_rows(text, cl, cl.rows) != text:
        return None, [f"Malformed non-row content in {rel} change-log table"]
    issues = sc.validate(cl)
    if issues:
        return None, [f"Invalid change-log in {rel}: {msg}" for msg in issues]
    snap = _SpecSnapshot(
        _sha(raw),
        _sha(text[: cl.start]),
        _sha(text[cl.end :]),
        _sha(cl.header_block),
        list(cl.rows),
        len(raw),
    )
    return snap, []


def compute_spec_changelog_anchors(
    repo_root: Path, spec_rel_path: str = "SPEC.md"
) -> dict[str, Any]:
    """Compute deterministic ``SPEC.md`` anchor fields from a valid checkout."""
    if not repo_root.is_dir():
        raise ValueError(f"repo_root must be an existing directory: {repo_root}")
    snap, errs = _parse_spec(repo_root, spec_rel_path)
    if errs or snap is None:
        raise ValueError("; ".join(errs))
    anchors: dict[str, Any] = {
        "path": spec_rel_path,
        "size_bytes": snap.size_bytes,
        "sha256": snap.raw_sha,
        "integrity_mode": "spec_changelog_collation",
        "spec_prefix_sha256": snap.prefix_sha,
        "spec_suffix_sha256": snap.suffix_sha,
        "changelog_header_sha256": snap.header_sha,
        "baseline_row_count": len(snap.rows),
        "baseline_rows_sha256": _sha(_render_rows(snap.rows)),
    }
    assert all(_hex64_re.match(anchors[k]) for k in _spec_hash_keys)
    return anchors


def _dfield(entry: Any, key: str) -> str:
    fields = entry.fields
    return str(fields.get(key, "")).strip()


def _check_collated_row(row: Any, devlog_entries: list[Any]) -> list[str]:
    if not _pr_key_re.match(row.key):
        return [f"Collated SPEC.md row must carry #<pr> key, got {row.key!r}"]
    pr_re = re.compile(rf"(?<!\d){re.escape(row.key)}(?!\d)")
    matches = [e for e in devlog_entries if pr_re.search(_dfield(e, "PR"))]
    if not matches:
        return [f"Uncollated SPEC.md row {row.key}: missing DEVELOPMENT_LOG entry"]
    dates: list[str] = []
    for item in matches:
        stamp = _collated_stamp_re.match(_dfield(item, "Last verified"))
        if stamp is None or stamp.group("date") < row.date:
            return [f"Row {row.key} lacks collated provenance in {item.entry_id}"]
        dates.append(stamp.group("date"))
    if all(
        _dfield(m, "PR") == row.key and _dfield(m, "Started") == dates[i]
        for i, m in enumerate(matches)
    ):
        if any(d != row.date for d in dates):
            return [f"SPEC.md row {row.key} date {row.date} disagrees with devlog"]
        sums = [_dfield(m, "Summary") for m in matches]
        allowed = {"; ".join(dict.fromkeys(sums)), "; ".join(dict.fromkeys(sums[::-1]))}
        if row.summary.strip() not in allowed:
            return [f"SPEC.md row {row.key} summary disagrees with collated devlog"]
    return []


def verify_spec_changelog_integrity(
    repo_root: Path, entry: dict[str, Any]
) -> list[str]:
    """Verify ``SPEC.md`` reviewed sections, baseline rows, and collated provenance."""
    if not isinstance(entry, dict):
        raise TypeError("entry must be a dictionary")
    rel = str(entry.get("path", ""))
    cnt = entry.get("baseline_row_count")
    bad_hash = any(not _hex64_re.match(str(entry.get(k, ""))) for k in _spec_hash_keys)
    if bad_hash or isinstance(cnt, bool) or not isinstance(cnt, int) or cnt < 0:
        return [f"Invalid spec_changelog anchor metadata for {rel}"]
    snap, parse_errs = _parse_spec(repo_root, rel)
    if parse_errs or snap is None:
        return parse_errs
    checks = (
        (snap.prefix_sha, entry["spec_prefix_sha256"], "Non-changelog prefix"),
        (snap.suffix_sha, entry["spec_suffix_sha256"], "Post-changelog section"),
        (snap.header_sha, entry["changelog_header_sha256"], "Change-log header"),
    )
    errs = [f"{lbl} modified in {rel}" for got, exp, lbl in checks if got != exp]
    added = len(snap.rows) - cnt
    if added < 0:
        return [*errs, f"Row count {len(snap.rows)} in {rel} < baseline {cnt}"]
    if _sha(_render_rows(snap.rows[added:])) != entry["baseline_rows_sha256"]:
        errs.append(f"Pinned baseline change-log rows modified in {rel}")
    if errs:
        return errs
    if added == 0:
        exp = entry["sha256"]
        return [] if snap.raw_sha == exp else [f"Hash mismatch for {rel}"]
    dl = _load_shared(repo_root, "development_log")
    canon = dl.resolve_canonical_devlog_path(repo_root)
    parent = canon.parent
    archives = sorted(parent.glob(f"{canon.stem}_ARCHIVE_*.md"))
    files = [p for p in [canon, *archives] if p.is_file()]
    if not files:
        return ["Missing canonical DEVELOPMENT_LOG.md for collated SPEC.md rows"]
    entries = [e for p in files for e in dl.parse_entries(p.read_text("utf-8"))]
    for row in snap.rows[:added]:
        errs.extend(_check_collated_row(row, entries))
    return errs
