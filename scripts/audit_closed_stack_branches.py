#!/usr/bin/env python3
"""Audit closed/folded golf-app PR stacks for content that never reached main.

Evidence-only helper for Tools #4921 (Fleet Readiness Program Phase 0).

For each PR in the folded stack the script resolves a diffable ref (the
``origin/<head>`` branch, falling back to the recorded head SHA), computes the
three-dot diff against ``origin/main``, and lists every file the branch added
that is *still* absent from main. Each such file is classified best-effort:

* ``landed-elsewhere`` -- a majority of the file's top-level symbols exist
  somewhere on main (the file was moved/renamed/re-implemented).
* ``obsolete`` -- agent scratch, drafts, assessments, plans, PR notes.
* ``missing`` -- product code / tests / docs with no trace on main.

Files are then grouped by their top two path components and the group takes the
majority classification. Output is a deterministic JSON evidence file plus a
Markdown summary. Only ``gh`` and ``git`` are shelled out to; all classification
helpers are pure and unit-tested.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_PRS = [4466, 4449, 4169, 4173, 4174, 4209, 4212, 4217, 4233, 4246, 4436]
DEFAULT_REPO = "D-sorganization/Tools"
DEFAULT_BASE = "main"
DEFAULT_OUT_JSON = Path("docs/release/closed_stack_gap_audit.v1.json")
DEFAULT_OUT_MD = Path("docs/release/closed_stack_gap_audit.md")

SCHEMA_VERSION = 1
MAX_SYMBOLS_PER_FILE = 8
MIN_SYMBOL_LEN = 4
GENERIC_SYMBOLS = frozenset(
    {"main", "test", "setup", "run", "render", "App", "index", "default", "init"}
)
SYMBOL_EXTENSIONS = (".py", ".ts", ".tsx")
OBSOLETE_DIR_PREFIXES = (
    "drafts/",
    ".gaai/",
    "docs/assessments/",
    "assessments/",
    ".codex/",
    ".jules/",
    ".Jules/",
    "agent_scratch/",
    "scratch/",
    ".codex-worktrees/",
    "_codex_",
    "_wt_claude_",
)
# Whole-word tokens in the file name (so ``planner.py`` is *not* obsolete).
_OBSOLETE_NAME_RE = re.compile(
    r"(?<![a-z0-9])(?:codex|jules|plans?|pr_details)(?![a-z0-9])"
)
PRODUCT_PREFIXES = ("src/rate_of_closure", "src/shared/python", "rust_core", "tests")
MD_LIST_CAP = 60

CLASS_LANDED = "landed-elsewhere"
CLASS_MISSING = "missing"
CLASS_OBSOLETE = "obsolete"
CLASS_ORDER = (CLASS_LANDED, CLASS_MISSING, CLASS_OBSOLETE)

_PY_SYMBOL_RE = re.compile(
    r"^(?:async\s+def|def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", re.M
)
_TS_SYMBOL_RE = re.compile(
    r"^export\s+(?:default\s+)?(?:async\s+)?(?:function|const|class|let|var|interface|type|enum)"
    r"\s+([A-Za-z_$][A-Za-z0-9_$]*)",
    re.M,
)


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------


def extract_symbols(source: str, path: str) -> list[str]:
    """Return up to MAX_SYMBOLS_PER_FILE top-level symbol names from ``source``.

    Python: ``def``/``class`` at column 0. TS/TSX: ``export function|const|class``.
    Short (<4 chars), underscore-private and generic names are dropped; order of
    first appearance is preserved and duplicates removed.
    """
    if path.endswith(".py"):
        pattern = _PY_SYMBOL_RE
    elif path.endswith((".ts", ".tsx")):
        pattern = _TS_SYMBOL_RE
    else:
        return []
    seen: list[str] = []
    for match in pattern.finditer(source):
        name = match.group(1)
        if (
            len(name) < MIN_SYMBOL_LEN
            or name.startswith("_")
            or name in GENERIC_SYMBOLS
            or name in seen
        ):
            continue
        seen.append(name)
        if len(seen) >= MAX_SYMBOLS_PER_FILE:
            break
    return seen


def is_obsolete_path(path: str) -> bool:
    """Return True for scratch/draft/assessment/plan-style paths."""
    lowered = path.lower()
    if any(lowered.startswith(prefix.lower()) for prefix in OBSOLETE_DIR_PREFIXES):
        return True
    return bool(_OBSOLETE_NAME_RE.search(Path(path).name.lower()))


def classify_file(
    path: str, symbols_checked: list[str], symbols_found: list[dict[str, Any]]
) -> str:
    """Apply the classification rules to one added-but-absent file."""
    if is_obsolete_path(path):
        return CLASS_OBSOLETE
    if symbols_checked:
        found = len(symbols_found)
        if found >= 1 and found * 2 >= len(symbols_checked):
            return CLASS_LANDED
    return CLASS_MISSING


def group_prefix(path: str) -> str:
    """Return the top two path components (or the whole path if shallower)."""
    parts = path.split("/")
    if len(parts) <= 2:
        return parts[0]
    if path.startswith("src/"):
        return _src_prefix(parts)
    return "/".join(parts[:2])


def _src_prefix(parts: list[str]) -> str:
    # src/<package>/<subdir> is a more useful bucket than src/<package>.
    if len(parts) >= 4:
        return "/".join(parts[:3])
    return "/".join(parts[:2])


def majority_class(classes: list[str]) -> str:
    """Majority class with deterministic tie-break (landed < missing < obsolete)."""
    if not classes:
        return CLASS_MISSING
    counts = Counter(classes)
    best = max(counts.values())
    for cls in CLASS_ORDER:
        if counts.get(cls) == best:
            return cls
    return CLASS_MISSING  # pragma: no cover - CLASS_ORDER covers all classes


def group_files(missing_files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Bucket classified files by prefix and derive per-group classification."""
    buckets: dict[str, list[dict[str, Any]]] = {}
    for entry in missing_files:
        buckets.setdefault(group_prefix(entry["path"]), []).append(entry)
    groups: list[dict[str, Any]] = []
    for prefix in sorted(buckets):
        entries = sorted(buckets[prefix], key=lambda e: e["path"])
        classes = [e["classification"] for e in entries]
        counts = {cls: classes.count(cls) for cls in CLASS_ORDER}
        groups.append(
            {
                "prefix": prefix,
                "classification": majority_class(classes),
                "files": [e["path"] for e in entries],
                "counts": counts,
            }
        )
    return groups


def is_product_prefix(prefix: str) -> bool:
    return any(prefix == p or prefix.startswith(p + "/") for p in PRODUCT_PREFIXES)


def keep_recommendation(pr: dict[str, Any]) -> str:
    """One-line keep/drop recommendation derived from group classifications."""
    if not pr.get("reachable"):
        return "undetermined (ref unreachable)"
    product_missing = [
        g["prefix"]
        for g in pr.get("groups", [])
        if g["classification"] == CLASS_MISSING and is_product_prefix(g["prefix"])
    ]
    if product_missing:
        return "keep for review: missing product/test groups " + ", ".join(
            product_missing
        )
    if any(g["classification"] == CLASS_MISSING for g in pr.get("groups", [])):
        return "drop (only non-product groups are missing)"
    return "drop (nothing missing; landed elsewhere or obsolete)"


def compute_totals(prs: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, Any] = {
        "prs": len(prs),
        "reachable": sum(1 for p in prs if p.get("reachable")),
        "unreachable": [p["number"] for p in prs if not p.get("reachable")],
        "files_by_class": {cls: 0 for cls in CLASS_ORDER},
        "groups_by_class": {cls: 0 for cls in CLASS_ORDER},
    }
    for pr in prs:
        for f in pr.get("missing_files", []):
            totals["files_by_class"][f["classification"]] += 1
        for g in pr.get("groups", []):
            totals["groups_by_class"][g["classification"]] += 1
    return totals


def render_markdown(result: dict[str, Any]) -> str:
    """Render the audit result as a Markdown summary (bounded length)."""
    base = result["base"]
    lines: list[str] = [
        "# Closed-stack gap audit (Tools #4921)",
        "",
        f"Generated: {result['generated_at']}  ",
        f"Base: `{base['ref']}` @ `{base['sha']}`  ",
        f"Repo: `{result['repo']}`",
        "",
        "## Method",
        "",
        "For each PR the head branch (or recorded head SHA) is diffed against "
        "`origin/main` with the three-dot form. Files added relative to the "
        "merge-base **and** absent from `origin/main` are listed and classified:",
        "",
        "- `landed-elsewhere`: >=50% of the file's public top-level symbols "
        "(max 8, `_private` and generic names skipped) exist on main under "
        "`src/`, `tests/` or `rust_core/`.",
        "- `obsolete`: drafts, `.gaai/`, assessments, agent worktrees/scratch "
        "(`.codex-worktrees/`, `_codex_*`, `_wt_claude_*`), or file names with "
        "the words `codex`/`jules`/`plan`/`pr_details`.",
        "- `missing`: everything else.",
        "",
        "Groups take the majority class of their files. Evidence JSON: "
        "`docs/release/closed_stack_gap_audit.v1.json`.",
        "",
    ]
    unreachable = result["totals"]["unreachable"]
    if unreachable:
        lines.append(
            "**Unreachable refs:** "
            + ", ".join(f"#{n}" for n in unreachable)
            + " (see per-PR reason)."
        )
    else:
        lines.append("**Unreachable refs:** none - every PR head was diffable.")
    lines.append("")

    for pr in result["prs"]:
        lines.append(f"## #{pr['number']} - {pr['title']}")
        lines.append("")
        lines.append(
            f"- State: `{pr['state']}` | head `{pr['head_ref']}` "
            f"(`{(pr.get('head_oid') or '')[:12]}`) | base `{pr['base_ref']}` "
            f"(on origin: {pr['base_ref_on_origin']}) | merged {pr['merged_at']} "
            f"| closed {pr['closed_at']}"
        )
        lines.append(f"- URL: {pr['url']}")
        if pr.get("error"):
            lines.append(f"- gh error: `{pr['error']}`")
        if not pr.get("reachable"):
            lines.append(f"- **Ref unreachable:** {pr.get('reason')}")
            lines.append(f"- Recommendation: {keep_recommendation(pr)}")
            lines.append("")
            continue
        counts = pr["counts"]
        tip_note = (
            ""
            if pr.get("head_oid_matches_diff_ref")
            else f" (branch tip `{(pr.get('diff_ref_sha') or '')[:12]}` differs "
            "from PR head SHA)"
        )
        lines.append(
            f"- Diff ref: `{pr['diff_ref_used']}`{tip_note} | diffstat: "
            f"{pr['diffstat_summary'] or 'n/a'} | added {counts['added']}, "
            f"modified {counts['modified']}, deleted {counts['deleted']}"
        )
        lines.append(f"- Recommendation: {keep_recommendation(pr)}")
        lines.append("")
        if pr["groups"]:
            lines.append(
                "| group | class | files | landed-elsewhere / missing / obsolete |"
            )
            lines.append("|---|---|---:|---|")
            for g in pr["groups"]:
                c = g["counts"]
                lines.append(
                    f"| `{g['prefix']}` | {g['classification']} | {len(g['files'])} "
                    f"| {c[CLASS_LANDED]} / {c[CLASS_MISSING]} / {c[CLASS_OBSOLETE]} |"
                )
            lines.append("")
        missing = [
            f["path"]
            for f in pr["missing_files"]
            if f["classification"] == CLASS_MISSING
        ]
        if missing:
            lines.append(f"Missing files ({len(missing)}):")
            lines.append("")
            for path in missing[:MD_LIST_CAP]:
                lines.append(f"- `{path}`")
            if len(missing) > MD_LIST_CAP:
                lines.append(f"- ... and {len(missing) - MD_LIST_CAP} more")
            lines.append("")
        else:
            lines.append("No `missing` files.")
            lines.append("")

    totals = result["totals"]
    fb, gb = totals["files_by_class"], totals["groups_by_class"]
    lines += [
        "## Totals",
        "",
        f"- PRs audited: {totals['prs']} (reachable {totals['reachable']}, "
        f"unreachable {len(unreachable)})",
        f"- Files absent from main: {sum(fb.values())} "
        f"(landed-elsewhere {fb[CLASS_LANDED]}, missing {fb[CLASS_MISSING]}, "
        f"obsolete {fb[CLASS_OBSOLETE]})",
        f"- Groups: {sum(gb.values())} "
        f"(landed-elsewhere {gb[CLASS_LANDED]}, missing {gb[CLASS_MISSING]}, "
        f"obsolete {gb[CLASS_OBSOLETE]})",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Shell wrappers (not unit-tested; thin)
# ---------------------------------------------------------------------------


class Shell:
    def __init__(self, cwd: Path) -> None:
        self.cwd = cwd

    def run(
        self, args: list[str], check: bool = False
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            args,
            cwd=self.cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=check,
        )

    def git(self, *args: str) -> subprocess.CompletedProcess[str]:
        return self.run(["git", *args])

    def git_ok(self, *args: str) -> bool:
        return self.git(*args).returncode == 0


def fetch_pr(shell: Shell, repo: str, number: int) -> tuple[dict[str, Any] | None, str]:
    cp = shell.run(
        [
            "gh",
            "pr",
            "view",
            str(number),
            "--repo",
            repo,
            "--json",
            "number,title,state,headRefName,headRefOid,baseRefName,mergedAt,closedAt,url",
        ]
    )
    if cp.returncode != 0:
        return None, (cp.stderr or cp.stdout).strip()[:500]
    try:
        return json.loads(cp.stdout), ""
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        return None, f"invalid JSON from gh: {exc}"


def resolve_ref(shell: Shell, head_ref: str, head_oid: str) -> tuple[str | None, str]:
    if head_ref and shell.git_ok(
        "rev-parse", "--verify", "--quiet", f"origin/{head_ref}"
    ):
        return f"origin/{head_ref}", ""
    if head_oid:
        shell.git("fetch", "origin", head_oid)
        if shell.git_ok("cat-file", "-e", f"{head_oid}^{{commit}}"):
            return head_oid, ""
        return None, f"origin/{head_ref} absent and head SHA {head_oid} not fetchable"
    return None, "no head ref or SHA available"


def diffstat_summary(shell: Shell, base: str, ref: str) -> str:
    cp = shell.git("diff", "--stat", f"{base}...{ref}")
    lines = [ln.strip() for ln in cp.stdout.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def name_status(shell: Shell, base: str, ref: str) -> dict[str, list[str]]:
    cp = shell.git("diff", "--name-status", f"{base}...{ref}")
    out: dict[str, list[str]] = {"A": [], "M": [], "D": [], "other": []}
    for line in cp.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        status = parts[0][0]
        out.get(status, out["other"]).append(parts[-1])
    return out


GREP_CHUNK = 40


def missing_on_base(shell: Shell, base: str, paths: list[str]) -> list[str]:
    """Return the subset of ``paths`` that do not exist on ``base`` (one git call)."""
    if not paths:
        return []
    cp = subprocess.run(
        ["git", "cat-file", "--batch-check"],
        cwd=shell.cwd,
        input="".join(f"{base}:{p}\n" for p in paths),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    absent: list[str] = []
    for line in cp.stdout.splitlines():
        if line.endswith(" missing"):
            spec = line[: -len(" missing")]
            absent.append(spec.partition(":")[2])
    return absent


def read_files_at_ref(shell: Shell, ref: str, paths: list[str]) -> dict[str, str]:
    """Return {path: text} for ``paths`` at ``ref`` using one ``cat-file --batch``."""
    if not paths:
        return {}
    cp = subprocess.run(
        ["git", "cat-file", "--batch"],
        cwd=shell.cwd,
        input="".join(f"{ref}:{p}\n" for p in paths).encode("utf-8"),
        capture_output=True,
        check=False,
    )
    data = cp.stdout
    out: dict[str, str] = {}
    pos = 0
    for path in paths:
        nl = data.find(b"\n", pos)
        if nl < 0:
            break
        header = data[pos:nl].decode("utf-8", "replace")
        pos = nl + 1
        parts = header.split()
        if len(parts) < 3 or parts[-1] == "missing":
            continue
        size = int(parts[2])
        out[path] = data[pos : pos + size].decode("utf-8", "replace")
        pos += size + 1  # trailing newline after the blob
    return out


def grep_symbols(shell: Shell, base: str, symbols: list[str]) -> dict[str, list[str]]:
    """Map each symbol to the sorted list of main paths containing it as a word."""
    found: dict[str, set[str]] = {s: set() for s in symbols}
    ordered = sorted(found)
    for i in range(0, len(ordered), GREP_CHUNK):
        chunk = ordered[i : i + GREP_CHUNK]
        pattern = "(" + "|".join(re.escape(sym) for sym in chunk) + ")"
        cp = shell.git(
            "grep",
            "-n",
            "-w",
            "-I",
            "-E",
            pattern,
            base,
            "--",
            "src",
            "tests",
            "rust_core",
        )
        matchers = [
            (
                sym,
                re.compile(
                    r"(?<![A-Za-z0-9_$])" + re.escape(sym) + r"(?![A-Za-z0-9_$])"
                ),
            )
            for sym in chunk
        ]
        for line in cp.stdout.splitlines():
            # <ref>:<path>:<lineno>:<content>
            rest = line.partition(":")[2]
            path, _, rest = rest.partition(":")
            content = rest.partition(":")[2]
            for sym, rx in matchers:
                if rx.search(content):
                    found[sym].add(path)
    return {sym: sorted(paths) for sym, paths in found.items()}


def audit_missing_files(
    shell: Shell, base: str, ref: str, paths: list[str]
) -> list[dict[str, Any]]:
    """Classify every added-but-absent file (batched git access)."""
    symbol_paths = [
        p for p in paths if p.endswith(SYMBOL_EXTENSIONS) and not is_obsolete_path(p)
    ]
    sources = read_files_at_ref(shell, ref, symbol_paths)
    symbols_by_path = {p: extract_symbols(src, p) for p, src in sources.items()}
    all_symbols = sorted({s for syms in symbols_by_path.values() for s in syms})
    hits = grep_symbols(shell, base, all_symbols)
    records: list[dict[str, Any]] = []
    for path in sorted(paths):
        symbols = symbols_by_path.get(path, [])
        found = [{"symbol": s, "paths": hits[s][:10]} for s in symbols if hits.get(s)]
        records.append(
            {
                "path": path,
                "classification": classify_file(path, symbols, found),
                "symbols_checked": symbols,
                "symbols_found_on_main": found,
            }
        )
    return records


def audit_pr(shell: Shell, repo: str, base: str, number: int) -> dict[str, Any]:
    meta, error = fetch_pr(shell, repo, number)
    record: dict[str, Any] = {
        "number": number,
        "title": "",
        "state": "",
        "url": f"https://github.com/{repo}/pull/{number}",
        "head_ref": "",
        "head_oid": "",
        "base_ref": "",
        "base_ref_on_origin": False,
        "merged_at": None,
        "closed_at": None,
        "diff_ref_used": None,
        "diff_ref_sha": None,
        "head_oid_matches_diff_ref": None,
        "reachable": False,
        "reason": "",
        "error": error,
        "diffstat_summary": "",
        "counts": {"added": 0, "modified": 0, "deleted": 0},
        "missing_files": [],
        "groups": [],
    }
    if meta is None:
        record["reason"] = f"gh pr view failed: {error}"
        return record
    record.update(
        {
            "title": meta.get("title", ""),
            "state": meta.get("state", ""),
            "url": meta.get("url", record["url"]),
            "head_ref": meta.get("headRefName", ""),
            "head_oid": meta.get("headRefOid", ""),
            "base_ref": meta.get("baseRefName", ""),
            "merged_at": meta.get("mergedAt"),
            "closed_at": meta.get("closedAt"),
        }
    )
    record["base_ref_on_origin"] = bool(record["base_ref"]) and shell.git_ok(
        "rev-parse", "--verify", "--quiet", f"origin/{record['base_ref']}"
    )
    ref, reason = resolve_ref(shell, record["head_ref"], record["head_oid"])
    if ref is None:
        record["reason"] = reason
        return record
    record["diff_ref_used"] = ref
    record["diff_ref_sha"] = shell.git("rev-parse", ref).stdout.strip()
    record["head_oid_matches_diff_ref"] = record["diff_ref_sha"] == record["head_oid"]
    record["reachable"] = True
    record["diffstat_summary"] = diffstat_summary(shell, base, ref)
    statuses = name_status(shell, base, ref)
    record["counts"] = {
        "added": len(statuses["A"]),
        "modified": len(statuses["M"]),
        "deleted": len(statuses["D"]),
    }
    truly_missing = missing_on_base(shell, base, statuses["A"])
    print(
        f"#{number}: {len(statuses['A'])} added, {len(truly_missing)} absent from {base}",
        file=sys.stderr,
        flush=True,
    )
    record["missing_files"] = audit_missing_files(shell, base, ref, truly_missing)
    record["groups"] = group_files(record["missing_files"])
    return record


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prs", type=int, nargs="+", default=DEFAULT_PRS)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--base", default=DEFAULT_BASE, help="branch name on origin")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--no-fetch", action="store_true", help="skip git fetch origin")
    parser.add_argument("--dry-run", action="store_true", help="print counts only")
    return parser.parse_args(argv)


def build_result(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    shell = Shell(root)
    if not args.no_fetch:
        shell.git("fetch", "origin")
    base = f"origin/{args.base}"
    base_sha = shell.git("rev-parse", base).stdout.strip()
    prs = [audit_pr(shell, args.repo, base, n) for n in args.prs]
    prs.sort(key=lambda p: p["number"])
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _dt.datetime.now(_dt.UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "repo": args.repo,
        "base": {"ref": base, "sha": base_sha},
        "prs": prs,
        "totals": compute_totals(prs),
    }


def write_outputs(result: dict[str, Any], out_json: Path, out_md: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(result, fh, indent=2, sort_keys=True, ensure_ascii=False)
        fh.write("\n")
    with out_md.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(render_markdown(result))
        if not render_markdown(result).endswith("\n"):
            fh.write("\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    result = build_result(args, root)
    if args.dry_run:
        for pr in result["prs"]:
            fb = Counter(f["classification"] for f in pr["missing_files"])
            print(
                f"#{pr['number']} reachable={pr['reachable']} "
                f"missing={fb[CLASS_MISSING]} landed={fb[CLASS_LANDED]} "
                f"obsolete={fb[CLASS_OBSOLETE]}"
            )
        print(json.dumps(result["totals"], sort_keys=True))
        return 0
    out_json = args.out_json if args.out_json.is_absolute() else root / args.out_json
    out_md = args.out_md if args.out_md.is_absolute() else root / args.out_md
    write_outputs(result, out_json, out_md)
    print(f"wrote {out_json} and {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
