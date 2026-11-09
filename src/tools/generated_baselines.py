"""Generated baselines inventory, contract, and bounded views (Tools #5995)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from .spec_baseline_integrity import (
    compute_spec_changelog_anchors,
    verify_spec_changelog_integrity,
)

CANONICAL_INVENTORY: dict[str, Any] = {
    "schema_version": "1.0.0",
    "issue_reference": "Tools#5995",
    "baselines": [
        {
            "id": "benchmarks_baseline",
            "path": ".benchmarks/baseline.json",
            "size_bytes": 2124878,
            "sha256": (
                "fe6c7cea90e2cbb7c77687a23fbfbf8d251af4faf88f765a3981cede49e9da64"  # pragma: allowlist secret  # noqa: E501
            ),
            "classification": "benchmark_timing",
            "determinism": "environment_dependent_benchmark",
            "generator": (
                "pytest tests/benchmarks/ --benchmark-json=.benchmarks/baseline.json"
            ),
            "canonical_inputs": "tests/benchmarks/*.py",
            "schema_format": "pytest-benchmark schema v0.1",
            "update_policy": "Explicit reviewed updates on dedicated benchmark runner.",
            "review_owner": "Core Benchmarks / Fleet Maintainers",
            "test_consumers": ["pytest --benchmark-compare=.benchmarks/baseline.json"],
        },
        {
            "id": "sidekick_api_baseline",
            "path": "tests/sidekick_api_baseline.json",
            "size_bytes": 1109927,
            "sha256": (
                "e7df3e82b7f092d6eb3f8103eef88bdf5dc2ac91660972bf8846d052826f62a4"  # pragma: allowlist secret  # noqa: E501
            ),
            "classification": "api_ast_contract",
            "determinism": "deterministic",
            "generator": (
                "pytest tests/test_sidekick_public_api_stability.py"
                " --regenerate-api-baseline"
            ),
            "canonical_inputs": "src/shared/python/sidekick/ public AST",
            "schema_format": "JSON mapping module_path -> symbols",
            "update_policy": "Coordinated breaking change or API expansion protocol.",
            "review_owner": "Sidekick Architecture Team",
            "test_consumers": ["tests/test_sidekick_public_api_stability.py"],
        },
        {
            "id": "swing_sim_api_baseline",
            "path": "tests/api_baselines/swing_sim_api_baseline.json",
            "size_bytes": 1052117,
            "sha256": (
                "68976d347d1b58c845ccf1d6f3b94117ea9eab1deae734d1acfc0b98032f6e64"  # pragma: allowlist secret  # noqa: E501
            ),
            "classification": "api_ast_contract",
            "determinism": "deterministic",
            "generator": (
                "pytest tests/test_shared_package_api_stability.py"
                " --regenerate-api-baseline"
            ),
            "canonical_inputs": "src/shared/python/swing_sim/ public AST",
            "schema_format": "JSON mapping module_path -> symbols",
            "update_policy": "Coordinated breaking change or API expansion protocol.",
            "review_owner": "Swing Sim Team",
            "test_consumers": ["tests/test_shared_package_api_stability.py"],
        },
        {
            "id": "spec_changelog",
            "path": "SPEC.md",
            "size_bytes": 1048482,
            "sha256": (
                "a81759099c02deec4cf324b05fcfb714a1148a6f5410bc52da824bfb711c1324"  # pragma: allowlist secret  # noqa: E501
            ),
            "integrity_mode": "spec_changelog_collation",
            "spec_prefix_sha256": "b1d7f9549d2b2f0d22e09da6a03c61ca737b7377d96f2a7ed537c8b51265ea8c",  # pragma: allowlist secret  # noqa: E501
            "spec_suffix_sha256": "96d335e3479f721f9f2fc5158655528ce51fd0ee50c4446f76cf2b617e733db1",  # pragma: allowlist secret  # noqa: E501
            "changelog_header_sha256": "a0d4f0165a13091931b5706074da85c4a256046ef026dfc007d082ec44806711",  # pragma: allowlist secret  # noqa: E501
            "baseline_row_count": 1293,
            "baseline_rows_sha256": "2abb2f058b0c3d6ad626d4fd862cb1fe4c9106f1fe963f7eb99312ef9d00271e",  # pragma: allowlist secret  # noqa: E501
            "classification": "spec_changelog_ledger",
            "determinism": "deterministic",
            "generator": "python scripts/ci/check_spec_freshness.py",
            "canonical_inputs": "changes/*.md fragments",
            "schema_format": "Markdown specification and changelog ledger",
            "update_policy": (
                "Collation from reviewed changes/*.md fragments; refresh non-changelog "
                "anchors via --print-spec-baseline on reviewed SPEC.md edits."
            ),
            "review_owner": "Tools Maintainers",
            "test_consumers": ["scripts/ci/check_spec_freshness.py"],
        },
    ],
}


def find_repo_root(start: Path | str | None = None) -> Path:
    """Find repository root by looking upward for pyproject.toml or .git."""
    cur = Path(start).resolve() if start else Path.cwd().resolve()
    for p in [cur, *cur.parents]:
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p
    return cur


def load_inventory(
    repo_root: Path | str | None = None,
    custom_inventory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load the canonical or custom generated baselines inventory."""
    return (
        dict(custom_inventory)
        if custom_inventory is not None
        else dict(CANONICAL_INVENTORY)
    )


def build_spec_changelog_baseline(
    repo_root: Path | str | None = None, spec_rel_path: str = "SPEC.md"
) -> dict[str, Any]:
    """Build deterministic reviewed-anchor metadata for ``SPEC.md``."""
    root = find_repo_root(repo_root)
    base = _resolve_entry("spec_changelog", CANONICAL_INVENTORY)
    return {**base, **compute_spec_changelog_anchors(root, spec_rel_path)}


def verify_baseline_integrity(
    repo_root: Path | str | None = None,
    check_hashes: bool = True,
    inventory: dict[str, Any] | None = None,
) -> list[str]:
    """Verify that all inventoried baselines exist and satisfy integrity contracts."""
    root = find_repo_root(repo_root)
    errors: list[str] = []
    inv = load_inventory(root, custom_inventory=inventory)
    for entry in inv.get("baselines", []):
        rel = str(entry.get("path", ""))
        fp = root / rel
        if not fp.is_file():
            errors.append(f"File not found: {rel}")
            continue
        if not check_hashes:
            continue
        mode = str(entry.get("integrity_mode", "full_sha256"))
        if mode == "full_sha256":
            dig = hashlib.sha256(fp.read_bytes()).hexdigest()
            if dig != entry.get("sha256", ""):
                errors.append(
                    f"Hash mismatch for {rel}: expected {entry['sha256']}, got {dig}"
                )
        elif mode == "spec_changelog_collation":
            errors.extend(verify_spec_changelog_integrity(root, entry))
        else:
            errors.append(f"Unsupported integrity_mode for {rel}: {mode!r}")
    return errors


def _resolve_entry(path_or_id: str, inventory: dict[str, Any]) -> dict[str, Any]:
    for e in inventory.get("baselines", []):
        if e.get("id") == path_or_id or e.get("path") == path_or_id:
            return dict(e)
    raise KeyError(f"Baseline not found: {path_or_id}")


def summarize_baseline(
    path_or_id: str, repo_root: Path | str | None = None
) -> dict[str, Any]:
    """Generate a bounded context summary (<10 KB) for a generated baseline."""
    root = find_repo_root(repo_root)
    entry = _resolve_entry(path_or_id, load_inventory(root))
    fp, b_id = root / entry["path"], entry.get("id", "")
    summary: dict[str, Any] = {
        "id": b_id,
        "path": entry["path"],
        "size_bytes": fp.stat().st_size,
        "classification": entry.get("classification"),
        "determinism": entry.get("determinism"),
        "generator": entry.get("generator"),
    }
    if b_id == "benchmarks_baseline":
        raw = json.loads(fp.read_text(encoding="utf-8")).get("benchmarks", [])
        summary["entry_count"] = len(raw)
        summary["benchmarks"] = [
            {"name": b.get("name"), "mean": b.get("stats", {}).get("mean")} for b in raw
        ]
    elif b_id in {"sidekick_api_baseline", "swing_sim_api_baseline"}:
        raw_m = json.loads(fp.read_text(encoding="utf-8"))
        summary["entry_count"] = len(raw_m)
        summary["total_symbols"] = sum(
            len(c["symbols"])
            for c in raw_m.values()
            if isinstance(c, dict) and "symbols" in c
        )
        summary["sample_modules"] = list(raw_m.keys())[:5]
    elif b_id == "spec_changelog":
        lines = fp.read_text(encoding="utf-8").splitlines()
        summary["entry_count"] = summary["line_count"] = len(lines)
        summary["preview"] = lines[:10]
    return summary


def query_baseline(
    path_or_id: str, query: str, repo_root: Path | str | None = None
) -> dict[str, Any]:
    """Query a baseline for matching symbols/benchmarks with bounded context."""
    root = find_repo_root(repo_root)
    entry = _resolve_entry(path_or_id, load_inventory(root))
    fp, b_id, q = root / entry["path"], entry.get("id", ""), query.lower()

    if b_id == "benchmarks_baseline":
        b_raw = json.loads(fp.read_text(encoding="utf-8")).get("benchmarks", [])
        matches = []
        for b in b_raw:
            if q in b.get("name", "").lower() or q in b.get("fullname", "").lower():
                stats = dict(b.get("stats", {}))
                stats.pop("data", None)
                matches.append(
                    {
                        "name": b.get("name"),
                        "fullname": b.get("fullname"),
                        "stats": stats,
                    }
                )
        return {
            "id": b_id,
            "query": query,
            "match_count": len(matches),
            "matches": matches,
        }

    if b_id in {"sidekick_api_baseline", "swing_sim_api_baseline"}:
        raw_api = json.loads(fp.read_text(encoding="utf-8"))
        matched_mods: dict[str, Any] = {}
        for mod, content in raw_api.items():
            if q in mod.lower():
                matched_mods[mod] = content
            elif isinstance(content, dict) and "symbols" in content:
                syms = {s: d for s, d in content["symbols"].items() if q in s.lower()}
                if syms:
                    matched_mods[mod] = {
                        "__all__": content.get("__all__"),
                        "symbols": syms,
                    }
        return {
            "id": b_id,
            "query": query,
            "match_count": len(matched_mods),
            "matches": matched_mods,
        }

    if b_id == "spec_changelog":
        lines = fp.read_text(encoding="utf-8").splitlines()
        matched = [(i, line) for i, line in enumerate(lines, 1) if q in line.lower()][
            :50
        ]
        return {
            "id": b_id,
            "query": query,
            "match_count": len(matched),
            "matches": matched,
        }

    return {"id": b_id, "query": query, "match_count": 0, "matches": []}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for generated baselines verification and bounded views."""
    p = argparse.ArgumentParser(description="Baseline contract & bounded views.")
    p.add_argument("--repo-root", type=Path, default=None, help="Repository root.")
    p.add_argument("--check", action="store_true", help="Verify baseline integrity.")
    p.add_argument("--print-spec-baseline", action="store_true", help="Emit SPEC JSON.")
    p.add_argument("--summary", nargs="?", const="ALL", help="Bounded summary.")
    p.add_argument("--query", help="Filter symbols or benchmarks.")
    p.add_argument("--baseline", help="Target baseline ID.")
    args = p.parse_args(argv)

    if args.print_spec_baseline:
        out = build_spec_changelog_baseline(args.repo_root)
        sys.stdout.write(json.dumps(out, indent=2) + "\n")
        return 0
    if args.check:
        errs = verify_baseline_integrity(args.repo_root)
        for e in errs:
            sys.stderr.write(f"FAIL: {e}\n")
        if not errs:
            sys.stdout.write("PASS: all generated baselines match recorded hashes.\n")
        return 1 if errs else 0
    if args.summary:
        inv = load_inventory()
        res = (
            [summarize_baseline(b["id"]) for b in inv.get("baselines", [])]
            if args.summary == "ALL"
            else summarize_baseline(args.summary)
        )
        sys.stdout.write(json.dumps(res, indent=2) + "\n")
        return 0
    if args.query:
        if not args.baseline:
            sys.stderr.write("ERROR: --query requires --baseline <ID>\n")
            return 1
        sys.stdout.write(
            json.dumps(query_baseline(args.baseline, args.query), indent=2) + "\n"
        )
        return 0
    p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
