"""Golden Q&A evaluation for knowledge packs (#5347, RM#1772 K4).

Measures retrieval quality (recall@k, Mean Reciprocal Rank, must-not-source
cleanliness) across a curated golden set of representative questions. Follows
the Runner_Dashboard routing_eval pattern.

Stdlib + PyYAML only: vendorable by Runner_Dashboard unchanged.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from .pack import KnowledgePack, Passage


@dataclass(frozen=True)
class GoldenQACase:
    """One golden Q&A evaluation question with expected ground-truth sources."""

    question: str
    expected_source: tuple[str, ...]
    must_not_source: tuple[str, ...] = ()
    id: str = ""
    category: str = "default"
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Convert case to plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], default_id: str = "") -> GoldenQACase:
        """Construct and validate a case from a raw mapping."""
        if not isinstance(data, Mapping):
            raise ValueError(f"case must be a mapping, got {type(data).__name__}")
        question = data.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("case 'question' must be a non-empty string")

        raw_expected = data.get("expected_source")
        if raw_expected is None:
            raise ValueError(f"case {question!r} missing 'expected_source'")
        expected: tuple[str, ...]
        if isinstance(raw_expected, str):
            expected = (raw_expected.strip(),)
        elif isinstance(raw_expected, (list, tuple)):
            expected = tuple(str(s).strip() for s in raw_expected if str(s).strip())
        else:
            raise ValueError("expected_source must be a string or list of strings")
        if not expected:
            raise ValueError("expected_source must contain at least one non-empty glob")

        raw_must_not = data.get("must_not_source", ())
        must_not: tuple[str, ...]
        if isinstance(raw_must_not, str):
            must_not = (raw_must_not.strip(),)
        elif isinstance(raw_must_not, (list, tuple)):
            must_not = tuple(str(s).strip() for s in raw_must_not if str(s).strip())
        else:
            raise ValueError("must_not_source must be a string or list of strings")

        case_id = str(data.get("id") or default_id).strip()
        category = str(data.get("category") or "default").strip()
        tags = tuple(str(t).strip() for t in data.get("tags") or () if str(t).strip())

        return cls(
            question=question.strip(),
            expected_source=expected,
            must_not_source=must_not,
            id=case_id,
            category=category,
            tags=tags,
        )


@dataclass(frozen=True)
class CaseEvalResult:
    """Outcome of evaluating one golden test question against a knowledge pack."""

    case: GoldenQACase
    passed: bool
    hit: bool
    reciprocal_rank: float
    first_hit_rank: int | None
    retrieved_citations: tuple[str, ...]
    retrieved_sources: tuple[str, ...]
    must_not_violations: tuple[str, ...] = ()
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert result to plain dictionary."""
        return {
            "case_id": self.case.id,
            "question": self.case.question,
            "category": self.case.category,
            "passed": self.passed,
            "hit": self.hit,
            "reciprocal_rank": self.reciprocal_rank,
            "first_hit_rank": self.first_hit_rank,
            "retrieved_citations": list(self.retrieved_citations),
            "retrieved_sources": list(self.retrieved_sources),
            "must_not_violations": list(self.must_not_violations),
            "latency_ms": self.latency_ms,
        }


@dataclass(frozen=True)
class EvalSummary:
    """Overall and category-level knowledge retrieval evaluation metrics."""

    total_cases: int
    passed_cases: int
    failed_cases: int
    recall_at_k: float
    mrr: float
    k: int
    results: tuple[CaseEvalResult, ...]
    category_metrics: Mapping[str, dict[str, Any]] = field(default_factory=dict)
    must_not_violation_count: int = 0
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert evaluation summary to plain dictionary."""
        return {
            "total_cases": self.total_cases,
            "passed_cases": self.passed_cases,
            "failed_cases": self.failed_cases,
            "recall_at_k": self.recall_at_k,
            "mrr": self.mrr,
            "k": self.k,
            "must_not_violation_count": self.must_not_violation_count,
            "category_metrics": dict(self.category_metrics),
            "timestamp": self.timestamp,
            "results": [r.to_dict() for r in self.results],
        }

    def to_markdown(self) -> str:
        """Format an executive markdown table summary."""
        lines = [
            "## Knowledge Pack Evaluation Report",
            f"**Timestamp:** `{self.timestamp}`  ",
            f"**Total Cases:** `{self.total_cases}`  ",
            f"**Recall@{self.k}:** `{self.recall_at_k:.1%}` "
            f"({self.passed_cases}/{self.total_cases})  ",
            f"**MRR:** `{self.mrr:.3f}`  ",
            f"**Must-Not Violations:** `{self.must_not_violation_count}`  ",
            "",
            "### Category Performance Breakdown",
            "| Category | Total | Passed | Recall@k | MRR |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ]
        for cat, met in sorted(self.category_metrics.items()):
            acc_str = f"{met['recall']:.1%}" if met["total"] > 0 else "N/A"
            mrr_str = f"{met['mrr']:.3f}" if met["total"] > 0 else "N/A"
            row = (
                f"| `{cat}` | {met['total']} | {met['passed']} | "
                f"{acc_str} | {mrr_str} |"
            )
            lines.append(row)

        failures = [r for r in self.results if not r.passed]
        if failures:
            lines.extend(
                [
                    "",
                    "### Failure Diagnostics",
                    "| ID | Question | Expected | First Retrieved | Violations |",
                    "| :--- | :--- | :--- | :--- | :--- |",
                ]
            )
            for f in failures[:15]:
                q_snip = f.case.question[:36] + (
                    "..." if len(f.case.question) > 36 else ""
                )
                exp_str = ", ".join(f.case.expected_source)
                first_ret = f.retrieved_sources[0] if f.retrieved_sources else "(none)"
                viol_str = ", ".join(f.must_not_violations) or "-"
                row = (
                    f"| `{f.case.id}` | {q_snip} | `{exp_str}` | "
                    f"`{first_ret}` | `{viol_str}` |"
                )
                lines.append(row)

        return "\n".join(lines)


def load_golden_set(path: Path | str) -> tuple[GoldenQACase, ...]:
    """Read and validate a golden Q&A dataset from YAML."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"golden dataset not found: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    items: list[Any]
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, Mapping):
        candidates = raw.get("cases") or raw.get("questions") or raw.get("items")
        if isinstance(candidates, list):
            items = candidates
        else:
            raise ValueError(
                f"{path}: mapping must contain 'cases' or 'questions' list"
            )
    else:
        raise ValueError(f"{path}: golden dataset must be a list or mapping")

    cases: list[GoldenQACase] = []
    for idx, item in enumerate(items):
        cases.append(GoldenQACase.from_dict(item, default_id=f"case-{idx + 1}"))
    return tuple(cases)


def _matches_any_glob(source: str, repo: str, patterns: Sequence[str]) -> bool:
    """Check if relative source path or repo:source matches any glob pattern."""
    normalized_source = source.replace("\\", "/")
    combined = f"{repo}:{normalized_source}"
    for pat in patterns:
        norm_pat = pat.replace("\\", "/")
        if fnmatch.fnmatch(normalized_source, norm_pat):
            return True
        if fnmatch.fnmatch(combined, norm_pat):
            return True
        # Also match bare filename if pattern has no slashes
        if "/" not in norm_pat and fnmatch.fnmatch(
            Path(normalized_source).name, norm_pat
        ):
            return True
    return False


def evaluate_case(
    case: GoldenQACase,
    pack: KnowledgePack,
    k: int = 8,
    include_superseded: bool = False,
    embedder: Any | None = None,
) -> CaseEvalResult:
    """Evaluate a single golden Q&A case against a knowledge pack."""
    t0 = time.perf_counter()
    # Support optional embedder argument when hybrid ranking is active
    if embedder is not None:
        passages: list[Passage] = pack.search(
            case.question, k=k, include_superseded=include_superseded, embedder=embedder
        )
    else:
        passages = pack.search(
            case.question, k=k, include_superseded=include_superseded
        )
    latency_ms = (time.perf_counter() - t0) * 1000.0

    retrieved_citations = tuple(p.citation for p in passages)
    retrieved_sources = tuple(p.source for p in passages)

    must_not_violations: list[str] = []
    first_hit_rank: int | None = None

    for rank_idx, passage in enumerate(passages, start=1):
        # Check for must-not violations
        if case.must_not_source and _matches_any_glob(
            passage.source, passage.repo, case.must_not_source
        ):
            must_not_violations.append(passage.citation)
            continue

        # Check for expected hit
        if first_hit_rank is None and _matches_any_glob(
            passage.source, passage.repo, case.expected_source
        ):
            first_hit_rank = rank_idx

    hit = first_hit_rank is not None
    reciprocal_rank = (1.0 / first_hit_rank) if hit and first_hit_rank else 0.0
    passed = hit and (len(must_not_violations) == 0)

    return CaseEvalResult(
        case=case,
        passed=passed,
        hit=hit,
        reciprocal_rank=reciprocal_rank,
        first_hit_rank=first_hit_rank,
        retrieved_citations=retrieved_citations,
        retrieved_sources=retrieved_sources,
        must_not_violations=tuple(must_not_violations),
        latency_ms=latency_ms,
    )


def evaluate_pack(
    pack: KnowledgePack,
    cases: Sequence[GoldenQACase],
    k: int = 8,
    include_superseded: bool = False,
    embedder: Any | None = None,
) -> EvalSummary:
    """Evaluate a knowledge pack across all cases in a golden dataset."""
    if not isinstance(pack, KnowledgePack):
        raise TypeError("pack must be a KnowledgePack instance")
    if not isinstance(cases, Sequence):
        raise TypeError("cases must be a sequence of GoldenQACase")
    if k < 1:
        raise ValueError("k must be a positive integer")

    results: list[CaseEvalResult] = []
    cat_counts: dict[str, dict[str, Any]] = {}
    total_must_not_violations = 0

    for case in cases:
        res = evaluate_case(
            case, pack, k=k, include_superseded=include_superseded, embedder=embedder
        )
        results.append(res)
        total_must_not_violations += len(res.must_not_violations)

        cat = case.category
        if cat not in cat_counts:
            cat_counts[cat] = {"total": 0, "passed": 0, "hits": 0, "rr_sum": 0.0}
        cat_counts[cat]["total"] += 1
        if res.passed:
            cat_counts[cat]["passed"] += 1
        if res.hit:
            cat_counts[cat]["hits"] += 1
        cat_counts[cat]["rr_sum"] += res.reciprocal_rank

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    hits = sum(1 for r in results if r.hit)
    failed = total - passed

    recall_at_k = (hits / total) if total > 0 else 0.0
    mrr = (sum(r.reciprocal_rank for r in results) / total) if total > 0 else 0.0

    category_metrics: dict[str, dict[str, Any]] = {}
    for cat, data in sorted(cat_counts.items()):
        c_tot = data["total"]
        category_metrics[cat] = {
            "total": c_tot,
            "passed": data["passed"],
            "hits": data["hits"],
            "recall": (data["hits"] / c_tot) if c_tot > 0 else 0.0,
            "mrr": (data["rr_sum"] / c_tot) if c_tot > 0 else 0.0,
        }

    return EvalSummary(
        total_cases=total,
        passed_cases=passed,
        failed_cases=failed,
        recall_at_k=recall_at_k,
        mrr=mrr,
        k=k,
        results=tuple(results),
        category_metrics=category_metrics,
        must_not_violation_count=total_must_not_violations,
        timestamp=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def _eval_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m shared.python.ai.knowledge.eval",
        description="Evaluate knowledge-pack retrieval against a golden Q&A dataset",
    )
    parser.add_argument("pack", type=Path, help="path to built knowledge pack (.pack)")
    parser.add_argument(
        "golden", type=Path, help="path to golden Q&A dataset (.eval.yml)"
    )
    parser.add_argument(
        "-k", type=int, default=8, help="top-k passages to retrieve (default: 8)"
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=None,
        help="fail if recall@k is below this floor (0.0 to 1.0)",
    )
    parser.add_argument(
        "--min-mrr",
        type=float,
        default=None,
        help="fail if MRR is below this floor (0.0 to 1.0)",
    )
    parser.add_argument(
        "--baseline",
        type=float,
        default=None,
        help="recorded baseline recall@k threshold (alias for --min-recall)",
    )
    parser.add_argument(
        "--all", action="store_true", help="include superseded/retracted passages"
    )
    parser.add_argument(
        "--json", action="store_true", help="output json instead of markdown"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the eval CLI. Returns 0 on pass, 1 on threshold failure, 2 on input error."""
    parser = _eval_parser()
    args = parser.parse_args(argv)

    min_recall = args.min_recall if args.min_recall is not None else args.baseline

    try:
        pack = KnowledgePack.open(args.pack)
        cases = load_golden_set(args.golden)
        summary = evaluate_pack(pack, cases, k=args.k, include_superseded=args.all)
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"error: {exc}\n")
        return 2

    if args.json:
        sys.stdout.write(json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n")
    else:
        sys.stdout.write(summary.to_markdown() + "\n")

    if min_recall is not None and summary.recall_at_k < min_recall:
        sys.stderr.write(
            f"FAIL: Recall@{args.k} ({summary.recall_at_k:.3f}) < "
            f"threshold ({min_recall:.3f})\n"
        )
        return 1

    if args.min_mrr is not None and summary.mrr < args.min_mrr:
        sys.stderr.write(
            f"FAIL: MRR ({summary.mrr:.3f}) < threshold ({args.min_mrr:.3f})\n"
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
