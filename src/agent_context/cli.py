"""Local commands for retrieving and maintaining verified agent context."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .catalog import CatalogError
from .paths import read_text, safe_path
from .service import ContextService


def build_parser() -> argparse.ArgumentParser:
    """Build the local command interface without running repository code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Target Git worktree (default: current directory)",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    search = sub.add_parser("search", help="Find registered components")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=5)
    context = sub.add_parser("context", help="Read source-cited context and consumers")
    context.add_argument("component")
    context.add_argument("--max-chars", type=int, default=16_000)
    sub.add_parser("status", help="Inspect checkout, corpus and review validity")
    sub.add_parser(
        "check", help="Validate generated views, reviews and pinned dependencies"
    )
    sub.add_parser("render", help="Regenerate Markdown and offline HTML views")
    evaluation = sub.add_parser("evaluate", help="Measure curated navigation tasks")
    evaluation.add_argument("--tasks", default="docs/agent_context/navigation.json")
    review = sub.add_parser(
        "review", help="Record a reviewed boundary against its exact inputs"
    )
    review.add_argument("relation")
    review.add_argument("--rationale", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Serve local requests with explicit failure status."""
    parser = build_parser()
    args = parser.parse_args(argv)
    service = ContextService(args.root)
    try:
        if args.command == "search":
            result = service.search(args.query, limit=args.limit)
        elif args.command == "context":
            result = service.context(args.component, max_chars=args.max_chars)
        elif args.command == "status":
            result = service.status()
        elif args.command == "check":
            result = {"ok": True, "errors": service.check()}
        elif args.command == "render":
            service.render()
            result = {
                "ok": True,
                "generated": [
                    "docs/agent_context/README.md",
                    "docs/agent_context/index.html",
                ],
            }
        elif args.command == "evaluate":
            from .evaluation import evaluate

            result = evaluate(
                args.root, json.loads(read_text(safe_path(args.root, args.tasks)))
            )
        else:
            service.review(args.relation, args.rationale)
            result = {
                "ok": True,
                "reviewed": args.relation,
                "test_execution": "not performed",
            }
    except (CatalogError, OSError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"agent-context: {exc}\n")
        return 2
    sys.stdout.write(json.dumps(result, indent=2) + "\n")
    if args.command == "evaluate" and not result["passed"]:
        return 1
    return 0
