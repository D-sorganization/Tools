"""``python -m shared.python.ai.knowledge build|search|info``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from .manifest import ManifestError, load_manifest
from .pack import KnowledgePack, PackFormatError, build_pack


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI; returns 0 on success, 2 on bad input."""
    args = _parser().parse_args(argv)
    _tolerate_narrow_console()
    try:
        return int(args.handler(args))
    except (ManifestError, PackFormatError, FileNotFoundError, ValueError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2


def _tolerate_narrow_console() -> None:
    """Passages carry arbitrary Unicode; a cp1252 console must not crash output."""
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(errors="replace")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m shared.python.ai.knowledge")
    sub = parser.add_subparsers(required=True)

    build = sub.add_parser("build", help="index a manifest's sources into a pack")
    build.add_argument("manifest", type=Path)
    build.add_argument(
        "--root",
        action="append",
        default=[],
        metavar="REPO=PATH",
        help="checkout directory for a manifest repository (repeatable)",
    )
    build.add_argument("--out", type=Path, required=True)
    build.set_defaults(handler=_build)

    search = sub.add_parser("search", help="query a pack")
    search.add_argument("pack", type=Path)
    search.add_argument("query")
    search.add_argument("-k", type=int, default=8)
    search.add_argument(
        "--all", action="store_true", help="include superseded passages"
    )
    search.set_defaults(handler=_search)

    info = sub.add_parser("info", help="print what a pack was built from")
    info.add_argument("pack", type=Path)
    info.set_defaults(handler=_info)

    eval_cmd = sub.add_parser(
        "eval", help="evaluate a pack against a golden Q&A dataset"
    )
    eval_cmd.add_argument("pack", type=Path)
    eval_cmd.add_argument("golden", type=Path)
    eval_cmd.add_argument("-k", type=int, default=8)
    eval_cmd.add_argument("--min-recall", type=float, default=None)
    eval_cmd.add_argument("--min-mrr", type=float, default=None)
    eval_cmd.add_argument("--baseline", type=float, default=None)
    eval_cmd.add_argument(
        "--all", action="store_true", help="include superseded passages"
    )
    eval_cmd.add_argument("--json", action="store_true", help="output json format")
    eval_cmd.set_defaults(handler=_eval)
    return parser


def _build(args: argparse.Namespace) -> int:
    roots: dict[str, Path] = {}
    for item in args.root:
        name, sep, path = item.partition("=")
        if not sep or not name or not path:
            raise ValueError(f"--root must be REPO=PATH, got {item!r}")
        roots[name] = Path(path)
    info = build_pack(load_manifest(args.manifest), roots, args.out)
    sys.stdout.write(
        f"{info.pack_id}: {info.passages} passages from {info.files} files\n"
    )
    return 0


def _search(args: argparse.Namespace) -> int:
    hits = KnowledgePack.open(args.pack).search(
        args.query, k=args.k, include_superseded=args.all
    )
    for hit in hits:
        flag = "" if hit.status == "current" else f" [{hit.status}]"
        sys.stdout.write(f"{hit.score:7.3f}  {hit.citation}{flag}\n")
        sys.stdout.write(f"         {hit.title or '(untitled)'}: {hit.text[:160]!r}\n")
    return 0


def _info(args: argparse.Namespace) -> int:
    info = KnowledgePack.open(args.pack).info()
    sys.stdout.write(json.dumps(asdict(info), indent=2, sort_keys=True) + "\n")
    return 0


def _eval(args: argparse.Namespace) -> int:
    from .eval import evaluate_pack, load_golden_set

    min_recall = args.min_recall if args.min_recall is not None else args.baseline
    pack = KnowledgePack.open(args.pack)
    cases = load_golden_set(args.golden)
    summary = evaluate_pack(pack, cases, k=args.k, include_superseded=args.all)
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
