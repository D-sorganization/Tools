"""Reproducible navigation task definitions with measured retrieval costs."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

from .catalog import CatalogError, records, text
from .service import ContextService


def _task(service: ContextService, task: dict[str, Any]) -> dict[str, Any]:
    query = text(task.get("query"), "query")
    expected = text(task.get("component"), "component")
    source = text(task.get("source"), "source")
    started = time.perf_counter()
    matches = service.search(query, limit=3)["matches"]
    found = expected in {match["id"] for match in matches}
    detail = service.context(expected) if found else None
    sources = {item["path"] for item in detail["sources"]} if detail else set()
    consumer = task.get("consumer")
    correct = found and source in sources
    if consumer is not None:
        correct = correct and consumer in detail["consumers"] if detail else False
    return {
        "query": query,
        "component": expected,
        "correct": correct,
        "top_three": [match["id"] for match in matches],
        "expected_source": source,
        "characters": len(json.dumps(detail, indent=2)) + 1 if detail else 0,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
    }


def evaluate(root: Path, tasks: Any) -> dict[str, Any]:
    """Evaluate explicit expectations; never score success from nonempty text alone."""
    definitions = records(tasks, "navigation tasks")
    if not definitions or len(definitions) > 100:
        raise CatalogError("Provide between 1 and 100 navigation tasks")
    service = ContextService(root)
    before = service.status()["provenance"]
    results = [_task(service, task) for task in definitions]
    after = service.status()["provenance"]
    if before != after:
        raise CatalogError("Checkout changed during navigation evaluation; retry")
    correct = sum(task["correct"] for task in results)
    return {
        "passed": correct == len(results),
        "correct": correct,
        "total": len(results),
        "recall_at_three": correct / len(results),
        "median_ms": statistics.median(task["elapsed_ms"] for task in results),
        "max_characters": max(task["characters"] for task in results),
        "tasks": results,
        "provenance": after,
        "limits": (
            "Curated task accuracy and local retrieval cost; "
            "not an end-to-end development speed benchmark."
        ),
    }
