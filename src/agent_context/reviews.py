"""Explicit boundary-review declarations tied to exact source evidence."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .catalog import Catalog, Relation
from .paths import CatalogError, file_hash, read_text, safe_path
from .workspace import digest

REVIEW_PATH = "docs/agent_context/reviews.json"


def inputs(catalog: Catalog, relation: Relation) -> dict[str, str]:
    """Include executable evidence and the documented contract in each review."""
    paths = sorted({relation.contract, *relation.inputs, *relation.tests})
    return {p: file_hash(safe_path(catalog.root, p)) for p in paths}


def fingerprint(relation: Relation, hashes: dict[str, str]) -> str:
    """Changing endpoints, test references or source bytes invalidates review."""
    return digest({"relation": asdict(relation), "inputs": hashes})


def load_reviews(root: Path) -> dict[str, Any]:
    """Absent review is incomplete evidence; malformed review is an error."""
    if not (root / REVIEW_PATH).exists():
        return {}
    try:
        data = json.loads(read_text(safe_path(root, REVIEW_PATH)))
    except json.JSONDecodeError as exc:
        raise CatalogError(f"Invalid review JSON: {exc}") from exc
    if (
        not isinstance(data, dict)
        or type(data.get("version")) is not int
        or data.get("version") != 1
        or not isinstance(data.get("reviews"), dict)
    ):
        raise CatalogError("Invalid review schema")
    return dict(data["reviews"])


def statuses(catalog: Catalog) -> list[dict[str, Any]]:
    """Return structural review validity, never imply that tests were executed."""
    records = load_reviews(catalog.root)
    unknown = set(records) - {r.id for r in catalog.relations}
    if unknown:
        raise CatalogError(f"Review references unknown relations: {sorted(unknown)}")
    result = []
    for relation in catalog.relations:
        hashes = inputs(catalog, relation)
        record = records.get(relation.id, {})
        if not isinstance(record, dict):
            raise CatalogError(f"Invalid review record: {relation.id}")
        reason = record.get("rationale", "")
        verified = (
            record.get("fingerprint") == fingerprint(relation, hashes)
            and record.get("inputs") == hashes
            and isinstance(reason, str)
            and len(reason.strip()) >= 20
        )
        result.append(
            {
                "id": relation.id,
                "verified": verified,
                "rationale": reason,
                "meaning": (
                    "review declaration matches source; "
                    "test execution is separate evidence"
                ),
            }
        )
    return result
