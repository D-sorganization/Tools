"""Focused live retrieval and deterministic checks over a semantic catalog."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from . import render, reviews
from .catalog import Catalog, CatalogError, Component, load_catalog, symbol_lines
from .paths import read_text, safe_path
from .workspace import snapshot


def write_json(path: Path, data: Any) -> None:
    """Atomically replace a generated file so readers never see partial JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
        ) as stream:
            name = stream.name
            json.dump(data, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(name, path)
    finally:
        if name and Path(name).exists():
            Path(name).unlink()


def gather_sources(
    catalog: Catalog, component: Component, state: dict[str, Any]
) -> list[dict[str, Any]]:
    """Select bounded excerpts and verify each against its snapshot hash."""
    relationships = [
        r for r in catalog.relations if component.id in (r.provider, r.consumer)
    ]
    sources = []
    selections = [
        (ref.path, *symbol_lines(catalog.root, ref))
        for ref in component.entrypoints[:4]
    ]
    selections += [
        (p, 1, 80)
        for p in dict.fromkeys(
            (*[r.contract for r in relationships], *component.documentation)
        )
    ][:4]
    for path, start, end in selections:
        content = read_text(safe_path(catalog.root, path))
        if hashlib.sha256(content.encode()).hexdigest() != state["files"][path]:
            raise CatalogError(f"Source changed during retrieval: {path}; retry")
        lines = content.splitlines()
        end = min(end, len(lines), start + 79)
        excerpt = "\n".join(lines[start - 1 : end])
        sources.append(
            {
                "path": path,
                "start_line": start,
                "end_line": end,
                "excerpt": excerpt,
                "sha256": state["files"][path],
            }
        )
    return sources


class ContextService:
    """Always resolve evidence from the requested worktree, with bounded output."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def _current(self) -> tuple[Catalog, dict[str, Any], str]:
        catalog = load_catalog(self.root)
        state = snapshot(catalog)
        cache = self.root / ".codemap/context.json"
        if not cache.resolve().is_relative_to(self.root):
            raise CatalogError("Context cache path escapes checkout")
        old: Any = None
        if cache.exists():
            try:
                old = json.loads(read_text(cache))
            except (CatalogError, json.JSONDecodeError, OSError):
                old = None
        cache_state = "verified" if old == state else "replaced"
        if old != state:
            try:
                write_json(cache, state)
            except OSError:
                cache_state = "unavailable"
        return catalog, state, cache_state

    def search(self, query: str, *, limit: int = 5) -> dict[str, Any]:
        """Find component descriptions and paths; return no invented matches."""
        if (
            not isinstance(query, str)
            or not query.strip()
            or type(limit) is not int
            or not 1 <= limit <= 20
        ):
            raise CatalogError(
                "A nonempty query and limit between 1 and 20 are required"
            )
        terms = set(re.findall(r"[a-z0-9_]+", query.lower()))
        if not terms:
            raise CatalogError("Query must contain searchable words")
        catalog, state, _ = self._current()
        ranked: list[dict[str, Any]] = []
        for c in catalog.components:
            words = re.findall(
                r"[a-z0-9_]+",
                " ".join(
                    (
                        c.id,
                        c.title,
                        c.summary,
                        *c.tags,
                        *c.sources,
                        *[r.symbol for r in c.entrypoints],
                    )
                ).lower(),
            )
            score = sum(words.count(term) for term in terms)
            if score:
                ranked.append(
                    {"id": c.id, "title": c.title, "summary": c.summary, "score": score}
                )
        ranked.sort(key=lambda hit: (-hit["score"], hit["id"]))
        return {
            "matches": ranked[:limit],
            "scope": "registered component descriptions and source paths",
            "provenance": self._provenance(state),
        }

    def status(self) -> dict[str, Any]:
        """Report live coverage, dependency identity and boundary review status."""
        catalog, state, cache_state = self._current()
        return {
            "provenance": self._provenance(state),
            "cache_state": cache_state,
            "components": len(catalog.components),
            "contract_reviews": reviews.statuses(catalog),
        }

    @staticmethod
    def _provenance(state: dict[str, Any]) -> dict[str, Any]:
        return {
            key: state[key]
            for key in (
                "version",
                "root",
                "commit",
                "implementation",
                "digest",
                "dependencies",
                "coverage",
            )
        }

    def context(self, component_id: str, *, max_chars: int = 16_000) -> dict[str, Any]:
        """Return cited current interfaces, contracts, consumers and test paths."""
        if type(max_chars) is not int or not 2_000 <= max_chars <= 64_000:
            raise CatalogError("Context max_chars must be between 2000 and 64000")
        catalog, state, cache_state = self._current()
        component = next((c for c in catalog.components if c.id == component_id), None)
        if component is None:
            raise CatalogError(f"Unknown component: {component_id}")
        relationships = [
            r for r in catalog.relations if component_id in (r.provider, r.consumer)
        ]
        review_states = [
            s
            for s in reviews.statuses(catalog)
            if s["id"] in {r.id for r in relationships}
        ]
        sources = gather_sources(catalog, component, state)
        dependency_verified = all(
            d["state"] == "verified" for d in state["dependencies"].values()
        )
        authority = (
            "source-verified; contracts-reviewed"
            if all(s["verified"] for s in review_states)
            else "source-verified; contract-review-required"
        )
        result = {
            "component": asdict(component),
            "consumers": sorted(
                {r.consumer for r in relationships if r.provider == component_id}
            ),
            "providers": sorted(
                {r.provider for r in relationships if r.consumer == component_id}
            ),
            "relations": [asdict(r) for r in relationships],
            "contract_reviews": review_states,
            "sources": sources,
            "authority": authority if dependency_verified else "dependency-unverified",
            "cache_state": cache_state,
            "provenance": self._provenance(state),
            "truncated": False,
        }
        while len(json.dumps(result, indent=2)) + 1 > max_chars and sources:
            sources.pop()
            result["truncated"] = True
        if len(json.dumps(result, indent=2)) + 1 > max_chars:
            raise CatalogError(
                "Component metadata exceeds context budget; "
                "narrow catalog records or increase max_chars"
            )
        return result

    def review(self, relation_id: str, rationale: str) -> None:
        """Record an explicit review declaration; never run evidence commands."""
        if not isinstance(rationale, str) or len(rationale.strip()) < 20:
            raise CatalogError(
                "A specific review rationale of at least 20 characters is required"
            )
        catalog = load_catalog(self.root)
        relation = next((r for r in catalog.relations if r.id == relation_id), None)
        if relation is None:
            raise CatalogError(f"Unknown relation: {relation_id}")
        records = reviews.load_reviews(self.root)
        hashes = reviews.inputs(catalog, relation)
        records[relation.id] = {
            "inputs": hashes,
            "fingerprint": reviews.fingerprint(relation, hashes),
            "rationale": rationale.strip(),
        }
        write_json(self.root / reviews.REVIEW_PATH, {"version": 1, "reviews": records})

    def check_reviews(self) -> list[str]:
        """Require review declarations to match the exact boundary evidence."""
        missing = [
            s["id"]
            for s in reviews.statuses(load_catalog(self.root))
            if not s["verified"]
        ]
        if missing:
            raise CatalogError(f"Boundary review required: {', '.join(missing)}")
        return []

    def render(self) -> None:
        """Regenerate views; do not refresh human review declarations."""
        catalog, state, _ = self._current()
        for name, content in self._outputs(catalog, state).items():
            target = self.root / "docs/agent_context" / name
            if not target.resolve().is_relative_to(self.root):
                raise CatalogError("Generated output path escapes checkout")
            target.write_text(content, encoding="utf-8", newline="\n")

    @staticmethod
    def _outputs(catalog: Catalog, state: dict[str, Any]) -> dict[str, str]:
        return {
            "README.md": render.markdown(catalog, state),
            "index.html": render.browser(catalog, state),
        }

    def check(self) -> list[str]:
        """Fail on stale views, missing evidence, unreviewed boundaries or pins."""
        catalog, state, _ = self._current()
        errors = []
        for name, expected in self._outputs(catalog, state).items():
            path = self.root / "docs/agent_context" / name
            if not path.is_file() or read_text(path) != expected:
                errors.append(f"stale generated view: {name}; run agent-context render")
        if any(d["state"] != "verified" for d in state["dependencies"].values()):
            errors.append("Pinned dependency checkout is unverified")
        errors += [
            f"stale boundary review: {s['id']}"
            for s in reviews.statuses(catalog)
            if not s["verified"]
        ]
        if errors:
            raise CatalogError("; ".join(errors))
        return []
