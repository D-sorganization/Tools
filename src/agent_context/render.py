"""Deterministic portable documentation and an accessible offline browser."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path, PurePosixPath
from string import Template
from typing import Any
from urllib.parse import quote

from .catalog import Catalog, Component
from .paths import CatalogError, read_text, safe_path


def link(path: str) -> str:
    """Link from docs/agent_context to an original repository-relative source."""
    return "../../" + quote(path, safe="/")


def inventories(catalog: Catalog) -> list[dict[str, Any]]:
    """Read mechanical records from existing JSON registries without copying."""
    result = []
    for entry in catalog.inventories:
        try:
            value: Any = json.loads(read_text(safe_path(catalog.root, entry["path"])))
            for segment in entry["key"].split("."):
                value = value[segment]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise CatalogError(
                f"Invalid inventory {entry['path']}:{entry['key']}"
            ) from exc
        if not isinstance(value, (list, dict)):
            raise CatalogError(f"Inventory must resolve to records: {entry['path']}")
        result.append({**entry, "count": len(value), "records": value})
    return result


def _component_lines(catalog: Catalog, c: Component) -> list[str]:
    """Render one component with original-source and reverse-consumer links."""
    lines: list[str] = []
    lines += [
        f"### {c.title}",
        "",
        f"ID: `{c.id}` · Owner: {c.owner} · Status: {c.status}",
        "",
        c.summary,
        "",
    ]
    for label, paths in (
        ("Sources", c.sources),
        ("Documentation", c.documentation),
        ("Tests", c.tests),
    ):
        refs = ", ".join(f"[{PurePosixPath(p).name}]({link(p)})" for p in paths)
        lines.append(f"- **{label}:** {refs}")
    if c.entrypoints:
        lines.append(
            "- **Public Interfaces:** "
            + ", ".join(
                f"`{r.symbol}` in [{r.path}]({link(r.path)})" for r in c.entrypoints
            )
        )
    consumers = [r.consumer for r in catalog.relations if r.provider == c.id]
    providers = [r.provider for r in catalog.relations if r.consumer == c.id]
    lines += [
        f"- **Consumers:** {', '.join(consumers) or 'None registered'}",
        f"- **Providers:** {', '.join(providers) or 'None registered'}",
        "",
    ]
    return lines


def markdown(catalog: Catalog, state: dict[str, Any]) -> str:
    """Render authoritative source links and human-authored semantic edges."""
    lines = [
        f"# {catalog.repository} Agent Context",
        "",
        "Generated from catalog.json and current source evidence. "
        "Edit the sources and regenerate.",
        "",
        f"Source fingerprint: `{state['digest']}`.",
        "",
        "This map covers registered components. A source match is not "
        "scientific approval or proof that tests passed.",
        "",
        "[Searchable Offline Browser](index.html) · [Catalog](catalog.json) · "
        "[Boundary Reviews](reviews.json)",
        "",
        "## Components",
        "",
    ]
    for component in catalog.components:
        lines += _component_lines(catalog, component)
    lines += [
        "## Integration Contracts",
        "",
        "| Provider | Consumer | Interaction | Contract |",
        "| --- | --- | --- | --- |",
    ]
    for relation in catalog.relations:
        lines.append(
            f"| {relation.provider} | {relation.consumer} | {relation.kind} | "
            f"[{relation.id}]({link(relation.contract)}) |"
        )
    lines += ["", "```mermaid", "flowchart LR"]
    names = {c.id: f"n{i}" for i, c in enumerate(catalog.components)}
    for c in catalog.components:
        label = re.sub(r"[^\w .:/-]", " ", c.title)
        lines.append(f'    {names[c.id]}["{label}"]')
    for relation in catalog.relations:
        label = re.sub(r"[^\w .:/-]", " ", relation.kind)
        lines.append(
            f'    {names[relation.provider]} -->|"{label}"| {names[relation.consumer]}'
        )
    lines += ["```", "", "## Existing Inventories", ""]
    for item in inventories(catalog):
        lines.append(
            f"- [{item['title']}]({link(item['path'])}): {item['count']} "
            f"records at `{item['key']}`; registry remains authoritative."
        )
    lines += [
        "",
        "## Provenance and Limits",
        "",
        f"- {len(state['files'])} source files hashed with SHA-256; "
        "UTF-8 line endings normalized.",
        "- Generated documents omit absolute paths and commit IDs "
        "to remain reproducible across worktrees.",
        "- Live CLI/MCP results include checkout identity and current revision.",
        "- Read integration contracts and their tests before modifying a boundary.",
        "- Review declarations never substitute for executing validation "
        "or scientific approval.",
        "",
    ]
    return "\n".join(lines)


def browser(catalog: Catalog, state: dict[str, Any]) -> str:
    """Render readable HTML with native links and progressively enhanced search."""
    esc = html.escape
    cards = []
    for c in catalog.components:
        links = "".join(
            f'<li><a href="{link(p)}">{esc(p)}</a></li>'
            for p in (*c.documentation, *c.sources, *c.tests)
        )
        edges = []
        for r in catalog.relations:
            if c.id not in (r.provider, r.consumer):
                continue
            peer = r.consumer if r.provider == c.id else r.provider
            direction = "Used by" if r.provider == c.id else "Uses"
            edges.append(
                f'<li>{direction} <a href="#{peer}">{esc(peer)}</a>: '
                f'{esc(r.kind)} · <a href="{link(r.contract)}">Contract</a></li>'
            )
        cards.append(
            f'<article id="{c.id}"><h2>{esc(c.title)}</h2><p class="meta">'
            f"{esc(c.owner)} · {esc(c.status)} · {esc(c.id)}</p>"
            f"<p>{esc(c.summary)}</p><h3>Sources and Evidence</h3><ul>{links}</ul>"
            "<h3>Interactions</h3><ul>"
            f"{''.join(edges) or '<li>None registered</li>'}</ul></article>"
        )
    template = Template(
        Path(__file__).with_name("browser.html").read_text(encoding="utf-8")
    )
    return template.substitute(
        title=esc(catalog.repository.replace("_", " ")),
        count=len(cards),
        cards="".join(cards),
        digest=state["digest"],
    )
