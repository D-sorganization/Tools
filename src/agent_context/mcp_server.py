"""Optional local stdio MCP adapter; no mutation tools or model calls."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from .service import ContextService


def build_server(root: Path, *, factory: Any = None) -> Any:
    """Bind one server to one worktree; keep MCP an optional installation."""
    if factory is None:
        from mcp.server.fastmcp import FastMCP

        factory = FastMCP
    server = factory("agent-context")
    service = ContextService(root)

    @server.tool()
    def search_components(query: str, limit: int = 5) -> dict[str, Any]:
        """Find registered subsystems, using current source-backed metadata."""
        return service.search(query, limit=limit)

    @server.tool()
    def get_component_context(
        component_id: str, max_chars: int = 16_000
    ) -> dict[str, Any]:
        """Read cited interfaces, integration contracts, consumers and test paths."""
        return service.context(component_id, max_chars=max_chars)

    @server.tool()
    def context_status() -> dict[str, Any]:
        """Inspect worktree identity, dependency pins and boundary-review status."""
        return service.status()

    return server


def main(argv: list[str] | None = None) -> int:
    """Run the same optional FastMCP transport used by existing CodeMap."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        server = build_server(args.root)
    except ImportError:
        sys.stderr.write(
            "agent-context-mcp requires the optional mcp package; "
            "the local CLI needs no extras.\n"
        )
        return 2
    server.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
