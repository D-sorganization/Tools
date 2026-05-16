"""MCP (Model Context Protocol) infrastructure.

Public exports for connecting to and managing external MCP servers
(stdio or HTTP) and aggregating their tools and resources into the local
``ToolRegistry``.

Public surface:
    - ``McpServerConfig`` / ``McpTransport`` (Pydantic config)
    - ``McpToolDescriptor`` / ``McpResourceDescriptor``
    - ``McpClient`` (single-server wrapper)
    - ``McpClientPool`` (aggregation across N servers)
    - ``load_mcp_servers`` (Claude-Desktop-compatible config loader)

External callers should reach MCP capabilities through the pool, never by
indexing into ``pool._clients`` directly (LOD).
"""

from __future__ import annotations

from src.shared.python.ai.mcp.client import McpClient
from src.shared.python.ai.mcp.config_loader import load_mcp_servers
from src.shared.python.ai.mcp.contracts import (
    McpResourceDescriptor,
    McpServerConfig,
    McpToolDescriptor,
    McpTransport,
)
from src.shared.python.ai.mcp.pool import McpClientPool

__all__ = [
    "McpClient",
    "McpClientPool",
    "McpResourceDescriptor",
    "McpServerConfig",
    "McpToolDescriptor",
    "McpTransport",
    "load_mcp_servers",
]
