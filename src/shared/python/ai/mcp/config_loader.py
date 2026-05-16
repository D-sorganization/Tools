"""Claude-Desktop-compatible MCP server config loader.

The on-disk format follows Claude Desktop's ``mcp_servers.json``:

.. code-block:: json

    {
      "mcpServers": {
        "notebooklm": {
          "command": "python",
          "args": ["-m", "notebooklm_mcp"],
          "env": {"API_KEY": "${MY_API_KEY}"}
        },
        "remote": {
          "transport": "http",
          "url": "https://example.com/mcp"
        }
      }
    }

Behaviour:
    * ``${ENV_VAR}`` placeholders inside ``env`` values are expanded from the
      current process environment. Unresolved placeholders are left intact
      and a WARNING is logged so the operator can fix the missing var.
    * Invalid entries (Pydantic validation error, unsupported transport,
      missing required field) are skipped with a WARNING — they do *not*
      bring down the rest of the configuration.
    * Missing or malformed config files return an empty list rather than
      raising, so the caller can degrade gracefully when MCP is unconfigured.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.shared.python.ai.mcp.contracts import McpServerConfig

_LOG = logging.getLogger(__name__)
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
DEFAULT_CONFIG_PATH = Path.home() / ".upstreamdrift" / "mcp_servers.json"


def expand_env_vars(value: str) -> str:
    """Expand ``${ENV_VAR}`` placeholders. Unset vars are left as-is."""

    def _replace(match: re.Match[str]) -> str:
        var_name = match.group(1)
        env_value = os.environ.get(var_name)
        if env_value is None:
            _LOG.warning("MCP config: environment variable %s is not set", var_name)
            return match.group(0)
        return env_value

    return _ENV_PATTERN.sub(_replace, value)


def _expand_env_map(env: dict[str, Any]) -> dict[str, str]:
    expanded: dict[str, str] = {}
    for key, raw in env.items():
        if not isinstance(raw, str):
            _LOG.warning("MCP config: non-string env value for %s; coercing", key)
        expanded[key] = expand_env_vars(str(raw))
    return expanded


def parse_mcp_servers(raw: dict[str, Any]) -> list[McpServerConfig]:
    """Parse the deserialized JSON config dict into validated configs."""
    servers = raw.get("mcpServers") or {}
    if not isinstance(servers, dict):
        _LOG.warning("MCP config: 'mcpServers' is not a JSON object; ignoring")
        return []
    configs: list[McpServerConfig] = []
    for name, entry in servers.items():
        if not isinstance(entry, dict):
            _LOG.warning("MCP config: entry %r is not a JSON object; skipping", name)
            continue
        try:
            entry_env = entry.get("env", {})
            if not isinstance(entry_env, dict):
                _LOG.warning(
                    "MCP config: %s 'env' is not a JSON object; skipping",
                    name,
                )
                continue
            expanded_env = _expand_env_map(entry_env)
            config = McpServerConfig(
                name=name,
                transport=entry.get("transport", "stdio"),
                command=entry.get("command"),
                args=list(entry.get("args", [])),
                env=expanded_env,
                url=entry.get("url"),
                timeout_seconds=float(entry.get("timeoutSeconds", 30.0)),
            )
        except (ValidationError, ValueError, TypeError) as exc:
            _LOG.warning("MCP config: skipping invalid entry %r: %s", name, exc)
            continue
        configs.append(config)
    return configs


def load_mcp_servers(path: Path | None = None) -> list[McpServerConfig]:
    """Load and validate the MCP servers configuration file.

    Args:
        path: Path to the JSON config file. Defaults to
            ``~/.upstreamdrift/mcp_servers.json``.

    Returns:
        List of validated configs. Empty list if the file is missing or
        malformed (logged at WARNING).
    """
    target = path or DEFAULT_CONFIG_PATH
    if not target.exists():
        _LOG.debug("MCP config file %s does not exist; using empty config", target)
        return []
    try:
        raw_text = target.read_text(encoding="utf-8")
        raw_data = json.loads(raw_text)
    except (OSError, json.JSONDecodeError) as exc:
        _LOG.warning("MCP config: failed to read %s: %s", target, exc)
        return []
    if not isinstance(raw_data, dict):
        _LOG.warning("MCP config: top-level JSON is not an object; ignoring")
        return []
    return parse_mcp_servers(raw_data)
