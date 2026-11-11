#!/usr/bin/env python3
"""Run the vendored Fleet Agent API conformance template (issue #6022)."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "fleet_agent_api_conformance"
    / "v1"
    / "template.json"
)


def _load_template(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _assert_payload(assertions: list[str], payload: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for item in assertions:
        if item.startswith("max_items:"):
            limit = int(item.split(":", 1)[1])
            calculators = payload.get("calculators")
            if isinstance(calculators, list) and len(calculators) > limit:
                failures.append(f"calculators length {len(calculators)} > {limit}")
            continue
        if item == "contract_version":
            if "contract_version" not in payload:
                failures.append("missing contract_version")
            continue
        if item == "request_schema":
            if "request_schema" not in payload:
                failures.append("missing request_schema")
            continue
        if item.startswith("status:"):
            want = item.split(":", 1)[1]
            if payload.get("status") != want:
                failures.append(f"status {payload.get('status')!r} != {want!r}")
            continue
        if item == "provenance.contract_version":
            prov = payload.get("provenance") or {}
            if not prov.get("contract_version"):
                failures.append("missing provenance.contract_version")
            continue
        if item == "structured_errors":
            errors = payload.get("errors")
            if not isinstance(errors, list) or not errors:
                failures.append("expected structured errors list")
            elif not isinstance(errors[0].get("field"), str):
                failures.append("expected error.field string")
    return failures


async def _run_stdio_checks(template: dict[str, Any], repo_root: Path) -> list[str]:
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError as exc:
        return [f"MCP SDK not installed: {exc}"]

    command = template["server"]["command"]
    if shutil.which(command) is None:
        command = sys.executable
        args = ["-m", "shared.python.calc_backend.mcp_server"]
    else:
        args = []

    server = StdioServerParameters(
        command=command,
        args=args,
        env={**os.environ, "PYTHONPATH": str(repo_root / "src")},
    )
    failures: list[str] = []
    async with stdio_client(server) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            listed = await session.list_tools()
            names = {tool.name for tool in listed.tools}
            required = set(template.get("required_tools", []))
            missing = required - names
            if missing:
                failures.append(f"missing tools: {sorted(missing)}")

            for check in template.get("checks", []):
                if check.get("kind") == "handshake":
                    continue
                if check.get("kind") == "list_tools":
                    continue
                tool = check["tool"]
                arguments = check.get("arguments", {})
                response = await session.call_tool(tool, arguments)
                if response.isError:
                    failures.append(f"{check['id']}: tool error")
                    continue
                text = response.content[0].text if response.content else "{}"
                payload = json.loads(text)
                failures.extend(
                    f"{check['id']}: {msg}" for msg in _assert_payload(check.get("assertions", []), payload)
                )
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--template",
        type=Path,
        default=TEMPLATE,
        help="Conformance template JSON path.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    template = _load_template(args.template)
    failures = asyncio.run(_run_stdio_checks(template, args.repo_root.resolve()))
    if failures:
        for line in failures:
            logger.error("%s", line)
        return 1
    logger.info("agent API conformance: %d checks passed", len(template.get("checks", [])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
