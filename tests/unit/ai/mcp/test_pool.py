"""Tests for ``McpClientPool`` — lifecycle, aggregation, failure isolation."""

from __future__ import annotations

from typing import Any

import pytest

from src.shared.python.ai.mcp.contracts import McpServerConfig, McpToolDescriptor
from src.shared.python.ai.mcp.pool import McpClientPool


class FakeClient:
    """A drop-in fake for ``McpClient`` used to validate pool behavior."""

    def __init__(
        self,
        config: McpServerConfig,
        tools: list[McpToolDescriptor] | None = None,
        fail_on_connect: bool = False,
    ) -> None:
        self.config = config
        self._tools = tools or []
        self._connected = False
        self._fail_on_connect = fail_on_connect

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        if self._fail_on_connect:
            raise RuntimeError(f"boom for {self.config.name}")
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def list_tools(self) -> list[McpToolDescriptor]:
        if not self._connected:
            raise RuntimeError("not connected")
        return self._tools

    async def list_resources(self) -> list[Any]:
        return []

    async def call_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        return {"called": name, "args": args, "server": self.config.name}


@pytest.mark.asyncio
async def test_add_and_start_all() -> None:
    pool = McpClientPool()
    cfg = McpServerConfig(name="nb", transport="stdio", command="python")
    fake = FakeClient(
        cfg,
        tools=[McpToolDescriptor(name="search", description="s")],
    )
    pool.add_server(cfg, client=fake)
    await pool.start_all()
    assert pool.is_started is True
    assert fake.is_connected is True
    await pool.stop_all()
    assert pool.is_started is False
    assert fake.is_connected is False


@pytest.mark.asyncio
async def test_tools_aggregation_with_namespace() -> None:
    pool = McpClientPool()
    cfg_a = McpServerConfig(name="nb", transport="stdio", command="python")
    cfg_b = McpServerConfig(name="other", transport="stdio", command="python")
    fake_a = FakeClient(
        cfg_a, tools=[McpToolDescriptor(name="search", description="s")]
    )
    fake_b = FakeClient(
        cfg_b, tools=[McpToolDescriptor(name="search", description="s2")]
    )
    pool.add_server(cfg_a, client=fake_a)
    pool.add_server(cfg_b, client=fake_b)
    await pool.start_all()
    aggregated = await pool.tools()
    names = sorted(t["namespaced_name"] for t in aggregated)
    assert names == ["nb:search", "other:search"]
    sources = sorted(t["source"] for t in aggregated)
    assert sources == ["mcp:nb", "mcp:other"]
    await pool.stop_all()


@pytest.mark.asyncio
async def test_failure_isolation() -> None:
    pool = McpClientPool()
    cfg_ok = McpServerConfig(name="ok", transport="stdio", command="python")
    cfg_bad = McpServerConfig(name="bad", transport="stdio", command="python")
    fake_ok = FakeClient(
        cfg_ok, tools=[McpToolDescriptor(name="ping", description="p")]
    )
    fake_bad = FakeClient(cfg_bad, fail_on_connect=True)
    pool.add_server(cfg_ok, client=fake_ok)
    pool.add_server(cfg_bad, client=fake_bad)
    await pool.start_all()
    # OK server should still be connected even though bad one failed.
    assert fake_ok.is_connected is True
    aggregated = await pool.tools()
    assert len(aggregated) == 1
    assert aggregated[0]["namespaced_name"] == "ok:ping"
    await pool.stop_all()


@pytest.mark.asyncio
async def test_remove_server() -> None:
    pool = McpClientPool()
    cfg = McpServerConfig(name="nb", transport="stdio", command="python")
    fake = FakeClient(cfg)
    pool.add_server(cfg, client=fake)
    await pool.start_all()
    assert fake.is_connected is True
    await pool.remove_server("nb")
    assert fake.is_connected is False
    aggregated = await pool.tools()
    assert aggregated == []
    await pool.stop_all()


@pytest.mark.asyncio
async def test_tools_precondition_requires_started() -> None:
    pool = McpClientPool()
    with pytest.raises(RuntimeError, match="not started"):
        await pool.tools()


def test_add_duplicate_server_raises() -> None:
    pool = McpClientPool()
    cfg = McpServerConfig(name="nb", transport="stdio", command="python")
    pool.add_server(cfg, client=FakeClient(cfg))
    with pytest.raises(ValueError):
        pool.add_server(cfg, client=FakeClient(cfg))
