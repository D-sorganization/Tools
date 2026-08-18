from __future__ import annotations

import builtins
import sys
import types
from collections.abc import Callable
from typing import Any

from codemap import mcp_server
from tests.helpers.codemap_optional_deps import CODEMAP_DEPS_SKIP

# Scoped to this module only; a session-wide skip hook silenced the whole
# suite here once already (issue #4497).
pytestmark = CODEMAP_DEPS_SKIP


class _Dumpable:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def model_dump(self) -> dict[str, Any]:
        return self._payload


class _FakeFastMCP:
    instances: list[_FakeFastMCP] = []

    def __init__(self, name: str) -> None:
        self.name = name
        self.tools: dict[str, Callable[..., Any]] = {}
        self.run_calls = 0
        self.__class__.instances.append(self)

    def tool(self) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def _decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.tools[func.__name__] = func
            return func

        return _decorator

    def run(self) -> None:
        self.run_calls += 1


def _install_fake_mcp(monkeypatch) -> None:
    mcp_pkg = types.ModuleType("mcp")
    server_pkg = types.ModuleType("mcp.server")
    fastmcp_mod = types.ModuleType("mcp.server.fastmcp")
    fastmcp_mod.FastMCP = _FakeFastMCP

    monkeypatch.setitem(sys.modules, "mcp", mcp_pkg)
    monkeypatch.setitem(sys.modules, "mcp.server", server_pkg)
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fastmcp_mod)


def test_repo_root_uses_env_var_when_present(monkeypatch) -> None:
    monkeypatch.delenv("CODEMAP_REPO_ROOT", raising=False)
    assert mcp_server._repo_root() is None

    monkeypatch.setenv("CODEMAP_REPO_ROOT", "C:/repo")
    assert mcp_server._repo_root() == "C:/repo"


def test_build_fastmcp_returns_none_without_optional_dependency(monkeypatch) -> None:
    original_import = builtins.__import__

    def _blocked_import(name, globals_=None, locals_=None, fromlist=(), level=0):
        if name == "mcp.server.fastmcp":
            raise ImportError("mcp not installed")
        return original_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)

    assert mcp_server._build_fastmcp() is None


def test_main_reports_missing_mcp_dependency(monkeypatch, capsys) -> None:
    monkeypatch.setattr(mcp_server, "_build_fastmcp", lambda: None)

    assert mcp_server.main() == 2

    assert "codemap-mcp: the 'mcp' package is not installed" in capsys.readouterr().err


def test_main_runs_built_server(monkeypatch) -> None:
    server = _FakeFastMCP("codemap")
    monkeypatch.setattr(mcp_server, "_build_fastmcp", lambda: server)

    assert mcp_server.main([]) == 0

    assert server.run_calls == 1


def test_fastmcp_tools_delegate_to_codemap_api(monkeypatch) -> None:
    _FakeFastMCP.instances.clear()
    _install_fake_mcp(monkeypatch)
    monkeypatch.setenv("CODEMAP_REPO_ROOT", "C:/repo")
    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def _search_code(
        query: str,
        *,
        k: int,
        kind: str | None,
        repo_root: str | None,
    ) -> list[_Dumpable]:
        calls.append(
            (
                "search_code",
                (query,),
                {"k": k, "kind": kind, "repo_root": repo_root},
            )
        )
        return [_Dumpable({"qualified": "pkg.mod.target"})]

    def _get_symbol(qualified_name: str, *, repo_root: str | None) -> _Dumpable | None:
        calls.append(("get_symbol", (qualified_name,), {"repo_root": repo_root}))
        return _Dumpable({"qualified": qualified_name})

    def _who_calls(qualified_name: str, *, repo_root: str | None) -> list[_Dumpable]:
        calls.append(("who_calls", (qualified_name,), {"repo_root": repo_root}))
        return [_Dumpable({"qualified": "pkg.mod.caller"})]

    def _imports_of(path: str, *, repo_root: str | None) -> list[str]:
        calls.append(("imports_of", (path,), {"repo_root": repo_root}))
        return ["os", "sys"]

    def _repo_summary(*, repo_root: str | None) -> _Dumpable:
        calls.append(("repo_summary", (), {"repo_root": repo_root}))
        return _Dumpable({"files": 3, "symbols": 5})

    monkeypatch.setattr(mcp_server.api_mod, "search_code", _search_code)
    monkeypatch.setattr(mcp_server.api_mod, "get_symbol", _get_symbol)
    monkeypatch.setattr(mcp_server.api_mod, "who_calls", _who_calls)
    monkeypatch.setattr(mcp_server.api_mod, "imports_of", _imports_of)
    monkeypatch.setattr(mcp_server.api_mod, "repo_summary", _repo_summary)

    server = mcp_server._build_fastmcp()

    assert isinstance(server, _FakeFastMCP)
    assert server.name == "codemap"
    assert set(server.tools) == {
        "get_symbol",
        "imports_of",
        "repo_summary",
        "search_code",
        "who_calls",
    }
    assert server.tools["search_code"]("target", k=2, kind="function") == [
        {"qualified": "pkg.mod.target"}
    ]
    assert server.tools["get_symbol"]("pkg.mod.target") == {
        "qualified": "pkg.mod.target"
    }
    assert server.tools["get_symbol"]("missing") == {"qualified": "missing"}
    assert server.tools["who_calls"]("pkg.mod.target") == [
        {"qualified": "pkg.mod.caller"}
    ]
    assert server.tools["imports_of"]("pkg/mod.py") == ["os", "sys"]
    assert server.tools["repo_summary"]() == {"files": 3, "symbols": 5}
    assert calls == [
        (
            "search_code",
            ("target",),
            {"k": 2, "kind": "function", "repo_root": "C:/repo"},
        ),
        ("get_symbol", ("pkg.mod.target",), {"repo_root": "C:/repo"}),
        ("get_symbol", ("missing",), {"repo_root": "C:/repo"}),
        ("who_calls", ("pkg.mod.target",), {"repo_root": "C:/repo"}),
        ("imports_of", ("pkg/mod.py",), {"repo_root": "C:/repo"}),
        ("repo_summary", (), {"repo_root": "C:/repo"}),
    ]


def test_get_symbol_tool_returns_none_when_symbol_missing(monkeypatch) -> None:
    _install_fake_mcp(monkeypatch)
    monkeypatch.setattr(
        mcp_server.api_mod,
        "get_symbol",
        lambda _qualified_name, *, repo_root: None,
    )

    server = mcp_server._build_fastmcp()

    assert isinstance(server, _FakeFastMCP)
    assert server.tools["get_symbol"]("pkg.mod.missing") is None
