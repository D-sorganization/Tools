"""Pre-lifespan resource ownership tests for the Morris authority child."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rate_of_closure.application.morris import child
from rate_of_closure.application.morris.router import MorrisJobRegistry
from rate_of_closure.application.morris.service import RateMorrisService


class _Socket:
    def __init__(self, *, fail_close: bool = False) -> None:
        self.close_count = 0
        self.fail_close = fail_close

    def setsockopt(self, *_args: object) -> None:
        return None

    def bind(self, _address: object) -> None:
        return None

    def listen(self, _backlog: int) -> None:
        return None

    def getsockname(self) -> tuple[str, int]:
        return ("127.0.0.1", 43210)

    def close(self) -> None:
        self.close_count += 1
        if self.fail_close:
            raise OSError("listener close failed")


class _Registry:
    def __init__(self, _service: object, *, fail_close: bool = False) -> None:
        self.close_count = 0
        self.fail_close = fail_close

    def close(self) -> None:
        self.close_count += 1
        if self.fail_close:
            raise OSError("registry close failed")


class _SocketModule:
    AF_INET = 2
    SOCK_STREAM = 1
    SOL_SOCKET = 0xFFFF
    SO_REUSEADDR = 4
    SOMAXCONN = 128

    def __init__(self, listener: _Socket) -> None:
        self._listener = listener

    def socket(self, *_args: object) -> _Socket:
        return self._listener


def test_child_closes_registry_when_app_setup_fails_before_lifespan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listener = _Socket()
    registries: list[_Registry] = []
    monkeypatch.setenv(child.AUTHORITY_TOKEN_ENV, "private-token")
    monkeypatch.setattr(child, "socket", _SocketModule(listener))

    def registry(service: object) -> _Registry:
        item = _Registry(service)
        registries.append(item)
        return item

    monkeypatch.setattr(child, "MorrisJobRegistry", registry)

    def fail(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("setup failed")

    monkeypatch.setattr(child, "create_morris_authority_app", fail)

    with pytest.raises(RuntimeError, match="setup failed"):
        child.main()
    assert listener.close_count == 1
    assert registries[0].close_count == 1


@pytest.mark.parametrize("registry_close_fails", [False, True])
def test_child_preserves_setup_error_and_attempts_all_cleanup_failures(
    monkeypatch: pytest.MonkeyPatch,
    registry_close_fails: bool,
) -> None:
    listener = _Socket(fail_close=True)
    registry = _Registry(object(), fail_close=registry_close_fails)
    setup_error = RuntimeError("setup failed")
    monkeypatch.setenv(child.AUTHORITY_TOKEN_ENV, "private-token")
    monkeypatch.setattr(child, "socket", _SocketModule(listener))
    monkeypatch.setattr(child, "MorrisJobRegistry", lambda _service: registry)

    def fail(*_args: object, **_kwargs: object) -> None:
        raise setup_error

    monkeypatch.setattr(child, "create_morris_authority_app", fail)

    with pytest.raises(RuntimeError) as caught:
        child.main()
    assert caught.value is setup_error
    assert listener.close_count == 1
    assert registry.close_count == 1


class _CountingRealRegistry(MorrisJobRegistry):
    def __init__(self) -> None:
        self.close_count = 0
        super().__init__(RateMorrisService())

    def close(self) -> None:
        self.close_count += 1
        super().close()


def test_child_transfers_registry_to_normal_asgi_lifespan_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listener = _Socket()
    registry = _CountingRealRegistry()
    monkeypatch.setenv(child.AUTHORITY_TOKEN_ENV, "private-token")
    monkeypatch.setattr(child, "socket", _SocketModule(listener))
    monkeypatch.setattr(child, "MorrisJobRegistry", lambda _service: registry)
    monkeypatch.setattr(child.uvicorn, "Config", lambda app, **_kwargs: app)

    class Server:
        def __init__(self, app: FastAPI) -> None:
            self.app = app
            self.should_exit = False

        def run(self, *, sockets: list[object]) -> None:
            assert sockets == [listener]
            with TestClient(self.app):
                pass

    monkeypatch.setattr(child.uvicorn, "Server", Server)

    assert child.main() == 0
    assert listener.close_count == 1
    assert registry.close_count == 1
