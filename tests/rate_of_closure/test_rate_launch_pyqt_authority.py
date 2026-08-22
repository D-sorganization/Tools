"""Standalone PyQt launcher ownership for the private Morris authority."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rate_of_closure.ui.pyqt6 import launcher as launcher_module

pytestmark = pytest.mark.unit


class _Runtime:
    base_url = "http://127.0.0.1:54321"
    authorization_headers = {"Authorization": "Bearer private-test-token"}

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_launcher_injects_client_and_closes_runtime_after_event_loop(
    monkeypatch,
) -> None:
    runtime = _Runtime()
    captured = []
    monkeypatch.setattr(
        launcher_module.MorrisAuthorityRuntime,
        "start",
        classmethod(lambda cls: runtime),
    )
    monkeypatch.setattr(
        launcher_module,
        "launch_pyqt6_app",
        lambda config: captured.append(config) or 23,
    )

    assert launcher_module.launch_rate_pyqt6() == 23
    assert runtime.closed
    client = captured[0].window_kwargs["morris_client"]
    durable = captured[0].window_kwargs["durable_ensemble_client"]
    assert client.base_url == runtime.base_url
    assert durable.base_url == runtime.base_url
    assert "private-test-token" not in repr(captured[0])


def test_launcher_degrades_honestly_when_optional_authority_is_unavailable(
    monkeypatch, caplog: pytest.LogCaptureFixture
) -> None:
    def unavailable():
        raise RuntimeError("child could not import optional dependency")

    captured = []
    monkeypatch.setattr(
        launcher_module.MorrisAuthorityRuntime,
        "start",
        classmethod(lambda cls: unavailable()),
    )
    monkeypatch.setattr(
        launcher_module,
        "launch_pyqt6_app",
        lambda config: captured.append(config) or 0,
    )

    assert launcher_module.launch_rate_pyqt6() == 0
    assert captured[0].window_kwargs == {
        "morris_client": None,
        "durable_ensemble_client": None,
    }
    assert "Morris Screening unavailable" in caplog.text


def test_launcher_closes_runtime_when_qt_launch_raises(monkeypatch) -> None:
    runtime = _Runtime()
    monkeypatch.setattr(
        launcher_module.MorrisAuthorityRuntime,
        "start",
        classmethod(lambda cls: runtime),
    )

    def fail(_config):  # type: ignore[no-untyped-def]
        raise RuntimeError("Qt failed")

    monkeypatch.setattr(launcher_module, "launch_pyqt6_app", fail)

    with pytest.raises(RuntimeError, match="Qt failed"):
        launcher_module.launch_rate_pyqt6()
    assert runtime.closed


def test_launch_config_hides_injected_values_from_repr() -> None:
    config = launcher_module._launch_config(SimpleNamespace(secret="not-public"))

    assert "not-public" not in repr(config)
