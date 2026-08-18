"""Rate web launcher owns authority lifetime around the Vite process."""

from __future__ import annotations

import pytest

from rate_of_closure import launch_web


class _Runtime:
    vite_env = {"ROC_MORRIS_AUTHORITY_URL": "http://127.0.0.1:1"}

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def __enter__(self) -> _Runtime:
        self.events.append("authority-enter")
        return self

    def __exit__(self, *_args: object) -> None:
        self.events.append("authority-exit")


def test_launch_web_owns_authority_around_vite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        launch_web.MorrisAuthorityRuntime,
        "start",
        lambda: _Runtime(events),
    )

    def launch(_info: object, _file: str, *, env_vars: dict[str, str]) -> int:
        events.append("vite")
        assert env_vars == _Runtime.vite_env
        return 23

    monkeypatch.setattr(launch_web, "launch_web_from_gui_info", launch)
    assert launch_web.main() == 23
    assert events == ["authority-enter", "vite", "authority-exit"]


@pytest.mark.parametrize("failure", [OSError("spawn failed"), KeyboardInterrupt()])
def test_launch_web_closes_authority_when_vite_launch_raises(
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        launch_web.MorrisAuthorityRuntime,
        "start",
        lambda: _Runtime(events),
    )

    def launch(_info: object, _file: str, *, env_vars: dict[str, str]) -> int:
        del env_vars
        raise failure

    monkeypatch.setattr(launch_web, "launch_web_from_gui_info", launch)
    with pytest.raises(type(failure)):
        launch_web.main()
    assert events == ["authority-enter", "authority-exit"]
