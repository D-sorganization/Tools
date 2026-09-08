"""Focused tests for the file-based realtime transport.

Pins the D-sorganization/UpstreamDrift#8942 Defect A contract: publish()
must hoist directory setup and handle opening out of the per-message hot
path, and subscribers must resume from a tracked read offset instead of
re-parsing delivered content. Message order, the append-log file layout,
and truncation-bounding behavior are pinned so the module stays drop-in
for the upstream consumer.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from shared.python.realtime import transport_file as transport_file_module
from shared.python.realtime.transport_file import (
    FileTransport,
    default_channel_path,
)


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate the per-test realtime root via the documented env override."""
    override = tmp_path / "realtime-root"
    monkeypatch.setenv("REALTIME_FILE_ROOT", str(override))
    return override


def _make_transport() -> FileTransport:
    return FileTransport(default_channel_path)


def _read_payloads(path: Path) -> list[Any]:
    """Decode the JSON-line append log into its payload sequence."""
    return [
        json.loads(line)["payload"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_publish_appends_one_json_line_per_message_in_order(root: Path) -> None:
    transport = _make_transport()
    channel = "pose/canonical"

    for index in range(5):
        transport.publish(channel, {"seq": index})

    assert _read_payloads(default_channel_path(channel)) == [
        {"seq": i} for i in range(5)
    ]


def test_publish_hoists_mkdir_and_handle_out_of_the_hot_path(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After the first publish, a message must cost zero filesystem setup.

    The upstream implementation ran mkdir + exists + stat + open per
    message; the fixed transport keeps one append handle per channel and
    only re-opens (and re-creates parents) when a truncation rollover
    closes it.
    """
    counts = {"mkdir": 0, "exists": 0, "stat": 0, "open": 0}
    originals = {
        "mkdir": Path.mkdir,
        "exists": Path.exists,
        "stat": Path.stat,
        "open": Path.open,
    }

    def make_wrapper(name: str, original: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(self: Path, *args: Any, **kwargs: Any) -> Any:
            counts[name] += 1
            return original(self, *args, **kwargs)

        return wrapper

    for name, original in originals.items():
        monkeypatch.setattr(Path, name, make_wrapper(name, original))

    transport = _make_transport()
    channel = "pose/canonical"

    transport.publish(channel, {"seq": 0})
    first = dict(counts)
    for index in range(1, 6):
        transport.publish(channel, {"seq": index})

    assert counts == first, (
        f"publish() performed filesystem setup after the first message: {counts}"
    )
    assert _read_payloads(default_channel_path(channel)) == [
        {"seq": i} for i in range(6)
    ]


def test_subscriber_resumes_from_offset_without_reparsing(root: Path) -> None:
    """Callbacks fire in order and delivered lines are never re-parsed."""
    transport = _make_transport()
    channel = "pose/canonical"
    received: list[Any] = []
    transport.subscribe(channel, received.append)

    for index in range(3):
        transport.publish(channel, {"seq": index})
    transport._poll_once()
    assert received == [{"seq": 0}, {"seq": 1}, {"seq": 2}]

    for index in range(3, 6):
        transport.publish(channel, {"seq": index})
    transport._poll_once()
    assert received == [{"seq": i} for i in range(6)]


def test_subscribe_only_receives_messages_published_after_it(root: Path) -> None:
    """A fresh subscription starts at end-of-file, not at history."""
    transport = _make_transport()
    channel = "pose/canonical"
    transport.publish(channel, "before")

    received: list[Any] = []
    transport.subscribe(channel, received.append)
    transport.publish(channel, "after")
    transport._poll_once()

    assert received == ["after"]


def test_truncation_rollover_resets_offset_and_keeps_delivery_flow(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The size bound still holds once the stat check left the hot path."""
    monkeypatch.setattr(transport_file_module, "_MAX_BYTES_PER_CHANNEL", 512)
    transport = _make_transport()
    channel = "pose/canonical"

    received: list[Any] = []
    transport.subscribe(channel, received.append)
    for index in range(12):
        transport.publish(channel, {"seq": index, "pad": "x" * 60})
        transport._poll_once()

    log = default_channel_path(channel)
    assert log.stat().st_size < 512 + 4096  # bounded, not runaway
    # Post-truncation messages keep flowing and the offset stays consistent.
    transport.publish(channel, {"seq": 99, "pad": "y" * 60})
    transport._poll_once()
    assert received[-1] == {"seq": 99, "pad": "y" * 60}


def test_shutdown_closes_open_append_handles(root: Path) -> None:
    """shutdown() must not leak the per-channel append handles."""
    transport = _make_transport()
    transport.publish("pose/canonical", {"seq": 0})

    transport.shutdown()

    assert all(fh.closed for fh in transport._append_handles.values())
