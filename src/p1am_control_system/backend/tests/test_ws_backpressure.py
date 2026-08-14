"""WebSocket broadcast backpressure (issue #4024).

One slow HMI client must never stall the control loop. ``broadcast`` serialises
the frame once, hands it to a bounded per-client queue and returns; a dedicated
task per client does the awaiting. When a client's queue is full the OLDEST
frame is dropped — for live telemetry only the newest value matters.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

sys.path.insert(0, str(Path(__file__).parent.parent))

from main import ConnectionManager  # noqa: E402


class _Socket:
    """A WebSocket double whose send can be made arbitrarily slow."""

    def __init__(self, *, delay_s: float = 0.0, fail: bool = False) -> None:
        self.sent: list[str] = []
        self.delay_s = delay_s
        self.fail = fail
        self.release = asyncio.Event()

    async def send_text(self, text: str) -> None:
        if self.fail:
            raise RuntimeError("socket is gone")
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        self.sent.append(text)

    async def send_json(self, message: object) -> None:  # pragma: no cover
        raise AssertionError("broadcast must serialise once via send_text")


@pytest.mark.asyncio
async def test_broadcast_never_awaits_a_slow_client() -> None:
    manager = ConnectionManager()
    slow = _Socket(delay_s=5.0)
    manager.register_accepted(slow)

    loop = asyncio.get_running_loop()
    start = loop.time()
    for i in range(5):
        await manager.broadcast({"seq": i})
    elapsed = loop.time() - start

    assert elapsed < 0.5, "the control loop was blocked by a slow socket"
    manager.disconnect(slow)


@pytest.mark.asyncio
async def test_full_client_queue_drops_the_oldest_frame() -> None:
    manager = ConnectionManager(queue_size=2)
    slow = _Socket(delay_s=5.0)
    manager.register_accepted(slow)

    for i in range(6):
        await manager.broadcast({"seq": i})

    assert manager.frames_dropped > 0
    # Whatever survives is the freshest telemetry, never a stale head.
    queued = [json.loads(t)["seq"] for t in manager.pending_frames(slow)]
    assert queued == sorted(queued)
    assert max(queued) == 5
    manager.disconnect(slow)


@pytest.mark.asyncio
async def test_frame_is_serialised_once_for_all_clients() -> None:
    manager = ConnectionManager()
    a, b = _Socket(), _Socket()
    manager.register_accepted(a)
    manager.register_accepted(b)

    await manager.broadcast({"tags": [1.0, 2.0], "plc_connected": True})
    await asyncio.sleep(0.05)

    assert a.sent == b.sent
    assert json.loads(a.sent[0])["plc_connected"] is True
    manager.disconnect(a)
    manager.disconnect(b)


@pytest.mark.asyncio
async def test_dead_client_is_pruned_without_touching_the_loop() -> None:
    manager = ConnectionManager()
    dead = _Socket(fail=True)
    manager.register_accepted(dead)

    await manager.broadcast({"seq": 0})
    await asyncio.sleep(0.05)

    assert dead not in manager.active_connections


@pytest.mark.asyncio
async def test_disconnect_cancels_the_client_pump() -> None:
    manager = ConnectionManager()
    socket = _Socket(delay_s=5.0)
    manager.register_accepted(socket)
    await manager.broadcast({"seq": 0})

    manager.disconnect(socket)
    await asyncio.sleep(0)

    assert socket not in manager.active_connections
    assert manager.pending_frames(socket) == []


@pytest.mark.asyncio
async def test_broadcast_with_no_clients_is_a_no_op() -> None:
    manager = ConnectionManager()
    await manager.broadcast({"seq": 0})
    assert manager.frames_dropped == 0


def test_register_accepted_outside_a_loop_still_tracks_the_socket() -> None:
    """The sync guard test in test_validation_guards_3745 relies on this."""
    manager = ConnectionManager()

    class _Plain:
        def accept(self) -> None:  # pragma: no cover - must not be called
            raise AssertionError("register_accepted must not accept()")

    socket = _Plain()
    manager.register_accepted(socket)
    assert socket in manager.active_connections
