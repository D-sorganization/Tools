"""WebSocket fan-out for the live telemetry stream.

Split out of ``main`` so the broadcast contract is a small, independently
testable unit: one slow HMI client must never be able to stall the PLC control
loop (issue #4024). The manager serialises a frame ONCE, hands it to a bounded
per-client queue and returns without awaiting anything socket-related; each
client is drained by its own task and can only ever starve itself.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger("dcs_backend.ws_broadcast")

__all__ = ["ConnectionManager", "WS_CLIENT_QUEUE_SIZE", "WS_SEND_TIMEOUT_S"]


#: Frames buffered per WebSocket client before the OLDEST is dropped. Live
#: telemetry is only useful fresh, so a stalled client loses history, not the
#: newest value — and never stalls the control loop (issue #4024).
WS_CLIENT_QUEUE_SIZE = 8
#: Hard bound on a single client send, so a half-open TCP socket cannot pin a
#: pump task forever.
WS_SEND_TIMEOUT_S = 5.0


class _ClientChannel:
    """A bounded outbox for one WebSocket client, drained by its own task."""

    def __init__(self, websocket: Any, *, queue_size: int) -> None:
        self.websocket = websocket
        self.queue: asyncio.Queue[str] = asyncio.Queue(maxsize=queue_size)
        self.dropped = 0
        self.task: asyncio.Task[None] | None = None

    def offer(self, text: str) -> None:
        """Enqueue a frame, evicting the oldest when the client is behind."""
        try:
            self.queue.put_nowait(text)
            return
        except asyncio.QueueFull:
            pass
        try:
            self.queue.get_nowait()
            self.dropped += 1
        except asyncio.QueueEmpty:  # pragma: no cover - racy drain
            pass
        try:
            self.queue.put_nowait(text)
        except asyncio.QueueFull:  # pragma: no cover - racy producer
            self.dropped += 1


class ConnectionManager:
    """Fans a live frame out to WebSocket clients without ever blocking.

    ``broadcast`` serialises the frame ONCE, drops it into each client's bounded
    queue and returns; it awaits nothing socket-related. Each client is drained
    by its own task, so a slow or half-open HMI can only starve itself (issue
    #4024).
    """

    def __init__(
        self,
        *,
        queue_size: int = WS_CLIENT_QUEUE_SIZE,
        send_timeout_s: float = WS_SEND_TIMEOUT_S,
    ) -> None:
        if not isinstance(queue_size, int) or isinstance(queue_size, bool):
            raise TypeError(
                f"queue_size must be an int, got {type(queue_size).__name__}"
            )
        if queue_size < 1:
            raise ValueError(f"queue_size must be >= 1, got {queue_size}")
        self._channels: dict[Any, _ClientChannel] = {}
        self._queue_size = queue_size
        self._send_timeout_s = float(send_timeout_s)
        self._frames_dropped = 0

    @property
    def active_connections(self) -> list[Any]:
        """The currently registered client sockets (snapshot)."""
        return list(self._channels)

    @property
    def frames_dropped(self) -> int:
        """Frames discarded across all clients because their outbox was full."""
        return self._frames_dropped + sum(c.dropped for c in self._channels.values())

    def pending_frames(self, websocket: Any) -> list[str]:
        """Serialised frames still queued for ``websocket`` (diagnostics/tests)."""
        channel = self._channels.get(websocket)
        if channel is None:
            return []
        # Read-only peek at the queue's buffer; asyncio.Queue exposes no public
        # non-destructive iteration and draining it here would lose frames.
        buffered = getattr(channel.queue, "_queue", ())
        return [str(item) for item in buffered]

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.register_accepted(websocket)
        logger.info("New WebSocket client connected.")

    def register_accepted(self, websocket: Any) -> None:
        """Register an already-accepted socket without re-accepting it.

        Used by the frame-authenticated path, which must ``accept()`` before it
        can read the credential frame and therefore cannot call ``connect()``.
        Routing the registration through the manager keeps connection
        bookkeeping — and now the per-client outbox and its pump task — in one
        place instead of reaching into the connection list directly.
        """
        if websocket in self._channels:
            return
        channel = _ClientChannel(websocket, queue_size=self._queue_size)
        self._channels[websocket] = channel
        try:
            channel.task = asyncio.get_running_loop().create_task(self._pump(channel))
        except RuntimeError:
            # No running loop (sync unit test / import-time registration): the
            # outbox still buffers, it is simply not being drained yet.
            channel.task = None

    def disconnect(self, websocket: Any) -> None:
        channel = self._channels.pop(websocket, None)
        if channel is None:
            return
        self._frames_dropped += channel.dropped
        task = channel.task
        if task is not None and task is not asyncio.current_task():
            task.cancel()
        logger.info("WebSocket client disconnected.")

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Publish one frame to every client. Awaits no socket, ever."""
        if not self._channels:
            return
        text = json.dumps(message, default=str)
        for channel in list(self._channels.values()):
            channel.offer(text)

    async def _pump(self, channel: _ClientChannel) -> None:
        """Drain one client's outbox until it disconnects or misbehaves."""
        while True:
            text = await channel.queue.get()
            try:
                await asyncio.wait_for(
                    channel.websocket.send_text(text), self._send_timeout_s
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - any send failure is fatal
                logger.warning("Dropping unreachable WebSocket client: %s", exc)
                self.disconnect(channel.websocket)
                return
