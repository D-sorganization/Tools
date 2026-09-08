"""Realtime IPC transports for cross-tool live data streaming.

Canonical home of the shared file transport in the fleet's provider tree
(``src/shared/python/``); UpstreamDrift consumes this module through its
``vendor/ud-tools`` pin. The file transport is the default hint-layer
transport: publishers append JSON lines to a per-channel log and
subscribers tail it. See :mod:`transport_file` for the semantics and the
performance contract.
"""

from __future__ import annotations

from .transport_file import FileTransport, default_channel_path

__all__ = ["FileTransport", "default_channel_path"]
