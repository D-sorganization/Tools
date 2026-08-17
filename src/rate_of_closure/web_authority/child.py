"""Environment-only child process with an atomically owned loopback listener."""

from __future__ import annotations

import os
import socket
import sys

import uvicorn

from .runtime import (
    AUTHORITY_APP_FACTORY_ENV,
    AUTHORITY_PORT_ENV,
    LOOPBACK_HOST,
    _app_factory,
)


def _requested_port() -> int:
    source = os.environ.get(AUTHORITY_PORT_ENV, "")
    try:
        port = int(source)
    except ValueError as exc:
        raise ValueError("authority child port must be an integer") from exc
    if str(port) != source or not 0 <= port <= 65_535:
        raise ValueError("authority child port lies outside its bound")
    return port


def main() -> int:
    """Bind before disclosure, report the selected port, and serve one app."""
    factory = _app_factory(os.environ.get(AUTHORITY_APP_FACTORY_ENV, ""))
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        listener.bind((LOOPBACK_HOST, _requested_port()))
        listener.listen(socket.SOMAXCONN)
        port = int(listener.getsockname()[1])
        os.write(sys.stdout.fileno(), f"{port}\n".encode("ascii"))
        config = uvicorn.Config(
            factory,
            factory=True,
            access_log=False,
            log_level="warning",
            server_header=False,
        )
        uvicorn.Server(config).run(sockets=[listener])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
