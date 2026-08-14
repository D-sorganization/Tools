"""Child-process entry point owning the authority's ephemeral loopback socket."""

from __future__ import annotations

import logging
import os
import socket
import sys

import uvicorn

from .host import create_morris_authority_app
from .router import MorrisJobRegistry
from .service import RateMorrisService

AUTHORITY_TOKEN_ENV = "ROC_MORRIS_AUTHORITY_CHILD_TOKEN"
logger = logging.getLogger(__name__)


def _private_token() -> str:
    token = os.environ.pop(AUTHORITY_TOKEN_ENV, "")
    if len(token) < 8:
        raise RuntimeError("missing Morris authority child token")
    return token


def _cleanup(
    listener: socket.socket,
    registry: MorrisJobRegistry | None,
    lifespan_owns_registry: bool,
) -> None:
    """Attempt every child cleanup without replacing an active primary error."""
    preserve_primary = sys.exception() is not None
    cleanup_error: BaseException | None = None
    try:
        listener.close()
    except BaseException as error:
        cleanup_error = error
        logger.warning("Morris authority listener cleanup failed")
    if registry is not None and not lifespan_owns_registry:
        try:
            registry.close()
        except BaseException as error:
            if cleanup_error is None:
                cleanup_error = error
            logger.warning("Morris authority registry cleanup failed")
    if cleanup_error is not None and not preserve_primary:
        raise cleanup_error.with_traceback(cleanup_error.__traceback__)


def main() -> int:
    """Bind IPv4 loopback port zero, announce its port, and serve until stopped."""
    token = _private_token()
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    registry: MorrisJobRegistry | None = None
    lifespan_owns_registry = False

    def transfer_registry() -> None:
        nonlocal lifespan_owns_registry
        lifespan_owns_registry = True

    try:
        registry = MorrisJobRegistry(RateMorrisService())
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        exclusive = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
        if exclusive is not None:
            listener.setsockopt(socket.SOL_SOCKET, exclusive, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(socket.SOMAXCONN)
        port = int(listener.getsockname()[1])
        holder: dict[str, uvicorn.Server] = {}
        app = create_morris_authority_app(
            token,
            registry,
            lambda: setattr(holder["server"], "should_exit", True),
            lifespan_started=transfer_registry,
        )
        config = uvicorn.Config(
            app,
            log_level="warning",
            access_log=False,
            server_header=False,
        )
        server = uvicorn.Server(config)
        holder["server"] = server
        print(  # noqa: T201 - private parent/child readiness protocol
            port, flush=True
        )
        server.run(sockets=[listener])
        return 0
    finally:
        _cleanup(listener, registry, lifespan_owns_registry)


if __name__ == "__main__":
    raise SystemExit(main())
