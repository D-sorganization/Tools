"""Child-process entry point owning the authority's ephemeral loopback socket."""

from __future__ import annotations

import logging
import os
import socket
import sys
from pathlib import Path

import platformdirs
import uvicorn

from rate_of_closure.application.durable_ensemble.registry import (
    DurableEnsembleJobRegistry,
)
from rate_of_closure.application.durable_ensemble.service import (
    RateDurableEnsembleService,
)

from .host import create_morris_authority_app
from .router import MorrisJobRegistry
from .service import RateMorrisService

AUTHORITY_TOKEN_ENV = "ROC_MORRIS_AUTHORITY_CHILD_TOKEN"
DURABLE_ARCHIVE_ROOT_ENV = "ROC_DURABLE_ENSEMBLE_ARCHIVE_ROOT"
logger = logging.getLogger(__name__)


def _private_token() -> str:
    token = os.environ.pop(AUTHORITY_TOKEN_ENV, "")
    if len(token) < 8:
        raise RuntimeError("missing Morris authority child token")
    return token


def _cleanup(
    listener: socket.socket,
    registries: tuple[object, ...],
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
    if not lifespan_owns_registry:
        for registry in registries:
            try:
                registry.close()  # type: ignore[attr-defined]
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
                logger.warning("authority registry cleanup failed")
    if cleanup_error is not None and not preserve_primary:
        raise cleanup_error.with_traceback(cleanup_error.__traceback__)


def main() -> int:
    """Bind IPv4 loopback port zero, announce its port, and serve until stopped."""
    token = _private_token()
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    registry: MorrisJobRegistry | None = None
    durable_registry: DurableEnsembleJobRegistry | None = None
    lifespan_owns_registry = False

    def transfer_registry() -> None:
        nonlocal lifespan_owns_registry
        lifespan_owns_registry = True

    try:
        registry = MorrisJobRegistry(RateMorrisService())
        configured_root = os.environ.pop(DURABLE_ARCHIVE_ROOT_ENV, "")
        archive_root = (
            Path(configured_root)
            if configured_root
            else platformdirs.user_state_path("rate-of-closure", appauthor=False)
            / "durable-ensemble-authority-v1"
        )
        durable_registry = DurableEnsembleJobRegistry(
            RateDurableEnsembleService(archive_root)
        )
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
            durable_ensemble_registry=durable_registry,
        )
        config = uvicorn.Config(
            app,
            log_level="warning",
            access_log=False,
            server_header=False,
        )
        server = uvicorn.Server(config)
        holder["server"] = server
        sys.stdout.write(f"{port}\n")
        sys.stdout.flush()
        server.run(sockets=[listener])
        return 0
    finally:
        registries = tuple(
            item for item in (registry, durable_registry) if item is not None
        )
        _cleanup(listener, registries, lifespan_owns_registry)


if __name__ == "__main__":
    raise SystemExit(main())
