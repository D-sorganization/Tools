from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from src.shared.python.ai.adapters.base import BaseAgentAdapter, ToolDeclaration
from src.shared.python.ai.types import (
    AgentChunk,
    AgentResponse,
    ConversationContext,
    ProviderCapabilities,
    ProviderCapability,
)
from src.shared.python.contracts import precondition

logger = logging.getLogger(__name__)

_WHEEL_MISSING_HINT = (
    "ai_backend Rust extension is not installed. "
    "Build it from the repo root with: "
    "`cd rust_core/ai_backend && maturin develop --features python`."
)


class RustAgentAdapter(BaseAgentAdapter):
    """Adapter that delegates to the high-performance Rust AI backend.

    Follows the Law of Demeter by encapsulating the ``ai_backend`` extension
    inside standard adapter methods. The wheel is built per-crate via
    ``maturin develop`` from ``rust_core/ai_backend/`` (see the crate's
    ``pyproject.toml``); if the wheel isn't installed in the active
    environment we raise a clear ``ImportError`` rather than failing with a
    bare ``ModuleNotFoundError`` later in the call chain.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        db_path: str = "./memory.db",
        *,
        chat_path: str | None = None,
        embed_path: str | None = None,
        embedding_model: str | None = None,
        use_local_embeddings: bool = False,
    ) -> None:
        """Initialize the Rust Agent Adapter.

        Args:
            api_key: The API key for the LLM.
            base_url: The base URL for the LLM endpoint
                (e.g. ``https://api.openai.com/v1``).
            model: The chat model name.
            db_path: Path to the local vector database.
            chat_path: Path suffix appended to ``base_url`` for chat
                completions. Defaults to ``/chat/completions``.
            embed_path: Path suffix for embeddings. Defaults to
                ``/embeddings``.
            embedding_model: Embedding model name. Defaults to
                ``text-embedding-3-small``.
            use_local_embeddings: When ``True``, embed via a local ONNX
                ``all-MiniLM-L6-v2`` model instead of the HTTP endpoint.
                Requires ``ai_backend`` to be built with
                ``--features python,local-embeddings``. The model is
                cached at ``$UPSTREAM_DRIFT_MODEL_CACHE`` or
                ``~/.cache/upstream-drift/models/`` and downloaded on
                first use.
        """
        try:
            import ai_backend
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise ImportError(_WHEEL_MISSING_HINT) from exc

        self.config = ai_backend.AIConfig(
            api_key,
            base_url,
            model,
            db_path,
            chat_path,
            embed_path,
            embedding_model,
        )
        self.engine = ai_backend.AIEngine(self.config)
        self.memory = ai_backend.MemoryManager(db_path)
        self.memory.initialize()
        # The RagPipeline signature changed when local-embeddings landed
        # (#5227): older wheels won't accept the third positional flag.
        # Fall back to the old call when the new path raises a TypeError so
        # mixed-version environments don't break.
        try:
            self.rag = ai_backend.RagPipeline(
                self.memory, self.config, use_local_embeddings
            )
        except TypeError:
            if use_local_embeddings:
                raise RuntimeError(
                    "Installed ai_backend wheel does not support "
                    "use_local_embeddings; rebuild with "
                    "`maturin develop --features python,local-embeddings`."
                ) from None
            self.rag = ai_backend.RagPipeline(self.memory, self.config)

    def stream_response(
        self,
        prompt: str,
        context: ConversationContext,
        tools: list[Any],
    ) -> Iterator[AgentChunk]:
        """Stream the response using the Rust backend.

        The Rust backend exposes ``stream_response(prompt) -> list[str]`` that
        eagerly drains the SSE stream and returns the ordered delta list.
        We re-emit each delta as an ``AgentChunk`` so callers that already
        iterate this iterator-of-chunks contract keep working. Truly-
        incremental streaming across the PyO3 boundary is a follow-up
        (tracked separately; see issue #2752).

        .. warning::

            **This call is BLOCKING.** The underlying Rust ``AIEngine``
            uses a Tokio ``block_on`` to drain the SSE stream eagerly,
            holding the GIL for the duration of the request. When invoked
            from a Qt UI it MUST be called from a worker thread (e.g.
            ``QThread`` or ``QRunnable``) — otherwise the event loop will
            freeze for the full duration of the LLM response.

            The canonical worker pattern lives in
            ``src/shared/python/ai/gui/assistant_widgets.py`` (see
            ``StreamWorker``); reuse it rather than reimplementing.
        """
        try:
            full_prompt = (
                "\n".join([m.content for m in context.messages]) + f"\n{prompt}"
            )

            # Move the blocking Rust call to a background thread to prevent
            # blocking the Qt event loop if called from the main thread.
            # Truly-incremental streaming across the PyO3 boundary is a follow-up.
            import queue
            import threading
            import time

            try:
                from PyQt6.QtCore import QCoreApplication, QThread

                app = QCoreApplication.instance()
                is_gui_thread = app and (QThread.currentThread() == app.thread())
            except ImportError:
                app = None
                is_gui_thread = False

            result_queue: queue.Queue[Any] = queue.Queue()

            def _fetch_deltas() -> None:
                try:
                    # The Rust backend exposes ``stream_response(prompt) -> list[str]``
                    # that eagerly drains the SSE stream and returns the ordered
                    # delta list.
                    res = self.engine.stream_response(full_prompt)
                    result_queue.put(res)
                except Exception as exc:
                    result_queue.put(exc)

            fetch_thread = threading.Thread(target=_fetch_deltas, daemon=True)
            fetch_thread.start()

            # Wait for the thread to complete while keeping the UI responsive
            while fetch_thread.is_alive() and result_queue.empty():
                if is_gui_thread and app:
                    app.processEvents()
                time.sleep(0.01)

            result = result_queue.get()
            if isinstance(result, Exception):
                # Fall back to the blocking single-shot path if streaming
                # fails (e.g. provider does not support stream=true).
                response = self.engine.generate_response(full_prompt)
                yield AgentChunk(content=response, is_final=True)
                return

            deltas = result
            if not deltas:
                yield AgentChunk(content="", is_final=True)
                return

            last_idx = len(deltas) - 1
            for idx, delta in enumerate(deltas):
                yield AgentChunk(content=delta, is_final=(idx == last_idx))
                # Yield control back to the event loop between chunks if we
                # are on the GUI thread.
                if is_gui_thread and app:
                    app.processEvents()

        except Exception as e:
            logger.exception("Rust backend error")
            yield AgentChunk(content=f"Error: {e}", is_final=True)

    def index_codebase(self, root_path: str) -> int:
        """Trigger the Rust-based RAG pipeline indexer."""
        return int(self.rag.index_codebase(root_path))

    def retrieve_context(self, prompt: str, top_k: int = 5) -> list[str]:
        """Retrieve semantic context using the Rust vector memory."""
        return [str(item) for item in self.rag.retrieve_context(prompt, top_k)]

    @precondition(
        lambda prompt: bool(prompt.strip()), "message must not be empty or blank"
    )
    def send_message(
        self, prompt: str, context: ConversationContext, tools: list[ToolDeclaration]
    ) -> AgentResponse:
        """Send a message synchronously."""
        del tools
        # Canonical zero-usage so callers always see the same keys (issue #2763).
        canonical_usage = self._normalize_token_counts({})
        try:
            full_prompt = (
                "\n".join([m.content for m in context.messages]) + f"\n{prompt}"
            )
            response = self.engine.generate_response(full_prompt)
            return AgentResponse(content=str(response), usage=canonical_usage)
        except Exception as e:
            logger.exception("Rust backend error")
            return AgentResponse(
                content=f"Error: {e}",
                finish_reason="error",
                usage=canonical_usage,
            )

    def validate_connection(self) -> tuple[bool, str]:
        """Validate connection to the backend."""
        return True, "Rust backend initialized successfully."

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Return adapter capabilities."""
        return ProviderCapabilities(
            supported=frozenset(
                {
                    ProviderCapability.STREAMING,
                    ProviderCapability.SYSTEM_MESSAGE,
                }
            ),
            max_tokens=8192,
            model_name=str(self.config.model),
            provider_name="rust",
        )
