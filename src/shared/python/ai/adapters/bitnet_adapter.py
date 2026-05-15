"""BitNet local adapter using direct subprocess execution.

This adapter allows the shared Tools AI chat interface to run 1.58b models locally
without requiring an external FastAPI server. It manages a llama-cli subprocess
directly.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Iterator

from src.shared.python.ai.adapters.base import (
    BaseAgentAdapter,
    ToolDeclaration,
    _ensure_final_chunk,
)
from src.shared.python.ai.exceptions import AIConnectionError, AIProviderError
from src.shared.python.ai.types import (
    AgentChunk,
    AgentResponse,
    ConversationContext,
    ProviderCapabilities,
    ProviderCapability,
)
from src.shared.python.contracts import precondition
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


class BitnetAdapter(BaseAgentAdapter):
    """Adapter for running local BitNet models via direct subprocess.

    This adapter launches and manages a llama.cpp / llama-cli process directly,
    providing a seamless local LLM experience within the shared chat interface.

    Token usage:
        BitNet's llama-cli does not surface token counts in a parseable
        form, so :pyattr:`AgentResponse.usage` is always a zero-valued
        :class:`TokenUsage`. This satisfies the cross-adapter contract
        (issue #2763) at the cost of zero observability into actual
        token consumption.
    """

    def __init__(
        self, model: str | None = None, bitnet_root: str | None = None
    ) -> None:
        """Initialize the BitNet adapter.

        Args:
            model: Name or path to the model file to run.
            bitnet_root: Path to the root of the bitnet installation.
        """
        self.model = model or "bitnet-1.58b-q4_0.gguf"
        self.bitnet_root = bitnet_root or os.environ.get("BITNET_ROOT", "")
        self.llama_cli = (
            os.path.join(self.bitnet_root, "llama-cli")
            if self.bitnet_root
            else "llama-cli"
        )
        self._process: subprocess.Popen | None = None
        self._capabilities = ProviderCapabilities(
            supported=frozenset(
                {
                    ProviderCapability.STREAMING,
                    ProviderCapability.SYSTEM_MESSAGE,
                }
            ),
            max_tokens=2048,
            model_name=self.model,
            provider_name="bitnet",
        )

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._capabilities

    def _handle_error(self, error: Exception) -> AgentResponse:
        """Classify BitNet subprocess errors into the AIProviderError hierarchy.

        Overrides the base implementation to map subprocess-specific failures:
        - FileNotFoundError (binary missing) → AIConnectionError
        - CalledProcessError (non-zero exit) → AIProviderError
        - All others → delegated to parent classify_ai_error

        Args:
            error: The raw exception from subprocess.

        Raises:
            AIConnectionError: When the llama-cli binary cannot be found.
            AIProviderError: For all other BitNet process failures.
        """
        if isinstance(error, FileNotFoundError):
            raise AIConnectionError(
                f"Failed to run BitNet: {error}", provider="bitnet"
            ) from error
        if isinstance(error, subprocess.CalledProcessError):
            raise AIProviderError(
                f"BitNet process failed: {error.stderr}", provider="bitnet"
            ) from error
        return super()._handle_error(error)

    def validate_connection(self) -> tuple[bool, str]:
        """Validate that the llama-cli executable is available."""
        try:
            # Just test if the executable exists and can be run
            result = subprocess.run(
                [self.llama_cli, "--help"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5.0,
            )
            if result.returncode == 0 or "usage" in result.stdout.lower():
                return True, f"Found {self.llama_cli}"
            return False, f"Executable failed: {result.stderr}"
        except Exception as e:
            return False, f"Failed to execute {self.llama_cli}: {e}"

    def _format_prompt(self, context: ConversationContext, message: str) -> str:
        """Format the conversation into a plain text prompt."""
        prompt = ""
        if context and context.messages:
            for msg in context.messages:
                if msg.role == "system":
                    prompt += f"System: {msg.content}\n"
                elif msg.role == "user":
                    prompt += f"User: {msg.content}\n"
                elif msg.role == "assistant":
                    prompt += f"Assistant: {msg.content}\n"
        prompt += f"User: {message}\nAssistant:"
        return prompt

    @precondition(
        lambda message: bool(message.strip()), "message must not be empty or blank"
    )
    def send_message(
        self,
        message: str,
        context: ConversationContext,
        tools: list[ToolDeclaration],
    ) -> AgentResponse:
        """Send a message synchronously."""
        prompt = self._format_prompt(context, message)

        try:
            cmd = [
                self.llama_cli,
                "-m",
                self.model,
                "-p",
                prompt,
                "-n",
                "512",
                "--log-disable",
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )

            output = result.stdout.replace(prompt, "").strip()

            return AgentResponse(
                content=output,
                metadata={"stdout": result.stdout},
            )
        except Exception as e:
            logger.error("Failed to run BitNet: %s", e)
            return self._handle_error(e)

    def stream_response(
        self,
        message: str,
        context: ConversationContext,
        tools: list[ToolDeclaration],
    ) -> Iterator[AgentChunk]:
        """Stream the response using subprocess (with final-chunk invariant, #2763)."""
        return _ensure_final_chunk(self._stream_response_raw(message, context, tools))

    def _stream_response_raw(
        self,
        message: str,
        context: ConversationContext,
        tools: list[ToolDeclaration],
    ) -> Iterator[AgentChunk]:
        """Raw BitNet streaming generator (pre-finality-wrapper)."""
        del tools
        prompt = self._format_prompt(context, message)

        cmd = [
            self.llama_cli,
            "-m",
            self.model,
            "-p",
            prompt,
            "-n",
            "512",
            "--log-disable",
        ]

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
            )

            if not process.stdout:
                raise AIProviderError(
                    "BitNet process stdout is unavailable", provider="bitnet"
                )

            for index, line in enumerate(process.stdout):
                yield AgentChunk(
                    content=line,
                    is_final=False,
                    index=index,
                )

            process.wait()
            yield AgentChunk(content="", is_final=True)

        except Exception as e:
            logger.error("Failed to stream BitNet: %s", e)
            yield AgentChunk(content=f"\n[Error: {e}]", is_final=True)
