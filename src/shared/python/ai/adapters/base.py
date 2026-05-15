"""Base adapter protocol for AI providers.

This module defines the abstract interface that all AI provider adapters
must implement, ensuring consistent behavior across providers.

The protocol pattern allows for easy addition of new providers without
modifying existing code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from src.shared.python.logging_pkg.logging_config import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator

from src.shared.python.ai.memory_manager import (
    build_memory_prompt_section,
    load_agents_md,
)
from src.shared.python.ai.system_prompts import get_prompt
from src.shared.python.ai.types import (
    AgentChunk,
    AgentResponse,
    ConversationContext,
    ProviderCapabilities,
)

logger = get_logger(__name__)


@dataclass
class ToolDeclaration:
    """Declaration of a tool available to the AI.

    This is a simplified version for adapter communication.
    The full ToolDeclaration with validation is in tool_registry.py.

    Attributes:
        name: Unique tool identifier.
        description: What the tool does (AI-consumable).
        parameters: JSON Schema for tool parameters.
        required: List of required parameter names.
    """

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    required: list[str] = field(default_factory=list)

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI function calling format.

        Returns:
            OpenAI-compatible function definition.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required,
                },
            },
        }

    def to_anthropic_format(self) -> dict[str, Any]:
        """Convert to Anthropic tool format.

        Returns:
            Anthropic-compatible tool definition.
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": self.parameters,
                "required": self.required,
            },
        }


class BaseAgentAdapter(ABC):
    """Abstract base class for AI provider adapters.

    All provider-specific adapters inherit from this class and implement
    the required methods for communication with their respective APIs.

    The adapter is responsible for:
    1. Translating AIP messages to provider format
    2. Managing authentication and connections
    3. Handling provider-specific errors
    4. Implementing streaming where supported

    Example:
        >>> class MyAdapter(BaseAgentAdapter):
        ...     def send_message(self, message, context, tools):
        ...         # Translate and send to provider
        ...         ...
    """

    @abstractmethod
    def send_message(
        self,
        message: str,
        context: ConversationContext,
        tools: list[ToolDeclaration],
    ) -> AgentResponse:
        """Send a message to the AI provider.

        This is the primary method for synchronous communication with
        the AI provider. It handles the full request/response cycle.

        Args:
            message: User message to process.
            context: Current conversation context with history.
            tools: List of tools available for this request.

        Returns:
            Provider response translated to standard AgentResponse format.

        Raises:
            AIProviderError: If provider communication fails.
            AIConnectionError: If network connection fails.
            AIRateLimitError: If rate limit is exceeded.
            AITimeoutError: If request times out.
        """
        ...

    @abstractmethod
    def stream_response(
        self,
        message: str,
        context: ConversationContext,
        tools: list[ToolDeclaration],
    ) -> Iterator[AgentChunk]:
        """Stream response chunks from the AI provider.

        For providers that support streaming, this method yields
        response chunks as they arrive, enabling real-time UI updates.

        Args:
            message: User message to process.
            context: Current conversation context with history.
            tools: List of tools available for this request.

        Yields:
            Response chunks as they arrive from the provider.

        Raises:
            AIProviderError: If provider communication fails.
        """
        ...

    @property
    @abstractmethod
    def capabilities(self) -> ProviderCapabilities:
        """Return the provider's capabilities.

        This allows the AIP to adjust behavior based on what features
        the current provider supports.

        Returns:
            ProviderCapabilities describing supported features.
        """
        ...

    @abstractmethod
    def validate_connection(self) -> tuple[bool, str]:
        """Test connection to the AI provider.

        This method should perform a lightweight check to verify:
        1. Network connectivity
        2. Authentication validity
        3. Model availability

        Returns:
            Tuple of (success: bool, diagnostic_message: str).
        """
        ...

    def format_messages_for_provider(
        self,
        context: ConversationContext,
        current_message: str,
    ) -> list[dict[str, Any]]:
        """Format conversation history for the provider.

        This default implementation provides a basic format that works
        for most providers. Override for provider-specific formatting.

        Args:
            context: Conversation context with history.
            current_message: The current user message.

        Returns:
            List of message dictionaries for the provider.
        """
        if not (context is not None):
            raise ValueError("context must be provided")
        if not (context is not None):
            raise ValueError("context must be provided")
        messages: list[dict[str, Any]] = []

        # Add conversation history
        for msg in context.messages:
            formatted: dict[str, Any] = {
                "role": msg.role,
                "content": msg.content,
            }
            if msg.tool_call_id:
                formatted["tool_call_id"] = msg.tool_call_id
            messages.append(formatted)

        # Add current message
        messages.append(
            {
                "role": "user",
                "content": current_message,
            }
        )

        return messages

    def build_system_prompt(
        self,
        tools: list[ToolDeclaration],
        expertise_level: str = "beginner",
        context: ConversationContext | None = None,
        app_context: str | None = None,
    ) -> str:
        """Build a system prompt including tool context.

        This default implementation provides a basic system prompt.
        Override for provider-specific or use-case-specific prompts.

        Args:
            tools: Available tools to describe.
            expertise_level: User's expertise level.
            context: Optional conversation context containing project and
                prompt-memory metadata.

        Returns:
            System prompt string.
        """
        if not (tools is not None):
            raise ValueError("tools must be provided")
        if not (tools is not None):
            raise ValueError("tools must be provided")
        tool_descriptions = "\n".join(
            f"- {tool.name}: {tool.description}" for tool in tools
        )
        memory_section = self.build_context_instruction_section(context)
        base_prompt = get_prompt(app_context)

        # Response style instructions (Tools #2750)
        style = (context.response_style if context else "standard").lower()
        style_instructions = ""
        if style == "concise":
            style_instructions = (
                "Reply concisely. Prefer code, tables, and short bullet lists over "
                "prose. Skip preamble and recap."
            )
        elif style == "detailed":
            style_instructions = (
                "Reply in detail. Walk through reasoning, name relevant trade-offs, "
                "and include worked examples when they clarify the answer."
            )
        else:  # standard
            style_instructions = (
                "Reply at a standard level of detail. Briefly explain reasoning "
                "where it helps the user act on the answer."
            )

        return (
            f"{base_prompt}\n\n"
            f"Response style: {style_instructions}\n\n"
            f"{memory_section}\n\n"
            f"Available tools:\n{tool_descriptions}"
        )

    def build_context_instruction_section(
        self,
        context: ConversationContext | None,
    ) -> str:
        """Build repository and persisted-memory prompt context."""
        if context is None:
            return ""

        project_root_value = context.metadata.get("project_root")
        project_root = None
        if isinstance(project_root_value, str) and project_root_value:
            from pathlib import Path

            project_root = Path(project_root_value)

        prompt_memory = context.metadata.get("prompt_memory")
        if not isinstance(prompt_memory, dict):
            prompt_memory = None

        return cast(
            str,
            build_memory_prompt_section(
                prompt_memory=prompt_memory,
                agents_md=load_agents_md(project_root),
            ),
        )
