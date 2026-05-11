"""Chat API contract models.

Generic Pydantic models for the WebSocket chat protocol used by
ChatDockWidget. These are reusable across any application that
integrates the shared chat system.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Response style (#2552).
#
# The chat used to expose a "skill level" selector (beginner / intermediate /
# expert) which conflated the user's expertise with how verbose they want the
# AI to be. We now describe the *response*, not the user — concise, standard,
# or detailed. The literal values are stable and form part of the wire
# protocol.
# ---------------------------------------------------------------------------

ResponseStyle = Literal["concise", "standard", "detailed"]

DEFAULT_RESPONSE_STYLE: ResponseStyle = "standard"

#: System-prompt fragment for each style. The chat backend appends the
#: appropriate string to its base system prompt so the model adapts its
#: verbosity to the user's choice.
RESPONSE_STYLE_PROMPTS: dict[ResponseStyle, str] = {
    "concise": (
        "Respond as concisely as possible. Prefer short sentences, bullet "
        "points, and minimal preamble. Skip examples unless explicitly asked."
    ),
    "standard": (
        "Respond at a balanced length: enough detail to be useful, but no "
        "filler. Include a short example when it materially clarifies the "
        "answer."
    ),
    "detailed": (
        "Respond in depth. Walk through the reasoning, cover edge cases and "
        "trade-offs, and include illustrative examples or code where they "
        "help. Err on the side of more context rather than less."
    ),
}

# Legacy skill-level labels mapped to the new response-style values so
# existing clients continue to work without coordination. New code should
# pass ``response_style`` directly.
_LEGACY_EXPERTISE_TO_STYLE: dict[str, ResponseStyle] = {
    "beginner": "detailed",
    "intermediate": "standard",
    "advanced": "concise",
    "expert": "concise",
}


def style_prompt(style: str | None) -> str:
    """Return the system-prompt fragment for ``style``.

    Falls back to the default style for unknown / ``None`` inputs so callers
    can pass user-supplied values without pre-validation.
    """
    if style in RESPONSE_STYLE_PROMPTS:
        return RESPONSE_STYLE_PROMPTS[style]  # type: ignore[index]
    return RESPONSE_STYLE_PROMPTS[DEFAULT_RESPONSE_STYLE]


class ChatMessageRequest(BaseModel):
    """Request to send a chat message.

    The ``response_style`` field (#2552) describes how verbose the AI's
    response should be. The legacy ``expertise_level`` field is kept for
    backward compatibility and is auto-translated to ``response_style``
    when the latter is not supplied.
    """

    message: str = Field(..., min_length=1, max_length=10000)
    app_context: str | None = Field(
        None, description="Active application context (e.g. 'mujoco', 'gasification')"
    )
    response_style: ResponseStyle = Field(
        DEFAULT_RESPONSE_STYLE,
        description=(
            "How verbose the AI's response should be: 'concise', 'standard', "
            "or 'detailed'. Describes the response, not the user's skill."
        ),
    )
    # Deprecated. Retained so existing UpstreamDrift / Gasification_Model
    # builds keep working. New code should use ``response_style``.
    expertise_level: str = Field(
        "beginner",
        description="DEPRECATED: use ``response_style`` instead (#2552).",
    )

    @model_validator(mode="after")
    def _coerce_legacy_expertise_level(self) -> ChatMessageRequest:
        """Map legacy ``expertise_level`` onto ``response_style``.

        Only fires when the caller explicitly passed ``expertise_level`` and
        did *not* pass ``response_style``, so new clients that set
        ``response_style`` always win and pure defaults always resolve to
        ``DEFAULT_RESPONSE_STYLE``.
        """
        fields_set = self.model_fields_set
        if (
            "expertise_level" in fields_set
            and "response_style" not in fields_set
            and self.expertise_level in _LEGACY_EXPERTISE_TO_STYLE
        ):
            object.__setattr__(
                self,
                "response_style",
                _LEGACY_EXPERTISE_TO_STYLE[self.expertise_level],
            )
        return self


class ChatChunkResponse(BaseModel):
    """A single streaming chunk from the AI."""

    content: str
    is_final: bool = False
    index: int = 0


class ChatSessionInfo(BaseModel):
    """Summary info for a chat session."""

    session_id: str
    message_count: int
    created_at: str
    last_active: str
    app_contexts: list[str] = Field(default_factory=list)


class ChatHistoryResponse(BaseModel):
    """Full message history for a session."""

    session_id: str
    messages: list[dict]
