"""Chat API contract models.

Generic Pydantic models for the WebSocket chat protocol used by
ChatDockWidget. These are reusable across any application that
integrates the shared chat system.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatMessageRequest(BaseModel):
    """Request to send a chat message."""

    message: str = Field(..., min_length=1, max_length=10000)
    app_context: str | None = Field(
        None, description="Active application context (e.g. 'mujoco', 'gasification')"
    )
    expertise_level: str = Field("beginner")


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


class ChatIndexStatusResponse(BaseModel):
    """Codebase indexing progress / completion status.

    Sent by the server in response to an ``index_codebase`` action (or
    pushed unsolicited while indexing runs in the background) so the chat
    UI can show progress and indicate when full-codebase context becomes
    available (#2549).
    """

    state: str = Field(
        ...,
        description="One of: 'idle', 'running', 'complete', 'error'",
    )
    files_parsed: int = Field(0, ge=0)
    symbols_inserted: int = Field(0, ge=0)
    duration_seconds: float | None = Field(None, ge=0.0)
    error: str | None = Field(None, description="Populated when state == 'error'")
