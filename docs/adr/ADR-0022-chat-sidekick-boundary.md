# ADR-0022: Chat Sidekick Boundary

> **Mirrored ADR (fleet ADR home: ADR-0049).**
> Source: UpstreamDrift `docs/adr/0022-chat-sidekick-boundary.md` @ `27b6eeadbbd9` (blob `573794a8b248`); mirrored 2026-09-03; canonical home: Tools (ADR-0049).
> This copy is byte-for-byte the UpstreamDrift text below this notice. Amend it here
> first and carry the change to UpstreamDrift in a paired PR; `scripts/check_adr_references.py`
> keeps every `ADR-NNNN` cited from `src/` resolvable to a file in this directory.

Status: Accepted

## Context

Issue #6098 requires consolidating chat boundaries. See also #5922, #5967, and #5969.

## Decision

Sidekick is the canonical chat surface (`ChatPanel`, `UnifiedToolsSidebar`); the legacy chat dock (`_chat_dock_widget_qt.py`, `AIAssistantPanel`) remains a documented compatibility shell.
