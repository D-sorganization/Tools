"""Glossary entry type — shared by the split data modules."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["GlossaryEntry"]


@dataclass(frozen=True)
class GlossaryEntry:
    """One glossary term.

    Attributes:
        term: Title Case display name.
        definition: 1-3 sentence sourced definition.
    """

    term: str
    definition: str
