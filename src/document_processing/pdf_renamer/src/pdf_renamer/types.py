"""Type definitions and data classes for the PDF Renamer."""

from dataclasses import dataclass


@dataclass
class TitleResult:
    title: str | None
    confidence: float
    method: str  # "metadata" | "heuristic" | "llm"
    details: str = ""
