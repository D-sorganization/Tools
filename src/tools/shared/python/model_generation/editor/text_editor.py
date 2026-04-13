"""Compatibility re-export for text editor module imports."""

from model_generation.editor.text_editor import (
    DiffHunk,
    DiffResult,
    EditorVersion,
    TextEditorDiffMixin,
    URDFTextEditor,
    ValidationMessage,
    ValidationSeverity,
)

TextEditor = URDFTextEditor

__all__ = [
    "DiffHunk",
    "DiffResult",
    "EditorVersion",
    "TextEditorDiffMixin",
    "URDFTextEditor",
    "TextEditor",
    "ValidationMessage",
    "ValidationSeverity",
]
