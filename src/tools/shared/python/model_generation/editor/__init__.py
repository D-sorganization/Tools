"""Compatibility bridge for model generation text editor import paths."""

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
