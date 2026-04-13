"""Compatibility tests for shared model generation editor import paths."""


def test_tools_shared_python_model_generation_editor_imports() -> None:
    """The compatibility path should expose the same editor type."""
    from tools.shared.python.model_generation.editor.text_editor import (
        TextEditor,
        TextEditorDiffMixin,
        URDFTextEditor,
    )

    assert TextEditor is URDFTextEditor
    assert TextEditorDiffMixin is not None
