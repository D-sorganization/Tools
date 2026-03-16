"""Unit tests for folder_tool_file_ops."""

from folder_tool.folder_tool_file_ops import (
    ArchiveOperationsMixin,
    BackupCopyMixin,
    FileOperationsMixin,
    FileValidationMixin,
    FolderOperationsMixin,
)


class DummyApp(FileOperationsMixin):
    pass


def test_file_operations_mixin_inheritance():
    app = DummyApp()
    assert isinstance(app, FileOperationsMixin)
    assert isinstance(app, ArchiveOperationsMixin)
    assert isinstance(app, BackupCopyMixin)
    assert isinstance(app, FileValidationMixin)
    assert isinstance(app, FolderOperationsMixin)
