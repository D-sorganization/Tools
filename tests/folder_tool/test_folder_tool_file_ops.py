"""Unit tests for folder_tool_file_ops."""

from folder_tool.archive_ops import ArchiveOperationsMixin
from folder_tool.backup_copy import BackupCopyMixin
from folder_tool.file_validation import FileValidationMixin
from folder_tool.folder_ops import FolderOperationsMixin
from folder_tool.folder_tool_file_ops import FileOperationsMixin


class DummyApp(FileOperationsMixin):
    pass


def test_file_operations_mixin_inheritance():
    app = DummyApp()
    assert isinstance(app, FileOperationsMixin)
    assert isinstance(app, ArchiveOperationsMixin)
    assert isinstance(app, BackupCopyMixin)
    assert isinstance(app, FileValidationMixin)
    assert isinstance(app, FolderOperationsMixin)
