"""FileOperationsMixin -- Composite mixin for file operation methods.

This module is a backward-compatible facade that composes the decomposed
mixin classes:
- FileValidationMixin: File filtering, size validation, path organization
- BackupCopyMixin: Backup creation, safe file copy, unique path generation
- ArchiveOperationsMixin: Archive extraction and validation
- FolderOperationsMixin: Folder combine, deduplicate, flatten, prune
"""

from __future__ import annotations

from archive_ops import ArchiveOperationsMixin
from backup_copy import BackupCopyMixin
from file_validation import FileValidationMixin
from folder_ops import FolderOperationsMixin


class FileOperationsMixin(
    FileValidationMixin,
    BackupCopyMixin,
    ArchiveOperationsMixin,
    FolderOperationsMixin,
):
    """File operation methods for FolderProcessorApp.

    Composite mixin that inherits all file operation capabilities from
    focused sub-mixins. The Method Resolution Order (MRO) ensures all
    methods are available through this single class.
    """
