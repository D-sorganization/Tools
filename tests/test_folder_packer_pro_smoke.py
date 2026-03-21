"""Smoke tests for folder_packer_pro package importability."""

import pytest


@pytest.mark.unit
def test_folder_packer_pro_imports() -> None:
    """Verify the folder_packer_pro package is importable."""
    from folder_packer_pro import EncryptionManager, FolderPackerPro, PackageManifest

    assert FolderPackerPro is not None
    assert EncryptionManager is not None
    assert PackageManifest is not None
