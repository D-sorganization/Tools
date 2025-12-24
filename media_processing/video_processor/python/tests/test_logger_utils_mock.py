import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_torch_available_seeds() -> None:
    """Test set_seeds when torch is available."""
    # Mock torch module
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True

    # Patch sys.modules to include torch
    with patch.dict(sys.modules, {"torch": mock_torch}):
        # Import (or reload) logger_utils to trigger torch import detection
        import src.logger_utils

        importlib.reload(src.logger_utils)

        # Verify TORCH_AVAILABLE is True
        assert src.logger_utils.TORCH_AVAILABLE is True

        # Call set_seeds
        src.logger_utils.set_seeds(123)

        # Verify torch functions were called
        mock_torch.manual_seed.assert_called_with(123)
        mock_torch.cuda.manual_seed_all.assert_called_with(123)
        mock_torch.cuda.manual_seed.assert_called_with(123)

    # Cleanup: reload logger_utils without mock to restore original state
    # (assuming torch is NOT installed in the environment, which caused the missing coverage)
    importlib.reload(src.logger_utils)
    # Note: If torch IS installed in the env, this reload will set TORCH_AVAILABLE=True again,
    # which is fine. The test assumes we want to force True to test that branch.
