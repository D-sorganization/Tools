"""test_logger_utils_mock.py module."""

import importlib
import sys
from types import ModuleType
from typing import Any, cast
from unittest.mock import Mock, patch


def test_torch_available_seeds() -> None:
    """Test set_seeds when torch is available."""
    mock_torch = cast(Any, ModuleType("torch"))
    manual_seed = Mock()
    cuda = Mock()
    cuda.is_available.return_value = True
    mock_torch.manual_seed = manual_seed
    mock_torch.cuda = cuda

    with patch.dict(sys.modules, {"torch": mock_torch}):
        import video_processor_src.logger_utils

        importlib.reload(video_processor_src.logger_utils)

        video_processor_src.logger_utils.set_seeds(123)

        assert video_processor_src.logger_utils.TORCH_AVAILABLE is True

        manual_seed.assert_called_with(123)
        cuda.manual_seed_all.assert_called_with(123)
        cuda.manual_seed.assert_called_with(123)

    importlib.reload(video_processor_src.logger_utils)
