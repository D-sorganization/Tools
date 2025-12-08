"""Tests for logger_utils.py."""

from unittest.mock import patch

from src.logger_utils import DEFAULT_SEED, set_seeds


class TestLoggerUtils:
    """Test cases for logger_utils.py."""

    def test_set_seeds_default(self) -> None:
        """Test set_seeds with default seed."""
        with patch("random.seed") as mock_random_seed:
            with patch("numpy.random.seed") as mock_np_seed:
                set_seeds()
                mock_random_seed.assert_called_once_with(DEFAULT_SEED)
                mock_np_seed.assert_called_once_with(DEFAULT_SEED)

    def test_set_seeds_custom(self) -> None:
        """Test set_seeds with custom seed."""
        custom_seed = 12345
        with patch("random.seed") as mock_random_seed:
            with patch("numpy.random.seed") as mock_np_seed:
                set_seeds(custom_seed)
                mock_random_seed.assert_called_once_with(custom_seed)
                mock_np_seed.assert_called_once_with(custom_seed)

    def test_set_seeds_numpy_missing(self) -> None:
        """Test set_seeds when numpy is missing."""
        import sys

        # Use simple patch.dict to safely simulate missing module
        # ModuleNotFoundError (raised when None is in sys.modules) inherits from ImportError
        with patch.dict(sys.modules, {"numpy": None}):
            with patch("src.logger_utils.logger") as mock_logger:
                set_seeds()
                mock_logger.warning.assert_called_once()
