"""Tests for logging_utils - Logging utilities.

These tests verify the logging utility functions using
Design by Contract principles.
"""

import json
import logging
import random

import pytest


class TestGetLoggerContract:
    """Design by Contract tests for get_logger function."""

    def test_returns_logger(self):
        """Postcondition: Returns a Logger instance."""
        from utils.logging_utils import get_logger

        result = get_logger("test_module")
        assert isinstance(result, logging.Logger)

    def test_logger_has_name(self):
        """Postcondition: Logger has specified name."""
        from utils.logging_utils import get_logger

        result = get_logger("my_custom_name")
        assert result.name == "my_custom_name"


class TestGetLogger:
    """Functional tests for get_logger."""

    def test_creates_logger_with_name(self):
        """Test creating logger with name."""
        from utils.logging_utils import get_logger

        logger = get_logger("test.module.name")
        assert logger.name == "test.module.name"

    def test_accepts_string_level(self):
        """Test accepting string level."""
        from utils.logging_utils import get_logger

        logger = get_logger("test_string_level", level="DEBUG")
        assert logger.level == logging.DEBUG

    def test_accepts_int_level(self):
        """Test accepting integer level."""
        from utils.logging_utils import get_logger

        logger = get_logger("test_int_level", level=logging.WARNING)
        assert logger.level == logging.WARNING

    def test_simple_format_option(self):
        """Test using simple format."""
        from utils.logging_utils import SIMPLE_FORMAT, get_logger

        # Create fresh logger
        logger = get_logger("test_simple_format_unique", use_simple_format=True)

        # Check handler format
        if logger.handlers:
            formatter = logger.handlers[0].formatter
            assert formatter._fmt == SIMPLE_FORMAT


class TestSetupLoggingContract:
    """Design by Contract tests for setup_logging function."""

    def test_does_not_raise(self, tmp_path):
        """Postcondition: Does not raise exceptions."""
        from utils.logging_utils import setup_logging

        # Should not raise
        setup_logging(level="INFO", force=True)


class TestSetupLogging:
    """Functional tests for setup_logging."""

    def test_sets_root_level(self):
        """Test setting root logger level."""
        from utils.logging_utils import setup_logging

        setup_logging(level=logging.WARNING, force=True)
        root = logging.getLogger()
        assert root.level == logging.WARNING

    def test_creates_file_handler(self, tmp_path):
        """Test creating file handler."""
        from utils.logging_utils import setup_logging

        log_file = tmp_path / "app.log"
        setup_logging(level="INFO", log_file=log_file, force=True)

        # Log a message
        logger = logging.getLogger("test_file_handler")
        logger.info("Test message")

        # File should be created
        assert log_file.exists()

    def test_json_format_option(self):
        """Test JSON format option."""
        from utils.logging_utils import JsonFormatter, setup_logging

        setup_logging(json_logs=True, force=True)

        # Check root handler has JSON formatter
        root = logging.getLogger()
        json_formatters = [
            h for h in root.handlers if isinstance(h.formatter, JsonFormatter)
        ]
        assert len(json_formatters) > 0


class TestJsonFormatterContract:
    """Design by Contract tests for JsonFormatter class."""

    def test_format_returns_string(self):
        """Postcondition: format() returns a string."""
        from utils.logging_utils import JsonFormatter

        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        assert isinstance(result, str)

    def test_format_returns_valid_json(self):
        """Postcondition: format() returns valid JSON."""
        from utils.logging_utils import JsonFormatter

        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)


class TestJsonFormatter:
    """Functional tests for JsonFormatter."""

    def test_includes_required_fields(self):
        """Test including required fields."""
        from utils.logging_utils import JsonFormatter

        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="mylogger",
            level=logging.ERROR,
            pathname="test.py",
            lineno=42,
            msg="Error occurred",
            args=(),
            exc_info=None,
        )

        result = json.loads(formatter.format(record))

        assert result["level"] == "ERROR"
        assert result["logger"] == "mylogger"
        assert result["message"] == "Error occurred"
        assert "timestamp" in result

    def test_includes_exception_info(self):
        """Test including exception info."""
        from utils.logging_utils import JsonFormatter

        formatter = JsonFormatter()

        try:
            raise ValueError("Test error")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Error",
            args=(),
            exc_info=exc_info,
        )

        result = json.loads(formatter.format(record))
        assert "exc_info" in result
        assert "ValueError" in result["exc_info"]


class TestSetSeedsContract:
    """Design by Contract tests for set_seeds function."""

    def test_raises_on_negative_seed(self):
        """Precondition: Raises ValueError for negative seed."""
        from utils.logging_utils import set_seeds

        with pytest.raises(ValueError, match="non-negative"):
            set_seeds(-1)


class TestSetSeeds:
    """Functional tests for set_seeds."""

    def test_sets_python_random_seed(self):
        """Test setting Python random seed."""
        from utils.logging_utils import set_seeds

        set_seeds(123)
        first_values = [random.random() for _ in range(5)]

        set_seeds(123)
        second_values = [random.random() for _ in range(5)]

        assert first_values == second_values

    def test_default_seed_is_42(self):
        """Test default seed is 42."""
        from utils.logging_utils import DEFAULT_SEED, set_seeds

        assert DEFAULT_SEED == 42

        set_seeds()  # Use default
        first_values = [random.random() for _ in range(5)]

        set_seeds(42)
        second_values = [random.random() for _ in range(5)]

        assert first_values == second_values

    def test_sets_numpy_seed_when_available(self):
        """Test setting NumPy seed when available."""
        from utils.logging_utils import NUMPY_AVAILABLE, set_seeds

        if not NUMPY_AVAILABLE:
            pytest.skip("NumPy not available")

        import numpy as np

        set_seeds(456)
        first_values = np.random.rand(5).tolist()

        set_seeds(456)
        second_values = np.random.rand(5).tolist()

        assert first_values == second_values


class TestInitDefaultLoggingContract:
    """Design by Contract tests for init_default_logging function."""

    def test_returns_logger(self):
        """Postcondition: Returns a Logger instance."""
        from utils.logging_utils import init_default_logging

        result = init_default_logging()
        assert isinstance(result, logging.Logger)


class TestInitDefaultLogging:
    """Functional tests for init_default_logging."""

    def test_returns_root_logger(self):
        """Test returning root logger."""
        from utils.logging_utils import init_default_logging

        result = init_default_logging()
        assert result is logging.getLogger()

    def test_sets_level(self):
        """Test setting level."""
        from utils.logging_utils import init_default_logging

        logger = init_default_logging(level=logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_accepts_string_level(self):
        """Test accepting string level."""
        from utils.logging_utils import init_default_logging

        logger = init_default_logging(level="WARNING")
        assert logger.level == logging.WARNING
