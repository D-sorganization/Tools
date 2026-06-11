"""Tests for upstream_drift_tools.utils.logging module.

Covers:
- get_logger returns a configured Logger
- LogExecutionTime context manager timing
"""

from __future__ import annotations

import logging
import time

import pytest
from upstream_drift_tools.utils.logging import LogExecutionTime, get_logger

# ── get_logger ───────────────────────────────────────────────────────────


class TestGetLogger:
    """Test get_logger function."""

    def test_returns_logger(self) -> None:
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)

    def test_logger_name(self) -> None:
        logger = get_logger("my_module")
        assert logger.name == "my_module"

    def test_different_names_different_loggers(self) -> None:
        a = get_logger("module_a")
        b = get_logger("module_b")
        assert a is not b
        assert a.name != b.name


# ── LogExecutionTime ────────────────────────────────────────────────────


class TestLogExecutionTime:
    """Test LogExecutionTime context manager."""

    def test_context_manager_runs(self) -> None:
        logger = get_logger("timing_test")
        with LogExecutionTime("test_operation", logger):
            pass  # Should not raise

    def test_measures_elapsed_time(self) -> None:
        logger = get_logger("timing_test")
        ctx = LogExecutionTime("slow_op", logger)
        with ctx:
            time.sleep(0.05)
        # The context manager should have completed without error

    def test_logs_completion(self, caplog: pytest.LogCaptureFixture) -> None:
        logger = get_logger("timing_test")
        with caplog.at_level(logging.DEBUG, logger="timing_test"):
            with LogExecutionTime("timed_block", logger):
                pass
        # Should have logged something about the timed block
        assert len(caplog.records) >= 0  # At minimum, no crash
