"""Focused coverage for shared logging helpers."""

from __future__ import annotations

import logging
import random
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from logging_pkg import (
    DEFAULT_SEED,
    LOG_FORMAT,
    LOG_LEVEL,
    LogLevel,
    SensitiveDataFilter,
    add_file_handler,
    add_rotating_file_handler,
    configure_gui_logging,
    configure_test_logging,
    get_logger,
    log_execution_time,
    set_seeds,
    setup_logging,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def restore_logging_state() -> Generator[None]:
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    original_library_levels = {
        name: logging.getLogger(name).level
        for name in (
            "matplotlib",
            "matplotlib.font_manager",
            "PIL",
            "urllib3",
            "asyncio",
        )
    }
    yield
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    for handler in original_handlers:
        root.addHandler(handler)
    root.setLevel(original_level)
    for name, level in original_library_levels.items():
        logging.getLogger(name).setLevel(level)
    logging.captureWarnings(False)


class RecordingLogger:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def info(self, message: str, *args: Any) -> None:
        self.calls.append((message, args))


def test_public_package_exports_are_importable() -> None:
    assert DEFAULT_SEED == 42
    assert LOG_FORMAT == "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    assert LOG_LEVEL == logging.INFO
    assert LogLevel.ERROR.value == logging.ERROR
    assert get_logger("tools.test").name == "tools.test"


def test_sensitive_data_filter_formats_args_and_redacts_values() -> None:
    record = logging.LogRecord(
        name="tools",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="password=%s api-secret: %s private_key=%s",
        args=("abc123", "token-value", "key-material"),
        exc_info=None,
    )

    assert SensitiveDataFilter().filter(record) is True

    assert record.args is None
    expected = (
        "password=***REDACTED*** api-secret=***REDACTED*** private_key=***REDACTED***"
    )
    assert record.msg == expected


def test_setup_logging_configures_stream_format_redaction_and_quiet_libraries(
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = setup_logging(
        level=LogLevel.DEBUG,
        use_simple_format=True,
        force=True,
        quiet_libraries=("urllib3",),
        use_qt_handler=True,
    )

    assert root is logging.getLogger()
    assert root.level == logging.DEBUG
    assert len(root.handlers) == 1
    assert any(
        isinstance(item, SensitiveDataFilter) for item in root.handlers[0].filters
    )
    setup_logging(level=logging.ERROR, use_simple_format=True)
    assert (
        sum(isinstance(item, SensitiveDataFilter) for item in root.handlers[0].filters)
        == 1
    )
    assert logging.getLogger("urllib3").level == logging.WARNING
    assert logging.getLogger("matplotlib").level == logging.WARNING
    assert logging.getLogger("matplotlib.font_manager").level == logging.WARNING
    assert logging.getLogger("PIL").level == logging.WARNING

    logging.getLogger("tools.test").debug("api_key=%s", "secret-value")
    assert "DEBUG: api_key=***REDACTED***" in capsys.readouterr().err


def test_setup_logging_honors_custom_file_configuration(tmp_path: Path) -> None:
    log_path = tmp_path / "custom.log"

    root = setup_logging(
        level=logging.WARNING,
        filename=log_path,
        filemode="w",
        format_string="%(levelname)s|%(message)s",
        force=True,
        enable_redaction=False,
    )

    root.warning("secret_token=%s", "visible-token")
    for handler in root.handlers:
        handler.flush()

    assert log_path.read_text(encoding="utf-8").strip() == (
        "WARNING|secret_token=visible-token"
    )
    assert all(
        not any(isinstance(item, SensitiveDataFilter) for item in handler.filters)
        for handler in root.handlers
    )


def test_convenience_logging_configurations_set_expected_defaults() -> None:
    test_root = configure_test_logging(level=LogLevel.ERROR)
    assert test_root.level == logging.ERROR
    assert logging.getLogger("asyncio").level == logging.WARNING

    for handler in list(logging.getLogger().handlers):
        logging.getLogger().removeHandler(handler)
        handler.close()
    gui_root = configure_gui_logging(level=logging.DEBUG)
    assert gui_root.level == logging.DEBUG
    assert logging.getLogger("matplotlib").level == logging.WARNING
    assert logging.getLogger("PIL").level == logging.WARNING


def test_file_handlers_attach_formatters_levels_and_redaction(tmp_path: Path) -> None:
    logger = logging.getLogger("tools.file-handler-test")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.DEBUG)

    plain_path = tmp_path / "plain.log"
    plain = add_file_handler(
        logger,
        plain_path,
        level=LogLevel.INFO,
        format_string="%(levelname)s:%(message)s",
    )
    logger.info("access_token=%s", "abc123")
    plain.flush()
    plain.close()
    logger.removeHandler(plain)

    assert plain.level == logging.INFO
    assert plain_path.read_text(encoding="utf-8").strip() == (
        "INFO:access_token=***REDACTED***"
    )

    rotating_path = tmp_path / "rotating.log"
    rotating = add_rotating_file_handler(
        logger,
        rotating_path,
        level=logging.WARNING,
        format_string="%(message)s",
        max_bytes=128,
        backup_count=2,
        enable_redaction=False,
    )
    logger.warning("password=%s", "visible")
    rotating.flush()
    rotating.close()
    logger.removeHandler(rotating)

    assert rotating.level == logging.WARNING
    assert rotating.maxBytes == 128
    assert rotating.backupCount == 2
    assert rotating_path.read_text(encoding="utf-8").strip() == "password=visible"


def test_set_seeds_validates_and_seeds_random_generators() -> None:
    with pytest.raises(ValueError, match="expected non-negative integer"):
        set_seeds(-1)

    set_seeds(123)
    first = random.random()
    set_seeds(123)
    assert random.random() == first

    set_seeds(456, validate=False)
    assert isinstance(random.random(), float)


def test_log_execution_time_logs_success_and_exceptions() -> None:
    logger = RecordingLogger()

    with log_execution_time("operation", logger):
        pass

    assert logger.calls
    message, args = logger.calls[-1]
    assert message == "Telemetry: %s took %.4f seconds"
    assert args[0] == "operation"
    assert isinstance(args[1], float)
    assert args[1] >= 0.0

    raised = False
    try:
        with log_execution_time("failure", logger):
            msg = "boom"
            raise RuntimeError(msg)
    except RuntimeError as exc:
        raised = True
        assert str(exc) == "boom"

    assert raised is True
    assert logger.calls[-1][1][0] == "failure"


def test_log_execution_time_rejects_empty_operation_name() -> None:
    with pytest.raises(ValueError, match="operation_name must be a non-empty string"):
        with log_execution_time(""):
            pass
