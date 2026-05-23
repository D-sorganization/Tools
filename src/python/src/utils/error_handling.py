# ruff: noqa: E501
"""
Shared error handling utility for consistent error management.

This module provides reusable functions for common error handling patterns
across the repository, following DRY principles.
"""

import logging
import sys
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def handle_file_errors(
    default: T | None = None,
    log_error: bool = True,
    reraise: bool = False,
) -> Callable:
    """Decorator for handling common file operation errors.

    Args:
        default: Default value to return on error
        log_error: Whether to log errors
        reraise: Whether to re-raise exceptions after handling

    Returns:
        Decorator function
    """

    if log_error is None:
        raise ValueError("log_error must be provided")

    def decorator(func: Callable[..., T]) -> Callable[..., T | None]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T | None:
            try:
                return func(*args, **kwargs)
            except FileNotFoundError as e:
                if log_error:
                    logger.error(f"File not found in {func.__name__}: {e}")
                if reraise:
                    raise
                return default
            except PermissionError as e:
                if log_error:
                    logger.error(f"Permission denied in {func.__name__}: {e}")
                if reraise:
                    raise
                return default
            except OSError as e:
                if log_error:
                    logger.error(f"OS error in {func.__name__}: {e}")
                if reraise:
                    raise
                return default
            except (ValueError, TypeError, RuntimeError, KeyError) as e:
                if log_error:
                    logger.error(f"Unexpected error in {func.__name__}: {e}")
                if reraise:
                    raise
                return default

        return wrapper

    return decorator


def safe_execute(
    func: Callable[..., T],
    *args: Any,
    default: T | None = None,
    log_error: bool = True,
    **kwargs: Any,
) -> T | None:
    """Safely execute a function with error handling.

    Args:
        func: Function to execute
        *args: Positional arguments for function
        default: Default value to return on error
        log_error: Whether to log errors
        **kwargs: Keyword arguments for function

    Returns:
        Function result or default value
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:  # noqa: BLE001 — intentional catch-all; safe_execute must not propagate
        if log_error:
            logger.error(f"Error executing {func.__name__}: {e}")
        return default


def handle_import_error(
    module_name: str,
    package_name: str | None = None,
    default: Any = None,
) -> Any:
    """Handle import errors with user-friendly messages.

    Args:
        module_name: Name of module to import
        package_name: Name of package to install (if different from module)
        default: Default value to return if import fails

    Returns:
        Imported module or default value
    """
    try:
        return __import__(module_name)
    except ImportError:
        install_name = package_name or module_name
        logger.warning(
            f"Module '{module_name}' not found. "
            f"Install with: pip install {install_name}"
        )
        return default


def log_and_continue(
    error_message: str,
    default: T | None = None,
    log_level: int = logging.WARNING,
) -> Callable:
    """Decorator that logs errors and continues execution.

    Args:
        error_message: Message to log on error
        default: Default value to return on error
        log_level: Logging level to use

    Returns:
        Decorator function
    """

    if error_message is None:
        raise ValueError("error_message must be provided")

    def decorator(func: Callable[..., T]) -> Callable[..., T | None]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T | None:
            try:
                return func(*args, **kwargs)
            except (OSError, ValueError, TypeError, RuntimeError, KeyError) as e:
                logger.log(log_level, f"{error_message}: {e}")
                return default

        return wrapper

    return decorator


def exit_on_error(
    error_message: str,
    exit_code: int = 1,
    log_error: bool = True,
) -> Callable:
    """Decorator that exits the program on error.

    Args:
        error_message: Message to display on error
        exit_code: Exit code to use
        log_error: Whether to log errors

    Returns:
        Decorator function
    """

    if error_message is None:
        raise ValueError("error_message must be provided")

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return func(*args, **kwargs)
            except (OSError, ValueError, TypeError, RuntimeError, KeyError) as e:
                if log_error:
                    logger.error(f"{error_message}: {e}")
                logger.error(f"ERROR: {error_message}: {e}")
                sys.exit(exit_code)

        return wrapper

    return decorator
