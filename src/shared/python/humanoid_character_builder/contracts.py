"""
Design by Contract decorators for Humanoid Character Builder.

This module provides decorators for enforcing preconditions, postconditions,
and invariants to improve code reliability and correctness.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")
R = TypeVar("R")


class ContractViolationError(AssertionError):
    """Exception raised when a contract is violated."""


def precondition(
    condition: Callable[..., bool], message: str = "Precondition failed"
) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """
    Decorator to enforce a precondition on a function or method.

    The condition function receives the arguments of the decorated function.
    It attempts to bind arguments by name if possible.
    """

    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> R:
            # Bind arguments to parameter names of the decorated function
            sig = inspect.signature(func)
            try:
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                arguments: dict[str, Any] = dict(bound_args.arguments)
            except TypeError:
                # Fallback for when binding fails
                arguments = {}

            # Prepare arguments for the condition function
            cond_sig = inspect.signature(condition)
            call_args = {}

            # Check if condition accepts *args or **kwargs
            accepts_varargs = any(
                p.kind == inspect.Parameter.VAR_POSITIONAL
                for p in cond_sig.parameters.values()
            )
            accepts_varkw = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in cond_sig.parameters.values()
            )

            if accepts_varargs or accepts_varkw:
                # If condition accepts varargs, pass everything
                try:
                    if not condition(*args, **kwargs):
                        raise ContractViolationError(
                            f"{message}. Args: {args}, Kwargs: {kwargs}"
                        )
                except Exception as e:
                    if isinstance(e, ContractViolationError):
                        raise
                    raise ContractViolationError(
                        f"Error checking precondition '{message}': {e}"
                    ) from e
            else:
                # Match arguments by name
                for name in cond_sig.parameters:
                    if name in arguments:
                        call_args[name] = arguments[name]

                try:
                    if not condition(**call_args):
                        raise ContractViolationError(
                            f"{message}. Arguments: {call_args}"
                        )
                except TypeError as e:
                    # This usually happens if we missed an argument
                    raise ContractViolationError(
                        f"Error checking precondition '{message}': {e}. Available args: {list(arguments.keys())}"
                    ) from e
                except Exception as e:
                    if isinstance(e, ContractViolationError):
                        raise
                    raise ContractViolationError(
                        f"Error checking precondition '{message}': {e}"
                    ) from e

            return func(*args, **kwargs)

        return wrapper

    return decorator


def postcondition(
    condition: Callable[[R], bool], message: str = "Postcondition failed"
) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """
    Decorator to enforce a postcondition on a function or method.

    The condition function receives the result of the decorated function.
    """

    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> R:
            result = func(*args, **kwargs)
            try:
                if not condition(result):
                    raise ContractViolationError(f"{message}. Result: {result}")
            except Exception as e:
                if isinstance(e, ContractViolationError):
                    raise
                raise ContractViolationError(
                    f"Error checking postcondition '{message}': {e}"
                ) from e
            return result

        return wrapper

    return decorator


def invariant(
    condition: Callable[[Any], bool], message: str = "Invariant failed"
) -> Callable[[type], type]:
    """
    Decorator to enforce an invariant on a class.

    Checks the invariant after __init__ and after every public method call.
    The condition receives the instance (self).
    """

    def decorator(cls: type) -> type:
        # Wrap __init__
        original_init = cls.__init__  # type: ignore[misc]

        @functools.wraps(original_init)
        def new_init(self: Any, *args: Any, **kwargs: Any) -> None:
            original_init(self, *args, **kwargs)
            try:
                if not condition(self):
                    raise ContractViolationError(f"{message} (after __init__)")
            except Exception as e:
                if isinstance(e, ContractViolationError):
                    raise
                raise ContractViolationError(
                    f"Error checking invariant '{message}' after __init__: {e}"
                ) from e

        cls.__init__ = new_init  # type: ignore

        # Wrap all public methods
        for name, method in inspect.getmembers(cls, inspect.isfunction):
            if not name.startswith("_") and name != "__init__":
                # We need to capture the original method properly
                def make_wrapper(
                    orig_method: Callable[..., Any], method_name: str
                ) -> Callable[..., Any]:
                    @functools.wraps(orig_method)
                    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
                        result = orig_method(self, *args, **kwargs)
                        try:
                            if not condition(self):
                                raise ContractViolationError(
                                    f"{message} (after {method_name})"
                                )
                        except Exception as e:
                            if isinstance(e, ContractViolationError):
                                raise
                            raise ContractViolationError(
                                f"Error checking invariant '{message}' after {method_name}: {e}"
                            ) from e
                        return result

                    return wrapper

                setattr(cls, name, make_wrapper(method, name))

        return cls

    return decorator
