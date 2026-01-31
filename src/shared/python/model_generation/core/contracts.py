"""Design by Contract Decorators for URDF Model Generation.

This module provides decorators for enforcing contracts at runtime:
- @precondition: Validate inputs before method execution
- @postcondition: Validate outputs after method execution
- @invariant: Validate class state after any method call
- @contract: Combine preconditions and postconditions

These decorators integrate with the existing validation framework,
using ValidationResult for error reporting.

Example:
    @precondition(lambda mass: mass > 0, "Mass must be positive")
    @postcondition(lambda result: result.is_valid, "Result must be valid")
    def create_link(mass: float, ...) -> Link:
        ...
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ParamSpec, TypeVar

from model_generation.core.validation import ValidationResult

P = ParamSpec("P")
R = TypeVar("R")

# Global flag to enable/disable contract checking
CONTRACTS_ENABLED = True


def set_contracts_enabled(enabled: bool) -> None:
    """Enable or disable contract checking globally.

    Args:
        enabled: If False, all contract checks are skipped.
    """
    global CONTRACTS_ENABLED
    CONTRACTS_ENABLED = enabled


@dataclass
class ContractViolation(Exception):
    """Exception raised when a contract is violated."""

    contract_type: str  # "precondition", "postcondition", or "invariant"
    message: str
    function_name: str
    details: dict[str, Any] | None = None

    def __str__(self) -> str:
        return (
            f"{self.contract_type.capitalize()} violation in "
            f"{self.function_name}: {self.message}"
        )


class PreconditionError(ContractViolation):
    """Exception raised when a precondition is violated."""

    def __init__(
        self,
        message: str,
        function_name: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            contract_type="precondition",
            message=message,
            function_name=function_name,
            details=details,
        )


class PostconditionError(ContractViolation):
    """Exception raised when a postcondition is violated."""

    def __init__(
        self,
        message: str,
        function_name: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            contract_type="postcondition",
            message=message,
            function_name=function_name,
            details=details,
        )


class InvariantError(ContractViolation):
    """Exception raised when an invariant is violated."""

    def __init__(
        self,
        message: str,
        function_name: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            contract_type="invariant",
            message=message,
            function_name=function_name,
            details=details,
        )


def precondition(
    condition: Callable[..., bool],
    message: str = "Precondition violated",
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to check preconditions before function execution.

    The condition function receives the same arguments as the decorated function.

    Args:
        condition: Function that takes the same args and returns bool.
        message: Error message if condition fails.

    Example:
        @precondition(lambda x, y: x > 0 and y > 0, "x and y must be positive")
        def divide(x: float, y: float) -> float:
            return x / y
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if CONTRACTS_ENABLED:
                # Get function signature to bind arguments
                sig = inspect.signature(func)
                try:
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()
                    if not condition(*bound.args, **bound.kwargs):
                        raise PreconditionError(
                            message=message,
                            function_name=func.__name__,
                            details={"args": args, "kwargs": kwargs},
                        )
                except TypeError as e:
                    # Condition function might have different signature
                    # Try calling with just positional args
                    try:
                        if not condition(*args):
                            raise PreconditionError(
                                message=message,
                                function_name=func.__name__,
                            )
                    except Exception:
                        raise PreconditionError(
                            message=f"Failed to check precondition: {e}",
                            function_name=func.__name__,
                        )
            return func(*args, **kwargs)

        return wrapper

    return decorator


def postcondition(
    condition: Callable[[R], bool],
    message: str = "Postcondition violated",
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to check postconditions after function execution.

    The condition function receives the return value.

    Args:
        condition: Function that takes return value and returns bool.
        message: Error message if condition fails.

    Example:
        @postcondition(lambda result: result >= 0, "Result must be non-negative")
        def compute_area(width: float, height: float) -> float:
            return width * height
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            result = func(*args, **kwargs)
            if CONTRACTS_ENABLED:
                try:
                    if not condition(result):
                        raise PostconditionError(
                            message=message,
                            function_name=func.__name__,
                            details={"result": str(result)[:100]},
                        )
                except PostconditionError:
                    raise
                except Exception as e:
                    raise PostconditionError(
                        message=f"Failed to check postcondition: {e}",
                        function_name=func.__name__,
                    )
            return result

        return wrapper

    return decorator


def contract(
    pre: Callable[..., bool] | None = None,
    post: Callable[[R], bool] | None = None,
    pre_msg: str = "Precondition violated",
    post_msg: str = "Postcondition violated",
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Combined precondition and postcondition decorator.

    Args:
        pre: Precondition function (receives same args as decorated function)
        post: Postcondition function (receives return value)
        pre_msg: Precondition error message
        post_msg: Postcondition error message

    Example:
        @contract(
            pre=lambda x: x >= 0,
            post=lambda result: result >= 0,
            pre_msg="Input must be non-negative",
            post_msg="Output must be non-negative",
        )
        def sqrt(x: float) -> float:
            return x ** 0.5
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        result_func = func

        if post is not None:
            result_func = postcondition(post, post_msg)(result_func)

        if pre is not None:
            result_func = precondition(pre, pre_msg)(result_func)

        return result_func

    return decorator


def invariant(
    condition: Callable[[Any], bool],
    message: str = "Invariant violated",
) -> Callable[[type], type]:
    """Class decorator to check invariants after method calls.

    The condition function receives self.

    Args:
        condition: Function that takes self and returns bool.
        message: Error message if condition fails.

    Example:
        @invariant(lambda self: len(self.links) > 0, "Model must have links")
        class URDFModel:
            ...
    """

    def class_decorator(cls: type) -> type:
        # Wrap all public methods
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            if not name.startswith("_"):
                setattr(cls, name, _wrap_with_invariant(method, condition, message))

        return cls

    return class_decorator


def _wrap_with_invariant(
    method: Callable[..., Any],
    condition: Callable[[Any], bool],
    message: str,
) -> Callable[..., Any]:
    """Wrap a method to check invariant after execution."""

    @functools.wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = method(self, *args, **kwargs)
        if CONTRACTS_ENABLED:
            try:
                if not condition(self):
                    raise InvariantError(
                        message=message,
                        function_name=f"{self.__class__.__name__}.{method.__name__}",
                    )
            except InvariantError:
                raise
            except Exception as e:
                raise InvariantError(
                    message=f"Failed to check invariant: {e}",
                    function_name=method.__name__,
                )
        return result

    return wrapper


# Convenience functions for common validations


def require_positive(value: float, name: str = "value") -> None:
    """Require a value to be positive.

    Args:
        value: Value to check
        name: Name for error message

    Raises:
        PreconditionError: If value is not positive
    """
    if CONTRACTS_ENABLED and value <= 0:
        raise PreconditionError(f"{name} must be positive (got {value})")


def require_finite(array: Any, name: str = "array") -> None:
    """Require all array elements to be finite.

    Args:
        array: Numpy array to check
        name: Name for error message

    Raises:
        PreconditionError: If any element is NaN or Inf
    """
    import numpy as np

    if CONTRACTS_ENABLED and not np.all(np.isfinite(array)):
        raise PreconditionError(f"{name} contains NaN or Inf values")


def require_unit_vector(vector: Any, name: str = "vector", tol: float = 1e-6) -> None:
    """Require vector to be a unit vector.

    Args:
        vector: Vector to check
        name: Name for error message
        tol: Tolerance for norm check

    Raises:
        PreconditionError: If vector is not unit length
    """
    import numpy as np

    if CONTRACTS_ENABLED:
        norm = np.linalg.norm(vector)
        if abs(norm - 1.0) > tol:
            raise PreconditionError(f"{name} must be a unit vector (norm = {norm})")


def ensure_valid_result(result: ValidationResult) -> None:
    """Ensure a ValidationResult is valid.

    Args:
        result: Validation result to check

    Raises:
        PostconditionError: If result is not valid
    """
    if CONTRACTS_ENABLED and not result.is_valid:
        errors = "; ".join(result.get_error_messages())
        raise PostconditionError(f"Validation failed: {errors}")


# Export common contract patterns as reusable conditions


def is_positive(value: float) -> bool:
    """Check if value is positive."""
    return value > 0


def is_non_negative(value: float) -> bool:
    """Check if value is non-negative."""
    return value >= 0


def is_valid_result(result: ValidationResult) -> bool:
    """Check if validation result is valid."""
    return result.is_valid


def has_finite_elements(array: Any) -> bool:
    """Check if all array elements are finite."""
    import numpy as np

    return np.all(np.isfinite(array))


__all__ = [
    # Decorators
    "precondition",
    "postcondition",
    "contract",
    "invariant",
    # Exceptions
    "ContractViolation",
    "PreconditionError",
    "PostconditionError",
    "InvariantError",
    # Convenience functions
    "require_positive",
    "require_finite",
    "require_unit_vector",
    "ensure_valid_result",
    "set_contracts_enabled",
    # Condition functions
    "is_positive",
    "is_non_negative",
    "is_valid_result",
    "has_finite_elements",
    # Global flag
    "CONTRACTS_ENABLED",
]
