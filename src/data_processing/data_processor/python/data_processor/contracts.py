"""Design by Contract (DbC) adapter for the data_processor package.

Re-exports the fleet-standard contract primitives from ``contracts``
with a transparent fallback so the module works both as part of the Tools monorepo
(editable install) and as a standalone installation.

Usage::

    from data_processor.contracts import require, ensure, PreconditionError

    def load_csv(file_path: str) -> pd.DataFrame:
        require(bool(file_path), "file_path must be non-empty", file_path)
        require(file_path.endswith(".csv"), "file_path must be a .csv file", file_path)
        ...

De-duplicated per https://github.com/D-sorganization/Tools/issues/1926 --
fallback now provides only the minimal stubs needed for standalone use.
"""

from __future__ import annotations

# Try the fleet-shared contracts module first (monorepo editable install)
try:
    from contracts import (
        ContractLevel,
        ContractViolationError,
        InvariantError,
        PostconditionError,
        PreconditionError,
        check_non_negative,
        check_positive,
        check_range,
        contract,
        ensure,
        get_contract_level,
        invariant,
        postcondition,
        precondition,
        require,
        require_finite,
        require_positive,
        set_contract_level,
    )
except ImportError:
    # Standalone fallback -- minimal stubs so the package works without the
    # full monorepo.  Only the core primitives are implemented; decorators
    # are identity no-ops.
    import enum
    import os
    from typing import Any

    class ContractLevel(enum.Enum):  # type: ignore[no-redef]
        OFF = "off"
        WARN = "warn"
        ENFORCE = "enforce"

    _LEVEL = ContractLevel(
        os.environ.get("DBC_LEVEL", "enforce").lower()
        if os.environ.get("DBC_LEVEL", "").lower() in ("off", "warn", "enforce")
        else "enforce"
    )

    class ContractViolationError(AssertionError, ValueError):  # type: ignore[no-redef]
        def __init__(self, kind: str, msg: str, value: Any = None) -> None:
            detail = f"[DbC {kind}] {msg}"
            if value is not None:
                detail += f" (got: {value!r})"
            super().__init__(detail)

    class PreconditionError(ContractViolationError):  # type: ignore[no-redef]
        def __init__(self, msg: str, value: Any = None) -> None:
            super().__init__("pre-condition", msg, value)

    class PostconditionError(ContractViolationError):  # type: ignore[no-redef]
        def __init__(self, msg: str, value: Any = None) -> None:
            super().__init__("post-condition", msg, value)

    class InvariantError(ContractViolationError):  # type: ignore[no-redef]
        def __init__(self, msg: str, value: Any = None) -> None:
            super().__init__("invariant", msg, value)

    def _fail(kind: str, msg: str, value: Any = None) -> None:
        if _LEVEL == ContractLevel.ENFORCE:
            exc_map = {
                "pre-condition": PreconditionError,
                "post-condition": PostconditionError,
                "invariant": InvariantError,
            }
            raise exc_map.get(kind, ContractViolationError)(msg, value)

    def require(condition: bool, message: str, value: Any = None) -> None:  # type: ignore[misc]
        if _LEVEL == ContractLevel.OFF:
            return
        if not condition:
            _fail("pre-condition", message, value)

    def ensure(condition: bool, message: str, value: Any = None) -> None:  # type: ignore[misc]
        if _LEVEL == ContractLevel.OFF:
            return
        if not condition:
            _fail("post-condition", message, value)

    def invariant(condition: bool, message: str, value: Any = None) -> None:  # type: ignore[misc]
        if _LEVEL == ContractLevel.OFF:
            return
        if not condition:
            _fail("invariant", message, value)

    def require_positive(value: float, name: str = "value") -> None:  # type: ignore[misc]
        require(value > 0, f"{name} must be positive", value)

    def check_positive(value: float, name: str = "value") -> None:  # type: ignore[misc]
        require_positive(value, name)

    def check_non_negative(value: float, name: str = "value") -> None:  # type: ignore[misc]
        require(value >= 0, f"{name} must be non-negative", value)

    def check_range(  # type: ignore[misc]
        value: float, low: float, high: float, name: str = "value"
    ) -> None:
        require(low <= value <= high, f"{name} must be in [{low}, {high}]", value)

    def require_finite(array: Any, name: str = "array") -> None:  # type: ignore[misc]
        import numpy as np

        if not np.all(np.isfinite(array)):
            raise PreconditionError(f"{name} contains NaN or Inf values")

    def get_contract_level() -> ContractLevel:  # type: ignore[misc]
        return _LEVEL

    def set_contract_level(level: ContractLevel) -> None:  # type: ignore[misc]
        pass

    def precondition(condition: Any, message: str = "") -> Any:  # type: ignore[misc]
        return lambda fn: fn

    def postcondition(condition: Any, message: str = "") -> Any:  # type: ignore[misc]
        return lambda fn: fn

    def contract(**kwargs: Any) -> Any:  # type: ignore[misc]
        return lambda fn: fn


__all__ = [
    "ContractLevel",
    "ContractViolationError",
    "InvariantError",
    "PostconditionError",
    "PreconditionError",
    "check_non_negative",
    "check_positive",
    "check_range",
    "contract",
    "ensure",
    "get_contract_level",
    "invariant",
    "postcondition",
    "precondition",
    "require",
    "require_finite",
    "require_positive",
    "set_contract_level",
]
