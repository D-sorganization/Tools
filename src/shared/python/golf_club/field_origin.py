"""Per-field origin enum and scalar origin value record."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class FieldOrigin(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    """Provenance category for an engineering state field."""

    MEASURED = "measured"
    IDENTIFIED = "identified"
    PRESCRIBED = "prescribed"
    SYNTHETIC = "synthetic"
    ABSENT = "absent"

    @classmethod
    def parse(cls, value: object) -> FieldOrigin:
        """Parse an exact member value or FieldOrigin instance.

        Accepts only an exact lowercase string member value or a FieldOrigin.
        Anything else (other types, unknown strings, different case, surrounding
        whitespace) raises ValueError naming the allowed values without coercion.

        Parameters
        ----------
        value : object
            Input to parse as FieldOrigin.

        Returns
        -------
        FieldOrigin
            The parsed FieldOrigin member.

        Raises
        ------
        ValueError
            If value cannot be parsed as an exact FieldOrigin member.
        """
        if isinstance(value, cls):
            return value
        if type(value) is str:  # exact str only; no str subclasses or coercion
            try:
                return cls(value)
            except ValueError:
                pass
        allowed = tuple(member.value for member in cls)
        raise ValueError(
            f"Invalid field origin {value!r}; allowed values are {allowed}"
        )


@dataclass(frozen=True)
class OriginValue:
    """A floating-point scalar value paired with its origin provenance.

    Parameters
    ----------
    origin : FieldOrigin
        Provenance classification of the field value.
    value : float | None, default=None
        Scalar magnitude. Must be None when origin is ABSENT; must be a finite
        float for all other origins (integers are converted to float).
    """

    origin: FieldOrigin
    value: float | None = None

    def __post_init__(self) -> None:
        raw_origin: object = self.origin
        if not isinstance(raw_origin, FieldOrigin):
            origin_type = type(raw_origin).__name__
            raise TypeError(f"origin must be a FieldOrigin instance, got {origin_type}")
        if self.origin is FieldOrigin.ABSENT:
            if self.value is not None:
                raise ValueError("ABSENT origin requires value to be None")
        else:
            if self.value is None:
                raise ValueError(
                    f"{self.origin.value} origin requires a finite float value, "
                    "got None"
                )
            if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
                raise ValueError(
                    f"{self.origin.value} origin requires a finite real number, "
                    f"got {self.value!r}"
                )
            num = float(self.value)
            if not math.isfinite(num):
                raise ValueError(
                    f"{self.origin.value} origin requires a finite float, got {num!r}"
                )
            object.__setattr__(self, "value", num)

    @property
    def number(self) -> float:
        """Return the numeric value, raising ValueError if origin is ABSENT."""
        if self.origin is FieldOrigin.ABSENT or self.value is None:
            raise ValueError(
                f"Field origin {self.origin.value} is absent; no numeric value exists"
            )
        return self.value


__all__ = [
    "FieldOrigin",
    "OriginValue",
]
