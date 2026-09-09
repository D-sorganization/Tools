"""Owned, explicitly normalized displacement input/output channels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._grip_contracts import finite_array


def _matrix(value: object, name: str) -> tuple[tuple[float, ...], ...]:
    shape = np.shape(value)
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError(f"{name} must be a nonempty matrix")
    return tuple(
        tuple(float(item) for item in row) for row in finite_array(value, shape, name)
    )


def _scales(value: object, size: int, name: str) -> tuple[float, ...]:
    values = finite_array(value, (size,), name)
    if np.any(values <= 0):
        raise ValueError(f"{name} must be strictly positive")
    return tuple(float(item) for item in values)


@dataclass(frozen=True)
class DisplacementPorts:
    """Maps and positive scales for dimensionless force-to-displacement transfer.

    In already length-scaled coordinates y, f=force_map @ u and physical
    observations v=observation_map @ y. Each input/output channel has its own
    declared physical unit and positive reference magnitude. With u=diag(su)
    u_hat and v_hat=diag(sv)^-1 v, normalized maps are B=force_map diag(su)
    and C=diag(sv)^-1 observation_map. Thus C D^-1 B is dimensionless.

    Maps must already include the existing work-conjugate coordinate transform:
    for q=S y, use S.T times the SI load map and the SI observation map times S.
    Units and physical identification are caller prescriptions, not inferred
    from numeric arrays. Channels need not be collocated or work conjugates
    of one another. This is displacement response, not velocity or acoustics.
    """

    force_map: object
    observation_map: object
    input_scales: object
    output_scales: object

    def __post_init__(self) -> None:
        force = _matrix(self.force_map, "force map")
        observed = _matrix(self.observation_map, "observation map")
        if len(force) != len(observed[0]):
            raise ValueError(
                "input/output maps must share the full coordinate topology"
            )
        object.__setattr__(self, "force_map", force)
        object.__setattr__(self, "observation_map", observed)
        object.__setattr__(
            self,
            "input_scales",
            _scales(self.input_scales, len(force[0]), "input scales"),
        )
        object.__setattr__(
            self,
            "output_scales",
            _scales(self.output_scales, len(observed), "output scales"),
        )
        self.normalized_arrays()

    def normalized_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return fresh B,C; refuse nonrepresentable channel scaling."""
        try:
            with np.errstate(
                over="raise", invalid="raise", divide="raise", under="raise"
            ):
                inputs = np.array(self.force_map) * np.array(self.input_scales)
                outputs = (
                    np.array(self.observation_map)
                    / np.array(self.output_scales)[:, None]
                )
        except (FloatingPointError, OverflowError) as error:
            raise ValueError("port normalization is not representable") from error
        return inputs, outputs


__all__ = ()
