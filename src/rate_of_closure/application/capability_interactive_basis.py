"""Exact capability basis supported by the current interactive controls."""

from shared.python.swing_sim.flight.capability_contract import ClubCapability

CANONICAL_INTERACTIVE_PARAMETERS = (
    ("ball_speed", "m/s"),
    ("launch_angle", "deg"),
    ("launch_direction", "deg"),
)
_INTERACTIVE_DIMENSION = len(CANONICAL_INTERACTIVE_PARAMETERS)


def validate_capability_interactive_basis(club: ClubCapability) -> None:
    """Reject any basis the controls cannot edit without transformation."""
    if club.matrix_kind != "correlation":
        raise ValueError("interactive workflow requires a correlation matrix")
    actual = tuple((item.parameter_id, item.unit) for item in club.parameters)
    if actual != CANONICAL_INTERACTIVE_PARAMETERS:
        raise ValueError(
            "interactive workflow requires canonical parameter order and units: "
            "ball_speed m/s, launch_angle deg, launch_direction deg"
        )
    if len(club.matrix) != _INTERACTIVE_DIMENSION or any(
        len(row) != _INTERACTIVE_DIMENSION for row in club.matrix
    ):
        raise ValueError("interactive workflow requires a 3x3 correlation matrix")


__all__ = [
    "CANONICAL_INTERACTIVE_PARAMETERS",
    "validate_capability_interactive_basis",
]
