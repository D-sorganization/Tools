"""
Electrode Advancement Calculator
===============================
Calculates electrode consumption and slip rates for arc furnaces.
"""

from shared.python.contracts import check_non_negative, require_positive


class ElectrodeAdvancementCalculator:
    """Calculates electrode consumption and advancement."""

    def __init__(self, consumption_rate: float = 0.5) -> None:
        """Initialize parameters.

        Args:
            consumption_rate: Electrode consumption rate in inches per kAh.
                Must be positive.
        """
        require_positive(consumption_rate, "consumption_rate")
        self.consumption_rate = consumption_rate

    def calculate_consumption(self, current_ka: float, time_hrs: float) -> float:
        """
        Calculate electrode consumption.

        Args:
            current_ka: Current in kA (must be non-negative)
            time_hrs: Time in hours (must be non-negative)

        Returns:
            Consumption in inches
        """
        check_non_negative(current_ka, "current_ka")
        check_non_negative(time_hrs, "time_hrs")
        return self.consumption_rate * current_ka * time_hrs
