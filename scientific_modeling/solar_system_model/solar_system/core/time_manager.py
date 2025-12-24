"""
Time Manager
============

Manages simulation time, including conversion between different time systems,
time acceleration, and synchronization with real time.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum

from .constants import J2000, SECONDS_PER_DAY


class TimeScale(Enum):
    """Time scales used in astronomical calculations."""

    UTC = "utc"  # Coordinated Universal Time
    TT = "tt"  # Terrestrial Time
    TDB = "tdb"  # Barycentric Dynamical Time
    JD = "julian_date"  # Julian Date


@dataclass
class SimulationTime:
    """
    Represents a point in simulation time with multiple representations.
    """

    julian_date: float
    datetime_utc: datetime
    year: float  # Decimal year (e.g., 2024.5)

    @classmethod
    def from_julian_date(cls, jd: float) -> "SimulationTime":
        """Create SimulationTime from Julian date."""
        dt = cls.julian_to_datetime(jd)
        year = cls.julian_to_decimal_year(jd)
        return cls(julian_date=jd, datetime_utc=dt, year=year)

    @classmethod
    def from_datetime(cls, dt: datetime) -> "SimulationTime":
        """Create SimulationTime from datetime."""
        jd = cls.datetime_to_julian(dt)
        year = cls.julian_to_decimal_year(jd)
        return cls(julian_date=jd, datetime_utc=dt, year=year)

    @staticmethod
    def datetime_to_julian(dt: datetime) -> float:
        """
        Convert datetime to Julian Date.

        Algorithm from Meeus, "Astronomical Algorithms", 2nd ed.
        """
        year = dt.year
        month = dt.month
        day = dt.day + dt.hour / 24.0 + dt.minute / 1440.0 + dt.second / 86400.0

        if month <= 2:
            year -= 1
            month += 12

        century = int(year / 100)
        correction = 2 - century + int(century / 4)

        jd = (
            int(365.25 * (year + 4716))
            + int(30.6001 * (month + 1))
            + day
            + correction
            - 1524.5
        )

        return jd

    @staticmethod
    def julian_to_datetime(jd: float) -> datetime:
        """
        Convert Julian Date to datetime.

        Algorithm from Meeus, "Astronomical Algorithms", 2nd ed.
        """
        jd = jd + 0.5
        z_val = int(jd)
        fractional = jd - z_val

        if z_val < 2299161:
            intermediate_a = z_val
        else:
            alpha = int((z_val - 1867216.25) / 36524.25)
            intermediate_a = z_val + 1 + alpha - int(alpha / 4)

        b_val = intermediate_a + 1524
        c_val = int((b_val - 122.1) / 365.25)
        d_val = int(365.25 * c_val)
        e_val = int((b_val - d_val) / 30.6001)

        day_frac = b_val - d_val - int(30.6001 * e_val) + fractional
        day = int(day_frac)
        frac = day_frac - day

        month = e_val - 1 if e_val < 14 else e_val - 13

        year = c_val - 4716 if month > 2 else c_val - 4715

        # Convert fractional day to hours, minutes, seconds
        hours_frac = frac * 24
        hour = int(hours_frac)
        minutes_frac = (hours_frac - hour) * 60
        minute = int(minutes_frac)
        second = int((minutes_frac - minute) * 60)

        try:
            return datetime(year, month, day, hour, minute, second, tzinfo=UTC)
        except ValueError:
            # Handle edge cases
            return datetime(2000, 1, 1, 12, 0, 0, tzinfo=UTC)

    @staticmethod
    def julian_to_decimal_year(jd: float) -> float:
        """Convert Julian Date to decimal year."""
        # Approximate conversion
        return 2000.0 + (jd - J2000) / 365.25

    def format_date(self) -> str:
        """Format as readable date string."""
        return self.datetime_utc.strftime("%Y-%m-%d %H:%M:%S UTC")

    def format_compact(self) -> str:
        """Format as compact date string."""
        return self.datetime_utc.strftime("%Y-%m-%d")


class TimeManager:
    """
    Manages the simulation clock and time controls.

    Features:
    - Real-time or accelerated simulation
    - Pause/resume functionality
    - Jump to specific dates
    - Time warp with various factors
    """

    # Preset time warp factors
    WARP_FACTORS = {
        "Real-time": 1.0,
        "1 min/sec": 60.0,
        "1 hour/sec": 3600.0,
        "1 day/sec": SECONDS_PER_DAY,
        "1 week/sec": 7 * SECONDS_PER_DAY,
        "1 month/sec": 30 * SECONDS_PER_DAY,
        "1 year/sec": 365.25 * SECONDS_PER_DAY,
    }

    def __init__(self, start_time: SimulationTime | None = None):
        """
        Initialize the time manager.

        Args:
            start_time: Initial simulation time (defaults to current time)
        """
        if start_time is None:
            # Start at current real time
            start_time = SimulationTime.from_datetime(datetime.now(UTC))

        self._simulation_time = start_time
        self._time_warp = 1.0  # Time multiplier
        self._paused = False
        self._last_update = time.time()

        # Callbacks for time changes
        self._on_time_change: list[Callable[[SimulationTime], None]] = []

        # Time bounds (for safety)
        self._min_julian_date = 2378497.0  # Year 1800
        self._max_julian_date = 2524594.0  # Year 2200

    @property
    def current_time(self) -> SimulationTime:
        """Get the current simulation time."""
        return self._simulation_time

    @property
    def julian_date(self) -> float:
        """Get current Julian Date."""
        return self._simulation_time.julian_date

    @property
    def time_warp(self) -> float:
        """Get current time warp factor."""
        return self._time_warp

    @time_warp.setter
    def time_warp(self, value: float):
        """Set time warp factor."""
        self._time_warp = max(
            -365.25 * SECONDS_PER_DAY, min(value, 365.25 * SECONDS_PER_DAY)
        )

    @property
    def is_paused(self) -> bool:
        """Check if simulation is paused."""
        return self._paused

    def update(self) -> float:
        """
        Update simulation time based on elapsed real time.

        Returns:
            Change in Julian date since last update
        """
        current_real_time = time.time()
        delta_real = current_real_time - self._last_update
        self._last_update = current_real_time

        if self._paused:
            return 0.0

        # Calculate simulation time change
        delta_sim_seconds = delta_real * self._time_warp
        delta_jd = delta_sim_seconds / SECONDS_PER_DAY

        # Update simulation time
        new_jd = self._simulation_time.julian_date + delta_jd

        # Clamp to bounds
        new_jd = max(self._min_julian_date, min(new_jd, self._max_julian_date))

        self._simulation_time = SimulationTime.from_julian_date(new_jd)

        # Notify listeners
        for callback in self._on_time_change:
            callback(self._simulation_time)

        return delta_jd

    def pause(self):
        """Pause the simulation."""
        self._paused = True

    def resume(self):
        """Resume the simulation."""
        self._paused = False
        self._last_update = time.time()

    def toggle_pause(self) -> bool:
        """Toggle pause state. Returns new paused state."""
        if self._paused:
            self.resume()
        else:
            self.pause()
        return self._paused

    def set_time(self, sim_time: SimulationTime):
        """
        Set simulation to a specific time.

        Args:
            sim_time: Target simulation time
        """
        jd = sim_time.julian_date
        jd = max(self._min_julian_date, min(jd, self._max_julian_date))

        self._simulation_time = SimulationTime.from_julian_date(jd)
        self._last_update = time.time()

        for callback in self._on_time_change:
            callback(self._simulation_time)

    def set_julian_date(self, jd: float):
        """Set simulation to a specific Julian date."""
        self.set_time(SimulationTime.from_julian_date(jd))

    def set_datetime(self, dt: datetime):
        """Set simulation to a specific datetime."""
        self.set_time(SimulationTime.from_datetime(dt))

    def set_to_now(self):
        """Set simulation to current real time."""
        self.set_datetime(datetime.now(UTC))

    def set_to_j2000(self):
        """Set simulation to J2000.0 epoch."""
        self.set_julian_date(J2000)

    def advance_days(self, days: float):
        """Advance simulation by specified number of days."""
        new_jd = self._simulation_time.julian_date + days
        self.set_julian_date(new_jd)

    def advance_years(self, years: float):
        """Advance simulation by specified number of years."""
        self.advance_days(years * 365.25)

    def set_time_warp_preset(self, preset_name: str) -> bool:
        """
        Set time warp to a preset value.

        Args:
            preset_name: Name of preset from WARP_FACTORS

        Returns:
            True if preset was found and set
        """
        if preset_name in self.WARP_FACTORS:
            self._time_warp = self.WARP_FACTORS[preset_name]
            return True
        return False

    def increase_time_warp(self, factor: float = 10.0):
        """Increase time warp by a factor."""
        if self._time_warp >= 0:
            self._time_warp *= factor
        else:
            self._time_warp /= factor

    def decrease_time_warp(self, factor: float = 10.0):
        """Decrease time warp by a factor."""
        if self._time_warp >= 0:
            self._time_warp /= factor
        else:
            self._time_warp *= factor

    def reverse_time(self):
        """Reverse the direction of time flow."""
        self._time_warp = -self._time_warp

    def add_time_change_listener(self, callback: Callable[[SimulationTime], None]):
        """Add a callback to be notified of time changes."""
        self._on_time_change.append(callback)

    def remove_time_change_listener(self, callback: Callable[[SimulationTime], None]):
        """Remove a time change callback."""
        if callback in self._on_time_change:
            self._on_time_change.remove(callback)

    def get_time_warp_string(self) -> str:
        """Get human-readable time warp description."""
        warp = abs(self._time_warp)
        direction = " (reverse)" if self._time_warp < 0 else ""

        if warp < 1:
            return f"{warp:.2f}x{direction}"
        elif warp < 60:
            return f"{warp:.1f}x{direction}"
        elif warp < 3600:
            return f"{warp/60:.1f} min/sec{direction}"
        elif warp < SECONDS_PER_DAY:
            return f"{warp/3600:.1f} hr/sec{direction}"
        elif warp < 7 * SECONDS_PER_DAY:
            return f"{warp/SECONDS_PER_DAY:.1f} day/sec{direction}"
        elif warp < 30 * SECONDS_PER_DAY:
            return f"{warp/(7*SECONDS_PER_DAY):.1f} week/sec{direction}"
        elif warp < 365.25 * SECONDS_PER_DAY:
            return f"{warp/(30*SECONDS_PER_DAY):.1f} month/sec{direction}"
        else:
            return f"{warp/(365.25*SECONDS_PER_DAY):.1f} year/sec{direction}"

    def format_current_time(self) -> str:
        """Get formatted current simulation time."""
        return self._simulation_time.format_date()

    def get_status_string(self) -> str:
        """Get complete status string for display."""
        status = self.format_current_time()
        if self._paused:
            status += " [PAUSED]"
        else:
            status += f" [{self.get_time_warp_string()}]"
        return status
