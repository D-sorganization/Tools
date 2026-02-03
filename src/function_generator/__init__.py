"""Function Generator Module.

Provides standalone GUI interfaces for signal/waveform generation.
Uses the shared SignalGenerator engine from signal_toolkit.
"""

from shared.python.signal_toolkit import Signal, SignalGenerator

__all__ = [
    "Signal",
    "SignalGenerator",
]
