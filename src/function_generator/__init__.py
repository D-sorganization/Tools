"""Function Generator Module.

Provides standalone GUI interfaces for signal/waveform generation.
Uses the shared SignalGenerator engine from signal_toolkit.
"""

__all__ = [
    "Signal",
    "SignalGenerator",
]


def __getattr__(name: str) -> object:
    """Load signal-toolkit symbols on demand for headless sidebar imports."""
    if name in __all__:
        from signal_toolkit import Signal, SignalGenerator

        return {"Signal": Signal, "SignalGenerator": SignalGenerator}[name]
    raise AttributeError(name)
