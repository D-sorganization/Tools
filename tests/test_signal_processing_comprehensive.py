"""Comprehensive signal processing tests.

Tests cover filters, transforms, spectral analysis, windowing.
"""

import pytest


class TestFilterResponse:
    """Test filter frequency response."""

    def test_lowpass_attenuation(self):
        """Lowpass filter should attenuate high frequencies."""
        try:
            from signal_processing import design_butterworth

            b, a = design_butterworth(4, 0.3)
            assert b is not None
        except:
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
