import pytest

from humanoid_character_builder.core.anthropometry import get_segment_mass_ratio, get_urdf_mass_ratios
from urdf_builder_gui.anthropometric_model import MASS_RATIOS as urdf_mass_ratios


def test_anthropometry_mass_ratio_parity() -> None:
    """Ensure all consumers report identical mass ratios for the male model."""
    # 1. Canonical source
    canonical_head = get_segment_mass_ratio("head", 1.0)
    canonical_thigh = get_segment_mass_ratio("thigh", 1.0)
    canonical_pelvis = get_segment_mass_ratio("pelvis", 1.0)

    # 2. GUI Helper (now we just test the canonical via get_urdf_mass_ratios directly)
    helper_masses = get_urdf_mass_ratios(1.0)
    assert helper_masses["head"] == canonical_head
    assert helper_masses["thigh"] == canonical_thigh
    assert helper_masses["pelvis"] == canonical_pelvis

    # 3. URDF Builder (sources from get_urdf_mass_ratios(1.0))
    assert urdf_mass_ratios["head"] == canonical_head
    assert urdf_mass_ratios["thigh"] == canonical_thigh
    assert urdf_mass_ratios["pelvis"] == canonical_pelvis

    # Parity check values
    assert canonical_head == 0.0694
    assert canonical_thigh == 0.1416
    assert canonical_pelvis == 0.1117

    # Check composite segments
    assert urdf_mass_ratios["torso"] == get_segment_mass_ratio("lumbar", 1.0) + get_segment_mass_ratio("thorax", 1.0)
