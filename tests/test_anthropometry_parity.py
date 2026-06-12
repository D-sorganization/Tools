from humanoid_character_builder.core.anthropometry import (
    DE_LEVA_DATA,
    URDF_HEIGHT_RATIOS,
    URDF_MASS_RATIOS,
    get_simple_length_ratios,
    get_simple_mass_ratios,
)
from urdf_builder_gui.anthropometric_model import HEIGHT_RATIOS as urdf_height_ratios
from urdf_builder_gui.anthropometric_model import MASS_RATIOS as urdf_mass_ratios


def test_anthropometry_parity():
    """Ensure anthropometry consumers use the same male ratios."""
    # 1. Canonical data
    canonical_male_head_mass = DE_LEVA_DATA.get_segment_data("head", 1.0).mass_ratio
    canonical_male_thigh_mass = DE_LEVA_DATA.get_segment_data("thigh", 1.0).mass_ratio
    canonical_male_head_length = DE_LEVA_DATA.get_segment_data("head", 1.0).length_ratio
    canonical_male_thigh_length = DE_LEVA_DATA.get_segment_data(
        "thigh", 1.0
    ).length_ratio

    # 2. Simple flat dicts for GUI (humanoid_builder_gui)
    simple_mass = get_simple_mass_ratios(1.0)
    simple_length = get_simple_length_ratios(1.0)

    # 3. URDF builder
    assert urdf_mass_ratios is URDF_MASS_RATIOS
    assert urdf_height_ratios is URDF_HEIGHT_RATIOS

    # Parity checks
    # Mass
    assert canonical_male_head_mass == simple_mass["head"]
    assert simple_mass["head"] == urdf_mass_ratios["head"]

    assert canonical_male_thigh_mass == simple_mass["thigh"]
    assert simple_mass["thigh"] == urdf_mass_ratios["thigh"]

    # Torso composite check
    assert urdf_mass_ratios["torso"] == simple_mass["thorax"] + simple_mass["lumbar"]
    assert (
        urdf_mass_ratios["torso"]
        == DE_LEVA_DATA.get_segment_data("thorax", 1.0).mass_ratio
        + DE_LEVA_DATA.get_segment_data("lumbar", 1.0).mass_ratio
    )

    # Height
    assert canonical_male_head_length == simple_length["head"]
    assert canonical_male_thigh_length == simple_length["thigh"]
    # URDF height ratios use original config definitions, but mass parity is strict.
