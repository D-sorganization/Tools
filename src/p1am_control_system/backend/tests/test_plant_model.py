import pytest
from pydantic import ValidationError

from p1am_control_system.backend.plant_model import (
    Area,
    Equipment,
    Plant,
    TagDefinition,
    Unit,
)


def test_tag_definition_dbc_constraints():
    # Valid tag
    tag = TagDefinition(
        name="PI_100",
        tag_type="Real",
        description="Pressure Indicator",
        rw_mode="Read-only",
        scale_factor=100.0,
        register_type="V",
        register_num=3000,
    )
    assert tag.name == "PI_100"

    # Invalid tag type (DbC)
    with pytest.raises(ValidationError):
        TagDefinition(name="Invalid", tag_type="Unknown")


def test_plant_hierarchy():
    # Build a small hierarchy
    tag1 = TagDefinition(name="PI_101", tag_type="Real")
    tag2 = TagDefinition(name="TI_101", tag_type="Real")

    equip = Equipment(name="Pump_A", tags={"PI_101": tag1, "TI_101": tag2})
    unit = Unit(name="Unit_1", equipment={"Pump_A": equip})
    area = Area(name="Area_A", units={"Unit_1": unit})
    plant = Plant(name="Half_Ton_Plant", areas={"Area_A": area})

    # Test LoD Facade method
    assert plant.get_tag("PI_101") == tag1
    assert plant.get_tag("TI_101") == tag2

    # Unknown tag
    with pytest.raises(KeyError):
        plant.get_tag("UNKNOWN_TAG")
