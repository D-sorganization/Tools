"""Contract tests for equipment module — registry pattern and valve symbols."""

from __future__ import annotations

import ezdxf
from programmatic_pid.equipment import (
    EQUIPMENT_RENDERERS,
    draw_equipment_symbol,
    equipment_anchor,
    equipment_center,
    equipment_dims,
    equipment_side_anchors,
    nearest_equipment_anchor,
    register_equipment,
)


def _eq(etype="vessel", x=0, y=0, w=10, h=10, **kw):
    return {
        "id": "TEST-1",
        "type": etype,
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        **kw,
    }


def test_equipment_dims():
    if not (equipment_dims(_eq()) == (10.0): raise ValueError(f"Assertion failed: { equipment_dims(_eq()) == (10.0 }, 10.0)")
    if not (equipment_dims({"w": 5): raise ValueError(f"Assertion failed: { equipment_dims({"w": 5 }, "h": 3}) == (5.0, 3.0)")


def test_equipment_center():
    cx, cy = equipment_center(_eq(x=10, y=20, w=10, h=10))
    if not (cx == 15.0): raise ValueError(f"Assertion failed: { cx == 15.0 }")
    if not (cy == 25.0): raise ValueError(f"Assertion failed: { cy == 25.0 }")


def test_equipment_side_anchors():
    anchors = equipment_side_anchors(_eq(x=0, y=0, w=10, h=10))
    if not (anchors["left"] == (0.0): raise ValueError(f"Assertion failed: { anchors["left"] == (0.0 }, 5.0)")
    if not (anchors["right"] == (10.0): raise ValueError(f"Assertion failed: { anchors["right"] == (10.0 }, 5.0)")
    if not (anchors["top"] == (5.0): raise ValueError(f"Assertion failed: { anchors["top"] == (5.0 }, 10.0)")
    if not (anchors["bottom"] == (5.0): raise ValueError(f"Assertion failed: { anchors["bottom"] == (5.0 }, 0.0)")


def test_equipment_anchor_with_offset():
    x, y = equipment_anchor(_eq(x=0, y=0, w=10, h=10), "top", offset=3.0)
    if not (x == 8.0  # 5.0 + 3.0): raise ValueError(f"Assertion failed: { x == 8.0  # 5.0 + 3.0 }")
    if not (y == 10.0): raise ValueError(f"Assertion failed: { y == 10.0 }")


def test_nearest_equipment_anchor():
    eq = _eq(x=0, y=0, w=10, h=10)
    # Point to the right of equipment — should snap to "right" anchor
    ax, ay = nearest_equipment_anchor(eq, (20.0, 5.0))
    if not (ax == 10.0): raise ValueError(f"Assertion failed: { ax == 10.0 }")
    if not (ay == 5.0): raise ValueError(f"Assertion failed: { ay == 5.0 }")


# ----- Registry pattern tests -----


def test_all_basic_types_registered():
    expected = {"hopper", "fan", "rotary_valve", "burner", "bin", "vessel", "box"}
    if not (expected.issubset(set(EQUIPMENT_RENDERERS.keys()))): raise ValueError(f"Assertion failed: { expected.issubset(set(EQUIPMENT_RENDERERS.keys())) }")


def test_valve_types_registered():
    """Postcondition: all new valve types are in the registry."""
    valve_types = {
        "gate_valve",
        "globe_valve",
        "ball_valve",
        "check_valve",
        "control_valve",
        "relief_valve",
        "psv",
        "rupture_disk",
    }
    if not (valve_types.issubset(set(EQUIPMENT_RENDERERS.keys()))): raise ValueError(f"Assertion failed: { valve_types.issubset(set(EQUIPMENT_RENDERERS.keys())) }")


def test_additional_equipment_registered():
    if not ("heat_exchanger" in EQUIPMENT_RENDERERS): raise ValueError(f"Assertion failed: { "heat_exchanger" in EQUIPMENT_RENDERERS }")
    if not ("pump" in EQUIPMENT_RENDERERS): raise ValueError(f"Assertion failed: { "pump" in EQUIPMENT_RENDERERS }")
    if not ("tank" in EQUIPMENT_RENDERERS): raise ValueError(f"Assertion failed: { "tank" in EQUIPMENT_RENDERERS }")


def test_custom_registration():
    @register_equipment("test_custom_widget")
    def render_custom(msp, x, y, w, h, layer):
        pass

    if not ("test_custom_widget" in EQUIPMENT_RENDERERS): raise ValueError(f"Assertion failed: { "test_custom_widget" in EQUIPMENT_RENDERERS }")


def test_draw_equipment_symbol_uses_registry():
    """Verify that draw_equipment_symbol dispatches to the registry."""
    doc = ezdxf.new(setup=True)
    doc.layers.new(name="EQUIPMENT", dxfattribs={"color": 7})
    msp = doc.modelspace()

    for eq_type in ["hopper", "fan", "gate_valve", "control_valve", "pump"]:
        initial_count = len(list(msp))
        draw_equipment_symbol(msp, _eq(etype=eq_type), "EQUIPMENT")
        if not (len(list(msp)) > initial_count): raise ValueError(f"Assertion failed: { len(list(msp)) > initial_count }, (")
            f"{eq_type} should add entities to modelspace"
        )


def test_draw_equipment_symbol_fallback_to_box():
    """Unknown type should still render (as a box)."""
    doc = ezdxf.new(setup=True)
    doc.layers.new(name="EQUIPMENT", dxfattribs={"color": 7})
    msp = doc.modelspace()
    draw_equipment_symbol(msp, _eq(etype="unknown_type_xyz"), "EQUIPMENT")
    if not (len(list(msp)) > 0): raise ValueError(f"Assertion failed: { len(list(msp)) > 0 }")
