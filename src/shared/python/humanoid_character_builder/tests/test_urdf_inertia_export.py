"""Physical matrix and native URDF export contracts for asymmetric inertias."""

import importlib.util
import json
import subprocess
import sys

import numpy as np
import pytest

from shared.python.humanoid_character_builder.mesh.inertia_calculator import (
    InertiaResult,
)


def test_urdf_coefficients_are_actual_symmetric_tensor_entries() -> None:
    inertia = InertiaResult(4.0, 5.0, 6.0, ixy=0.25, ixz=-0.5, iyz=0.75)
    coefficients = inertia.as_urdf_dict()
    reconstructed = np.array(
        [
            [coefficients["ixx"], coefficients["ixy"], coefficients["ixz"]],
            [coefficients["ixy"], coefficients["iyy"], coefficients["iyz"]],
            [coefficients["ixz"], coefficients["iyz"], coefficients["izz"]],
        ]
    )
    np.testing.assert_array_equal(reconstructed, inertia.as_matrix())


def test_actual_mujoco_urdf_readback_and_kinetic_energy() -> None:
    if importlib.util.find_spec("mujoco") is None:
        pytest.skip("actual native MuJoCo SDK is unavailable")
    # Analytic uniform triangle: mass2kg, vertices (0,0,0),(3,0,0),(0,6,0).
    inertia = InertiaResult(
        4.0, 1.0, 5.0, ixy=1.0, mass=2.0, center_of_mass=(1.0, 2.0, 0.0)
    )
    coefficients = " ".join(
        f'{name}="{value}"' for name, value in inertia.as_urdf_dict().items()
    )
    source = f"""<robot name="analytic_lamina">
      <link name="base"/><link name="lamina"><inertial>
        <origin xyz="1 2 0" rpy="0 0 0"/><mass value="2"/>
        <inertia {coefficients}/></inertial></link>
      <joint name="hinge" type="continuous"><parent link="base"/>
        <child link="lamina"/><axis xyz="1 2 3"/></joint></robot>"""
    # Isolate native plugin DLLs from pytest/Qt plugin load order on Windows.
    native = subprocess.run(
        [sys.executable, "-c", _NATIVE_READBACK],
        input=source,
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    readback = json.loads(native.stdout)
    native_tensor = np.asarray(readback["tensor"])
    np.testing.assert_allclose(native_tensor, inertia.as_matrix(), atol=1e-13)
    np.testing.assert_allclose(readback["center"], inertia.center_of_mass, atol=0.0)
    assert readback["mass"] == inertia.mass
    axis = np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0)
    center = np.asarray(inertia.center_of_mass)
    expected = 0.5 * (
        axis @ inertia.as_matrix() @ axis
        + inertia.mass * np.dot(np.cross(axis, center), np.cross(axis, center))
    )
    assert readback["kinetic_energy"] == pytest.approx(expected, rel=1e-13, abs=1e-13)


_NATIVE_READBACK = """
import mujoco
import json
import sys
import numpy as np
model = mujoco.MjModel.from_xml_string(sys.stdin.read())
body = model.body('lamina')
rotation = np.empty(9)
mujoco.mju_quat2Mat(rotation, body.iquat)
frame = rotation.reshape(3, 3)
tensor = frame @ np.diag(body.inertia) @ frame.T
data = mujoco.MjData(model)
data.qvel[:] = 1.
mujoco.mj_forward(model, data)
sys.stdout.write(json.dumps({'tensor': tensor.tolist(), 'center': body.ipos.tolist(),
                  'mass': float(body.mass[0]), 'kinetic_energy': .5*data.qM[0],
                  'runtime': mujoco.__version__}))
"""
