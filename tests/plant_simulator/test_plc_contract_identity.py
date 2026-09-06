"""The PLC contract must have one module identity (issue #3984).

The p1am backend imports its own modules **flat** (``from plc_interface import
BasePLCClient``) because ``src/p1am_control_system/backend`` is the directory
that lands on ``sys.path``: in the container the Docker build context *is*
``backend/`` with ``PYTHONPATH=/app``, and under pytest that same directory is
listed in ``[tool.pytest.ini_options] pythonpath``. In neither environment is
the backend imported as ``p1am_control_system.backend``.

``plant_simulator.neural_simulator_client`` used to reach ``plc_interface`` and
``models`` by their *package* path instead. Those are different ``sys.modules``
keys, so each file executed a second time and produced a second, distinct copy
of every class in it. Two concrete consequences, not hypotheticals:

* ``PLCFactory.create_client`` is annotated as returning the flat-path
  ``BasePLCClient``, but ``NeuralSimulatorClient`` subclassed the package-path
  copy -- so ``isinstance(client, BasePLCClient)`` was ``False`` for the neural
  driver, and any ``isinstance`` gate or ``ABC.register`` call would have failed
  silently while reading as correct.
* ``models`` defines SQLModel tables. Executing it twice raises
  ``sqlalchemy.exc.InvalidRequestError: Table 'taglog' is already defined for
  this MetaData instance`` -- so on the pre-fix tree, importing both spellings
  in one interpreter is a hard failure, not a subtle one. That is why the tests
  below assert on the *source* and on ``issubclass`` rather than importing both
  paths: doing so would itself create the duplicate the fix exists to prevent.
"""

from __future__ import annotations

import pytest


def test_neural_client_does_not_name_the_package_path() -> None:
    """Pin the fix at its cause, not only at its effect.

    ``neural_simulator_client`` must not import ``p1am_control_system.*``: doing
    so re-creates the duplicate classes *and* re-creates the package-level
    import cycle between the two applications. Inspect AST / source directly
    so the contract holds even in lean CI environments without torch.
    """
    import ast
    from pathlib import Path

    target_file = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "plant_simulator"
        / "neural_simulator_client.py"
    )
    assert target_file.is_file()
    source = target_file.read_text(encoding="utf-8")

    tree = ast.parse(source, filename=str(target_file))
    offending: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if "p1am_control_system" in alias.name:
                    offending.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and "p1am_control_system" in node.module:
                offending.append(node.module)
    assert offending == [], (
        "neural_simulator_client must import the p1am contract flat, the way the "
        f"backend itself does (#3984); found: {offending}"
    )


def test_neural_client_subclasses_the_factory_contract() -> None:
    """The hazard #3984 names: the factory's declared contract must hold.

    ``PLCFactory.create_client`` declares ``-> BasePLCClient`` using the flat
    class, so the classes it can return must be subclasses of *that* object.
    This is the assertion that was ``False`` before the fix.
    """
    pytest.importorskip("torch")

    from plc_interface import BasePLCClient

    from plant_simulator.neural_simulator_client import NeuralSimulatorClient

    assert issubclass(NeuralSimulatorClient, BasePLCClient)


def test_every_backend_client_subclasses_the_flat_contract() -> None:
    """The same invariant for the two in-tree drivers, so it cannot regress."""
    pytest.importorskip("pymodbus")

    from modbus_client import AsyncModbusManager
    from plc_interface import BasePLCClient
    from simulator_client import SimulatedPLCClient

    assert issubclass(SimulatedPLCClient, BasePLCClient)
    assert issubclass(AsyncModbusManager, BasePLCClient)


def test_neural_driver_withdrawn_from_factory(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The uninstantiable 'neural' driver has been withdrawn from PLCFactory (#4950).

    Selecting plc_driver='neural' must not attempt to construct the unmaintained
    NeuralSimulatorClient; it falls back to SimulatedPLCClient with an explicit
    boot banner.
    """
    import logging

    from plc_factory import PLCFactory
    from settings import P1AMSettings
    from simulator_client import SimulatedPLCClient

    with caplog.at_level(logging.WARNING):
        client = PLCFactory.create_client(P1AMSettings(plc_driver="neural"))

    assert isinstance(client, SimulatedPLCClient)
    assert (
        "PLC_DRIVER='neural' is not a known driver; FELL BACK to the simulator"
        in caplog.text
    )
