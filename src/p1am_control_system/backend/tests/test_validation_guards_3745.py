"""Regression tests for the p1am validation/guard findings in issue #3745.

Covers:
- PIDConfig.pv_tag/cv_tag validation (reject malformed/out-of-range tags,
  accept valid tags and the kUnmappedTag sentinel).
- start_pid_tuning double-start returns 409 instead of silently overwriting.
- ConnectionManager.register_accepted encapsulates connection bookkeeping.
"""

import copy
import os
from collections.abc import Generator, Iterable
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

os.environ["PLC_DRIVER"] = "modbus"

import hardware
from models import PIDConfig

# ---------------------------------------------------------------------------
# Finding [B] models.py: PIDConfig pv_tag/cv_tag validation
# ---------------------------------------------------------------------------


def _valid_pid(**overrides: object) -> PIDConfig:
    base = {
        "pv_tag": "TAG_1",
        "cv_tag": "TAG_2",
        "setpoint": 50.0,
        "kp": 1.0,
        "ki": 0.5,
        "kd": 0.1,
    }
    base.update(overrides)
    return PIDConfig(**base)


def test_pidconfig_accepts_valid_tags() -> None:
    pid = _valid_pid(pv_tag="TAG_0", cv_tag="TAG_31")
    assert pid.pv_tag == "TAG_0"
    assert pid.cv_tag == "TAG_31"


def test_pidconfig_accepts_unmapped_sentinel() -> None:
    """TAG_255 (kUnmappedTag) is a legitimate value after an NVRAM default boot."""
    sentinel = hardware.UNMAPPED_TAG_NAME
    pid = _valid_pid(pv_tag=sentinel, cv_tag=sentinel)
    assert pid.pv_tag == "TAG_255"
    assert pid.cv_tag == "TAG_255"


@pytest.mark.parametrize(
    "bad_tag",
    [
        "",
        "TAG_",
        "TAG_x",
        "FOO_1",
        "TAG_32",  # one past the broker range
        "TAG_254",  # out of range and not the sentinel
        "TAG_999",
    ],
)
def test_pidconfig_rejects_malformed_or_out_of_range_pv_tag(bad_tag: str) -> None:
    with pytest.raises(ValueError):
        _valid_pid(pv_tag=bad_tag)


@pytest.mark.parametrize("bad_tag", ["", "nope", "TAG_40"])
def test_pidconfig_rejects_malformed_cv_tag(bad_tag: str) -> None:
    with pytest.raises(ValueError):
        _valid_pid(cv_tag=bad_tag)


# ---------------------------------------------------------------------------
# Finding [C] main.py: ConnectionManager.register_accepted
# ---------------------------------------------------------------------------


def test_connection_manager_register_accepted_tracks_socket() -> None:
    from main import ConnectionManager

    manager = ConnectionManager()
    socket = MagicMock()
    assert socket not in manager.active_connections

    manager.register_accepted(socket)

    assert socket in manager.active_connections
    # register_accepted must NOT call accept() — the frame-auth path already did.
    socket.accept.assert_not_called()


# ---------------------------------------------------------------------------
# Finding [I] main.py: get_ladder_explorer N+1 parent lookups
# ---------------------------------------------------------------------------


class _Rows:
    def __init__(self, rows: Iterable[object]) -> None:
        self._rows = list(rows)

    def all(self) -> list[object]:
        return self._rows


class _LadderExplorerSession:
    def __init__(self) -> None:
        self.exec_count = 0
        self.get_count = 0
        self.area = SimpleNamespace(id=1, name="Area A")
        self.unit = SimpleNamespace(id=2, name="Unit 1", area_id=1)
        self.equip = SimpleNamespace(id=3, name="Pump 1", unit_id=2)
        self.tag = SimpleNamespace(
            name="TAG_1",
            tag_type="Real",
            description="Process value",
            rw_mode="Read-only",
            register_type="holding",
            register_num=1,
            data_format="float",
            scale_factor=1.0,
            equipment_id=3,
        )

    def exec(self, statement: object) -> _Rows:
        self.exec_count += 1
        text = str(statement).lower()
        if "plantarea" in text:
            return _Rows([self.area])
        if "plantunit" in text:
            return _Rows([self.unit])
        if "plantequipment" in text:
            return _Rows([self.equip])
        if "tagdefinitiondb" in text:
            return _Rows([self.tag])
        raise AssertionError(f"unexpected statement: {statement!r}")

    def get(self, *_args: object, **_kwargs: object) -> object:
        self.get_count += 1
        raise AssertionError("get_ladder_explorer should preload parent lookups")


def test_ladder_explorer_preloads_parent_lookup_tables() -> None:
    # get_ladder_explorer is a sync (threadpool-offloaded) route handler, so it
    # is called directly rather than awaited.
    from main import get_ladder_explorer

    session = _LadderExplorerSession()

    result = get_ladder_explorer(db=session)

    assert session.exec_count == 4
    assert session.get_count == 0
    assert result == [
        {
            "name": "TAG_1",
            "tag_type": "Real",
            "description": "Process value",
            "rw_mode": "Read-only",
            "register_type": "holding",
            "register_num": 1,
            "data_format": "float",
            "scale_factor": 1.0,
            "equipment": "Pump 1",
            "unit": "Unit 1",
            "area": "Area A",
        }
    ]


# ---------------------------------------------------------------------------
# Finding [B] main.py: start_pid_tuning double-start returns 409
# ---------------------------------------------------------------------------

pytest.importorskip("httpx")
pytest.importorskip("fastapi.testclient")

from fastapi.testclient import TestClient  # noqa: E402
from main import (  # noqa: E402
    app,
    backup_simulator,
    control_context,
    modbus_manager,
)

# The HMI marker header: cors_config.RequestGuardMiddleware refuses a
# state-changing request that carries no preflight-forcing signal, because a
# bodyless control POST is otherwise a CORS-"simple" request any page can make
# (#4037). Set it once on the client so every request below is HMI-shaped.


@pytest.fixture(autouse=True)
def _bench_no_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-establish the bench auth bypass for EVERY test in this module.

    This used to be a bare ``os.environ`` assignment at import time, which is
    order-dependent: a sibling suite that clears the variable at *its* import
    time silently disables the bypass for this whole module, and the tests then
    fail with 503 ("no credential configured") depending only on collection
    order and xdist worker assignment (#4061). A per-test ``monkeypatch`` is
    immune to that and unwinds cleanly afterwards.
    """
    monkeypatch.setenv("P1AM_DEV_NO_AUTH", "1")


client = TestClient(app, headers={"X-Requested-With": "p1am-hmi"})


@pytest.fixture(autouse=True)
def restore_control_context() -> Generator[None, None, None]:
    original_config = control_context.active_config
    original_sessions = dict(control_context.tuning_sessions)
    try:
        yield
    finally:
        control_context.apply_config(original_config, modbus_manager, backup_simulator)
        control_context.tuning_sessions.clear()
        control_context.tuning_sessions.update(original_sessions)


def test_start_pid_tuning_succeeds_when_no_session_active() -> None:
    """A clean start (no pre-existing session) is unaffected by the new guard."""
    config = copy.deepcopy(control_context.active_config)
    config.pids[0] = PIDConfig(
        pv_tag="TAG_1", cv_tag="TAG_2", setpoint=50.0, kp=1.0, ki=0.5, kd=0.1
    )
    control_context.apply_config(config, modbus_manager, backup_simulator)
    control_context.tuning_sessions.pop(0, None)
    control_context.latest_tags["TAG_1"] = 10.0
    control_context.latest_tags["TAG_2"] = 20.0

    response = client.post("/api/pid/0/tuning/start")

    assert response.status_code == 200
    assert 0 in control_context.tuning_sessions


def test_start_pid_tuning_double_start_returns_409() -> None:
    """A second start while a session is active must 409, not overwrite it."""
    config = copy.deepcopy(control_context.active_config)
    config.pids[0] = PIDConfig(
        pv_tag="TAG_1", cv_tag="TAG_2", setpoint=50.0, kp=1.0, ki=0.5, kd=0.1
    )
    control_context.apply_config(config, modbus_manager, backup_simulator)

    sentinel_session = {"start_time": 123.0, "history": ["preexisting"]}
    control_context.tuning_sessions[0] = sentinel_session

    response = client.post("/api/pid/0/tuning/start")

    assert response.status_code == 409
    assert "already active" in response.json()["detail"]
    # The in-progress session must be left untouched.
    assert control_context.tuning_sessions[0] is sentinel_session
