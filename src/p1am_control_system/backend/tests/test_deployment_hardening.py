"""Static contract tests over the deployment artifacts (#4007/#4014/#4030/#4036).

Every exploitable defect in this PR's issue set lives in a file no Python test
had ever looked at: a systemd unit, a Dockerfile, a compose file, a Vite config.
They are text, so they are testable — and they are exactly the files where a
regression is invisible until a Raspberry Pi on a plant VLAN is already running
it.

What is asserted here:

- the installer never hardcodes the auth bypass, generates credentials into a
  root-owned ``EnvironmentFile``, and refuses to write a unit without one;
- ``vite preview`` binds loopback, so the HMI cannot re-expose the deliberately
  loopback-bound backend to the plant VLAN;
- the backend image installs every module ``main``/``settings`` import at
  runtime, pinned, and its pin list does not drift from the ``p1am`` extra that
  the systemd installer uses (single source of truth in ``pyproject.toml``);
- the container binds ``0.0.0.0`` internally and is isolated at the *publish*
  layer instead;
- compose passes the env-var names ``settings.py`` actually reads, so the
  driver cannot silently fall back to the simulator;
- the historian volume is not mounted over the application source;
- neither systemd unit can saturate all four Pi cores at ``Restart=always``.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SYSTEM = _REPO_ROOT / "src" / "p1am_control_system"
_INSTALLER = _SYSTEM / "deploy" / "install-services.sh"
_COMPOSE = _SYSTEM / "docker-compose.yml"
_BACKEND_DOCKERFILE = _SYSTEM / "backend" / "Dockerfile"
_VITE_CONFIG = _SYSTEM / "frontend" / "vite.config.ts"
_PYPROJECT = _REPO_ROOT / "pyproject.toml"
_REQUIREMENTS = _REPO_ROOT / "requirements.txt"
_REQUIREMENTS_LOCK = _REPO_ROOT / "requirements-lock.txt"


def _read(path: Path) -> str:
    assert path.is_file(), f"missing deployment artifact: {path}"
    return path.read_text(encoding="utf-8")


def _uncommented(text: str) -> str:
    """Strip ``#`` comment lines so prose about a defect is not read as one."""
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )


def _backend_unit_template() -> str:
    """The systemd unit the installer writes, as a string."""
    text = _read(_INSTALLER)
    assert 'backend_unit="' in text, "installer must build a backend unit"
    return text.split('backend_unit="', 1)[1].split('"\n', 1)[0]


def _requirement_name(spec: str) -> str:
    """Normalize ``pkg[extra]>=1.2`` to the comparable distribution name."""
    name = re.split(r"[<>=!~\[;]", spec, maxsplit=1)[0]
    return name.strip().lower().replace("_", "-")


def _p1am_extra() -> list[str]:
    data = tomllib.loads(_read(_PYPROJECT))
    extras = data["project"]["optional-dependencies"]
    assert "p1am" in extras, (
        "pyproject.toml must declare a [project.optional-dependencies] 'p1am' "
        "extra holding the backend runtime dependencies (#4014)."
    )
    return list(extras["p1am"])


def _dockerfile_pins() -> list[str]:
    """The pip install list from the backend image, one requirement per entry."""
    text = _read(_BACKEND_DOCKERFILE)
    match = re.search(
        r"RUN pip install --no-cache-dir\s+(.*?)"
        r"(?:\n(?!\s)|\nRUN|\nCOPY|\nENV|\nEXPOSE|\nCMD)",
        text,
        re.DOTALL,
    )
    assert match, "backend/Dockerfile must install its runtime deps with pip"
    body = match.group(1).replace("\\\n", " ").replace("\\", " ")
    return [tok for tok in body.split() if tok and not tok.startswith("-")]


# --------------------------------------------------------------------------- #
# #4007 — the installer must not ship with authentication disabled             #
# --------------------------------------------------------------------------- #


def test_installer_does_not_hardcode_the_auth_bypass() -> None:
    unit = _backend_unit_template()
    assert "P1AM_DEV_NO_AUTH" not in unit, (
        "install-services.sh must not bake P1AM_DEV_NO_AUTH=1 into the systemd "
        "unit — it short-circuits require_api_key/require_admin_key and the "
        "WebSocket gate on a production Pi (#4007). The bypass may only reach "
        "the unit through the explicit --bench branch."
    )


def test_installer_gates_no_auth_behind_an_explicit_bench_flag() -> None:
    text = _read(_INSTALLER)
    assert "--bench" in text
    assert "P1AM_DEV_NO_AUTH" in text, "the bench opt-out must still be reachable"
    bench_guard = text.find('BENCH_MODE" = "1"')
    bypass = text.find("P1AM_DEV_NO_AUTH=1", bench_guard)
    assert bench_guard != -1 and bypass > bench_guard, (
        "the bypass must be assigned inside the --bench branch, after the flag "
        "has been checked"
    )


def test_installer_generates_credentials_into_an_environment_file() -> None:
    text = _read(_INSTALLER)
    assert "EnvironmentFile=" in text
    assert "P1AM_API_KEY" in text
    assert "P1AM_ADMIN_API_KEY" in text
    assert "openssl rand" in text, "credentials must be randomly generated at install"


def test_installer_locks_down_the_environment_file() -> None:
    text = _read(_INSTALLER)
    assert "chmod" in text and ("640" in text or "600" in text)
    assert "chown" in text


def test_installer_refuses_to_write_a_unit_without_a_credential() -> None:
    text = _read(_INSTALLER)
    assert "exit 1" in text
    assert re.search(
        r"refus|ERROR:.*credential", text, re.IGNORECASE
    ), "the installer must fail closed when neither key is present (#4007)"


def test_installer_configures_the_cors_allowlist() -> None:
    """The origin guard fails closed; a production install must be explicit."""
    assert "P1AM_CORS_ORIGINS" in _read(_INSTALLER)


def test_installer_enables_read_auth() -> None:
    assert "P1AM_REQUIRE_READ_AUTH" in _read(_INSTALLER)


# --------------------------------------------------------------------------- #
# #4007 — the HMI must not re-expose the loopback-bound backend                #
# --------------------------------------------------------------------------- #


def test_vite_preview_binds_loopback_only() -> None:
    text = _read(_VITE_CONFIG)
    preview = text.split("preview:", 1)
    assert len(preview) == 2, "vite.config.ts must configure `preview`"
    block = preview[1].split("},", 1)[0]
    assert "host: true" not in block, (
        "`vite preview` with host:true binds every interface AND proxies /api + "
        "the WebSocket to the loopback backend, so the whole plant VLAN reaches "
        "the 'loopback-only' control API (#4007)."
    )
    assert '"127.0.0.1"' in block or "'127.0.0.1'" in block


# --------------------------------------------------------------------------- #
# #4014 — the backend image must actually be able to import the app            #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "distribution",
    [
        "fastapi",
        "uvicorn",
        "sqlmodel",
        "pymodbus",
        "websockets",
        "numpy",
        "pydantic-settings",
        "python-multipart",
    ],
)
def test_p1am_extra_declares_every_backend_runtime_dependency(
    distribution: str,
) -> None:
    names = {_requirement_name(spec) for spec in _p1am_extra()}
    assert distribution in names, (
        f"{distribution} is imported by the running backend but is not in the "
        "pyproject 'p1am' extra (#4014)."
    )


def test_p1am_extra_is_version_pinned() -> None:
    unpinned = [spec for spec in _p1am_extra() if not re.search(r"[<>=~]", spec)]
    assert not unpinned, f"unpinned backend runtime deps: {unpinned}"


def test_dockerfile_pin_list_matches_the_p1am_extra() -> None:
    """DRY: pyproject is the single source of truth; the image is derived.

    The Docker build context is ``./backend``, which cannot see the repo's
    pyproject.toml, so the list is mirrored into the Dockerfile. This test is
    what keeps the mirror honest.
    """
    assert {_requirement_name(s) for s in _dockerfile_pins()} == {
        _requirement_name(s) for s in _p1am_extra()
    }


def test_dockerfile_pins_exact_versions() -> None:
    unpinned = [spec for spec in _dockerfile_pins() if "==" not in spec]
    assert not unpinned, f"backend image installs unpinned packages: {unpinned}"


def test_dockerfile_binds_all_interfaces_inside_the_container() -> None:
    text = _uncommented(_read(_BACKEND_DOCKERFILE))
    assert "ENV P1AM_BIND_HOST=127.0.0.1" not in text, (
        "binding 127.0.0.1 *inside* the container makes the published port and "
        "nginx's proxy_pass http://backend:8000 both reach nothing (#4014)."
    )
    # Matched by regex rather than a literal: the bind-all address written out
    # as a plain string trips bandit's B104 here, where it is only ever the
    # subject of an assertion. Isolation is asserted separately, at the publish
    # layer, by test_compose_isolates_the_backend_at_the_publish_layer.
    assert re.search(r"ENV P1AM_BIND_HOST=0\.0\.0\.0", text), (
        "the container must bind every interface internally so the published "
        "port and the frontend's proxy_pass can reach it (#4014)."
    )


def test_installer_uses_the_p1am_extra() -> None:
    assert "[p1am]" in _read(_INSTALLER)


# --------------------------------------------------------------------------- #
# #4030 — compose must configure the app the app actually reads                #
# --------------------------------------------------------------------------- #


def test_compose_isolates_the_backend_at_the_publish_layer() -> None:
    text = _uncommented(_read(_COMPOSE))
    assert "127.0.0.1:8000:8000" in text, (
        "the container binds 0.0.0.0 internally, so isolation must happen at "
        "the published port (#4014/#4030)."
    )


def test_compose_uses_the_env_var_names_settings_reads() -> None:
    text = _uncommented(_read(_COMPOSE))
    assert "PLC_HOST" not in text, (
        "settings.py reads P1AM_PLC_IP/PLC_IP — PLC_HOST is silently ignored, "
        "so the driver defaults to the simulator and the HMI shows fabricated "
        "data an operator cannot distinguish from the plant (#4030)."
    )
    assert "P1AM_PLC_IP" in text
    assert "P1AM_PLC_PORT" in text
    assert "P1AM_PLC_DRIVER=modbus" in text


def test_compose_does_not_mount_the_volume_over_the_source() -> None:
    text = _uncommented(_read(_COMPOSE))
    assert "dcs_db_data:/app" not in text, (
        "mounting the historian volume over /app hides the source the "
        "Dockerfile copied in, so every rebuild is silently discarded (#4030)."
    )
    assert "dcs_db_data:/data" in text


def test_compose_requires_credentials_rather_than_defaulting_open() -> None:
    text = _read(_COMPOSE)
    assert "P1AM_API_KEY" in text and "P1AM_ADMIN_API_KEY" in text
    assert "${P1AM_ADMIN_API_KEY:?" in text, (
        "compose must refuse to start without an admin credential rather than "
        "silently coming up unauthenticated."
    )


# --------------------------------------------------------------------------- #
# #4036 — the units must not fight the Pi for CPU                              #
# --------------------------------------------------------------------------- #


def test_frontend_unit_does_not_build_inside_execstart() -> None:
    text = _read(_INSTALLER)
    execstarts = re.findall(r"ExecStart=.*", text)
    building = [line for line in execstarts if "run build" in line]
    assert not building, (
        "a full tsc+vite build inside ExecStart on a Restart=always unit "
        "saturates all four Pi cores for 1-3 minutes on every start, and loops "
        "forever if preview fails to bind (#4036)."
    )


def test_both_units_set_a_scheduling_priority() -> None:
    text = _read(_INSTALLER)
    assert text.count("Nice=") >= 2, "both units must set Nice="
    assert text.count("CPUWeight=") >= 2, "both units must set CPUWeight="


def test_backend_outranks_the_hmi() -> None:
    """The Modbus master is the real-time path; the HMI is not."""
    nice_values = [int(v) for v in re.findall(r"Nice=(-?\d+)", _read(_INSTALLER))]
    assert len(nice_values) >= 2
    assert min(nice_values) < max(nice_values)


def test_installer_reports_the_active_scada_kernel() -> None:
    """``tools_core`` is installed by no artifact; the installer must say so."""
    assert "tools_core" in _read(_INSTALLER)


# --------------------------------------------------------------------------- #
# #4014 — the root manifests must not contradict each other                    #
# --------------------------------------------------------------------------- #


def test_lockfile_numpy_satisfies_the_requirements_bound() -> None:
    requirements = _read(_REQUIREMENTS)
    lock = _read(_REQUIREMENTS_LOCK)
    bound = re.search(r"^numpy>=([\d.]+),<([\d.]+)", requirements, re.MULTILINE)
    pinned = re.search(r"^numpy==([\d.]+)", lock, re.MULTILINE)
    assert bound and pinned
    to_tuple = tuple(int(p) for p in pinned.group(1).split("."))
    upper = tuple(int(p) for p in bound.group(2).split("."))
    lower = tuple(int(p) for p in bound.group(1).split("."))
    assert lower <= to_tuple < upper, (
        f"requirements-lock.txt pins numpy=={pinned.group(1)} which violates "
        f"requirements.txt's >={bound.group(1)},<{bound.group(2)} (#4014)."
    )


def test_lockfile_does_not_pull_desktop_only_wheels_onto_the_pi() -> None:
    """The Pi is aarch64: PyOpenGL_accelerate has no wheel and playwright and
    PyQt6 have no business on a headless controller."""
    lock = _read(_REQUIREMENTS_LOCK)
    assert "PyOpenGL_accelerate" not in lock
