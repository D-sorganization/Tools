"""In-process runtime coverage for canonical standalone Sidekick modules."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_cli_parsing_and_dispatch_are_in_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLI defaults, validation, and dispatch execute parent-owned code."""
    cli = importlib.import_module("sidekick.__main__")
    input_path = tmp_path / "inputs.json"
    input_path.write_text("{}", encoding="utf-8")

    default_args = cli.parse_cli_args([])
    assert default_args.command == "gui"
    assert default_args.profile == "chat-first"

    run_args = cli.parse_cli_args(
        [
            "run",
            "--calculator",
            "demo",
            "--inputs",
            str(input_path),
            "--output",
            str(tmp_path / "result.json"),
            "--format",
            "csv",
        ]
    )
    assert run_args.inputs == input_path.resolve()
    assert run_args.format == "csv"

    with pytest.raises(SystemExit, match="2"):
        cli.build_parser().parse_args(["run", "--calculatr", "demo"])
    with pytest.raises(SystemExit, match="2"):
        cli.parse_cli_args(["run", "--calculator", "demo", "--inputs", "missing.json"])

    monkeypatch.setattr(
        cli,
        "parse_cli_args",
        lambda _argv: SimpleNamespace(handler=lambda _args: 7),
    )
    assert cli.main(["gui"]) == 7


def test_cli_deferred_handlers_normalize_exit_codes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deferred GUI and headless handlers return stable integer exit codes."""
    cli = importlib.import_module("sidekick.__main__")
    launcher_factory = importlib.import_module("sidekick.launcher_factory")
    runner = importlib.import_module("sidekick.standalone.runner")
    launch_calls: list[tuple[Any, Any]] = []
    run_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(
        launcher_factory,
        "create_launcher_config",
        lambda **kwargs: kwargs,
    )

    def _launch_app(config: Any, window_factory: Any) -> str:
        launch_calls.append((config, window_factory))
        return "5"

    monkeypatch.setattr(launcher_factory, "launch_app", _launch_app)
    gui_args = SimpleNamespace(
        data_dir=tmp_path,
        profile="chat-first",
        theme="Solarized",
        skip_onboarding=True,
    )
    assert cli.launch_gui(gui_args) == 5
    assert launch_calls[0][0]["data_dir"] == str(tmp_path)
    assert callable(launch_calls[0][1])

    def _run_calculator(**kwargs: Any) -> str:
        run_calls.append(kwargs)
        return "6"

    monkeypatch.setattr(runner, "run_calculator", _run_calculator)
    run_args = SimpleNamespace(
        calculator="demo",
        inputs=tmp_path / "inputs.json",
        output=None,
        format="json",
    )
    assert cli.run_headless(run_args) == 6
    assert run_calls == [
        {
            "calculator": "demo",
            "inputs_path": str(run_args.inputs),
            "output": "-",
            "format": "json",
        }
    ]


def test_profile_schema_and_migrations_enforce_contracts() -> None:
    """Canonical payloads round-trip while invalid and legacy forms are explicit."""
    persistence = importlib.import_module("sidekick.persistence")
    schema = importlib.import_module("sidekick.persistence.schema")
    state_profile = importlib.import_module("sidekick.persistence.state_profile")

    payload = schema.ProfilePayload(data={"profile": "chat-first"})
    assert payload.to_dict() == {
        "profile": "chat-first",
        schema.PROFILE_SCHEMA_VERSION_KEY: schema.PROFILE_SCHEMA_VERSION,
    }
    assert schema.ProfilePayload.from_dict(payload.to_dict()) == payload
    assert state_profile.current_schema_version() == schema.PROFILE_SCHEMA_VERSION
    assert state_profile.wrap_state(payload.to_dict()).data == {"profile": "chat-first"}
    assert state_profile.unwrap_payload(payload) == ({"profile": "chat-first"}, 1)

    with pytest.warns(state_profile.SchemaMigration):
        assert state_profile.unwrap_payload({"future_key": 3}) == ({"future_key": 3}, 1)

    state_profile.validate(payload.to_dict())
    state_profile.validate({"schema_version": 999, "future_key": True})
    assert persistence.ProfilePayload is schema.ProfilePayload

    with pytest.raises(TypeError, match="data must be a dict"):
        schema.ProfilePayload(data=[])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive int"):
        schema.ProfilePayload(schema_version=0)
    with pytest.raises(TypeError, match="raw profile payload"):
        schema.ProfilePayload.from_dict([])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="mapping"):
        state_profile.wrap_state([])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ProfilePayload or mapping"):
        state_profile.unwrap_payload([])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="required key"):
        state_profile.validate({})
    with pytest.raises(ValueError, match="non-negative int"):
        state_profile.validate({"schema_version": "one"})


def test_onboarding_and_preferences_cover_success_and_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """First-run state and typed preferences preserve their DbC boundaries."""
    standalone = importlib.import_module("sidekick.standalone")
    onboarding_module = importlib.import_module("sidekick.standalone.onboarding")
    preferences_module = importlib.import_module("sidekick.standalone.preferences")
    stores = importlib.import_module("sidekick.standalone.session_store")

    onboarding = onboarding_module.StandaloneOnboarding(tmp_path / "config")
    assert onboarding.current_state() is onboarding_module.OnboardingState.WELCOME
    assert onboarding.needs_onboarding()
    onboarding.collect_profile("calc-first")
    assert onboarding.chosen_profile() == "calc-first"
    for _ in onboarding_module.OnboardingState:
        onboarding.advance()
    assert onboarding.is_complete()
    assert not onboarding.needs_onboarding()
    assert not onboarding_module.StandaloneOnboarding(
        tmp_path, skip=True
    ).needs_onboarding()
    with pytest.raises(ValueError, match="Unknown profile"):
        onboarding.collect_profile("unknown")

    memory_store = stores.InMemorySessionStore()
    preferences = preferences_module.StandalonePreferences(memory_store)
    assert preferences.profile() == preferences_module.DEFAULT_PROFILE
    assert preferences.theme() == preferences_module.DEFAULT_THEME
    assert preferences.data_dir() == preferences_module.DEFAULT_DATA_DIR
    assert preferences.llm_provider() == preferences_module.DEFAULT_LLM_PROVIDER

    preferences.set_profile("calc-first")
    preferences.set_theme("Solarized")
    preferences.set_data_dir("relative/data")
    preferences.set_llm_provider("local")
    assert (
        preferences.profile(),
        preferences.theme(),
        preferences.data_dir(),
        preferences.llm_provider(),
    ) == ("calc-first", "Solarized", "relative/data", "local")
    assert standalone.__all__ == [
        "preferences",
        "onboarding",
        "runner",
        "session_store",
    ]

    with pytest.raises(ValueError, match="Invalid profile"):
        preferences.set_profile("unknown")
    with pytest.raises(ValueError, match="non-empty"):
        preferences.set_theme("")
    with pytest.raises(TypeError, match="data_dir"):
        preferences.set_data_dir(3)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        preferences.set_llm_provider(" ")
    with pytest.raises(AssertionError, match=r"get\(\) and set\(\)"):
        preferences_module.StandalonePreferences(object())

    monkeypatch.setattr(
        preferences_module.platformdirs,
        "user_config_dir",
        lambda _name: str(tmp_path / "default-config"),
    )
    default_preferences = preferences_module.StandalonePreferences()
    default_preferences.set_theme("Default store")
    assert default_preferences.theme() == "Default store"


def test_session_stores_round_trip_and_validate(tmp_path: Path) -> None:
    """Both persistence implementations round-trip values and profiles."""
    schema = importlib.import_module("sidekick.persistence.schema")
    stores = importlib.import_module("sidekick.standalone.session_store")

    memory = stores.InMemorySessionStore()
    assert isinstance(memory, stores.SessionStore)
    assert memory.get("missing", 4) == 4
    memory.set("key", {"nested": True})
    assert memory.get("key") == {"nested": True}
    with pytest.raises(TypeError, match="key must be a str"):
        memory.get(3)
    with pytest.raises(ValueError, match="non-empty"):
        memory.set("", 1)

    file_path = tmp_path / "settings" / "session.json"
    file_store = stores.FileSessionStore(file_path)
    file_store.set("theme", "Solarized")
    assert stores.FileSessionStore(file_path).get("theme") == "Solarized"
    file_path.write_text("{broken", encoding="utf-8")
    assert stores.FileSessionStore(file_path).get("fallback", 5) == 5
    with pytest.raises(TypeError, match="path must be"):
        stores.FileSessionStore(str(file_path))

    profile_store = stores.StandaloneSessionStore(tmp_path / "profiles-root")
    payload = schema.ProfilePayload(data={"profile": "chat-first"})
    assert profile_store.list_profiles() == []
    assert profile_store.last_profile() is None
    profile_store.save_profile("primary", payload)
    assert profile_store.load_profile("primary") == payload
    assert profile_store.list_profiles() == ["primary"]
    profile_store.set_last_profile("primary")
    assert profile_store.last_profile() == "primary"
    profile_store.delete_profile("primary")
    with pytest.raises(KeyError):
        profile_store.load_profile("primary")
    with pytest.raises(KeyError):
        profile_store.delete_profile("primary")
    with pytest.raises(ValueError, match="profile name"):
        profile_store.save_profile("../escape", payload)
    with pytest.raises(TypeError, match="payload"):
        profile_store.save_profile("valid", {})  # type: ignore[arg-type]


def test_runner_handles_registered_calculators_and_failures(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Headless dispatch covers JSON/CSV output and structured failures."""
    runner = importlib.import_module("sidekick.standalone.runner")
    monkeypatch.setattr(runner, "_REGISTRY", {})
    monkeypatch.setattr(runner, "_REGISTERED", True)
    inputs_path = tmp_path / "inputs.json"
    inputs_path.write_text('{"value": 3}', encoding="utf-8")

    @runner.register("double")
    def _double(inputs: dict[str, Any]) -> dict[str, Any]:
        return {"result": inputs["value"] * 2}

    assert runner.list_calculators() == ["double"]
    assert runner.run_calculator("double", str(inputs_path)) == 0
    assert json.loads(capsys.readouterr().out) == {"result": 6}

    csv_path = tmp_path / "results.csv"
    assert runner.run_calculator("double", str(inputs_path), str(csv_path), "csv") == 0
    assert "result,6," in csv_path.read_text(encoding="utf-8")

    assert runner.run_calculator("duble", str(inputs_path)) == 4
    assert "closest" in capsys.readouterr().err
    assert runner.run_calculator("double", str(tmp_path / "missing.json")) == 1
    assert "not found" in capsys.readouterr().err

    inputs_path.write_text("{bad", encoding="utf-8")
    assert runner.run_calculator("double", str(inputs_path)) == 1
    assert "parse" in capsys.readouterr().err

    inputs_path.write_text("{}", encoding="utf-8")

    @runner.register("invalid")
    def _invalid(_inputs: dict[str, Any]) -> dict[str, Any]:
        raise ValueError("bad calculation")

    assert runner.run_calculator("invalid", str(inputs_path)) == 3
    assert "bad calculation" in capsys.readouterr().err

    class _ContractCalculator:
        def validate_inputs(self, inputs: dict[str, Any]) -> SimpleNamespace:
            return SimpleNamespace(valid=bool(inputs.get("valid")), errors=["invalid"])

        def calculate(self, inputs: dict[str, Any]) -> SimpleNamespace:
            return SimpleNamespace(
                values={"result": inputs["value"]},
                units={"result": "unit"},
                warnings=["demonstration warning"],
            )

    runner._REGISTRY["contract"] = _ContractCalculator()
    inputs_path.write_text('{"valid": false, "value": 5}', encoding="utf-8")
    assert runner.run_calculator("contract", str(inputs_path)) == 3
    assert "invalid" in capsys.readouterr().err

    inputs_path.write_text('{"valid": true, "value": 5}', encoding="utf-8")
    assert runner.run_calculator("contract", str(inputs_path)) == 0
    contract_output = json.loads(capsys.readouterr().out)
    assert contract_output == {
        "values": {"result": 5},
        "units": {"result": "unit"},
        "warnings": ["demonstration warning"],
    }


def test_standalone_window_isolated_runtime(
    qapp: Any,
    monkeypatch: pytest.MonkeyPatch,
    profile_store: Any,
) -> None:
    """The PyQt6 shell remains usable when hosted panels are isolated."""
    from PyQt6.QtWidgets import QLabel, QMessageBox

    window_module = importlib.import_module("sidekick.standalone.window")
    monkeypatch.setattr(
        window_module.StandaloneSidekickWindow,
        "_create_chat_panel",
        lambda _self: QLabel("chat"),
    )
    monkeypatch.setattr(
        window_module.StandaloneSidekickWindow,
        "_create_sidebar_panel",
        lambda _self: QLabel("sidebar"),
    )
    monkeypatch.setattr(QMessageBox, "about", lambda *_args: None)

    store = profile_store
    config = window_module.StandaloneSidekickConfig(
        profile="chat-first",
        theme_name=None,
        session_store=store,
        host_action_port="host-port",
    )
    window = window_module.StandaloneSidekickWindow(config)
    assert window.windowTitle() == "Sidekick"
    assert window.active_profile() == "chat-first"
    assert window.active_theme() is None
    assert window.host_action_port() == "host-port"
    assert window.panel_for("chat-first").text() == "chat"
    assert window.sidebar().text() == "sidebar"
    assert sum(window.splitter_handle_positions()) > 0

    window.save_profile_to_store("primary")
    store.profiles["alternate"] = {
        "profile": "calc-first",
        "theme_name": None,
    }
    window.load_profile_from_store("alternate")
    assert window.active_profile() == "calc-first"
    assert window.panel_for("calc-first").text() == "sidebar"

    window._switch_profile("chat-first")
    assert window.active_profile() == "chat-first"
    window.show()
    qapp.processEvents()
    hidden = window.sidebar().isHidden()
    window._toggle_sidebar()
    assert window.sidebar().isHidden() is not hidden
    window._on_about()
    window._flush_session()
    assert store.last == "chat-first"

    with pytest.raises(ValueError, match="Invalid profile"):
        window_module.StandaloneSidekickConfig("invalid", None, store)
    with pytest.raises(TypeError, match="StandaloneSidekickConfig"):
        window_module.StandaloneSidekickWindow(object())
    with pytest.raises(ValueError, match="Unknown profile"):
        window.panel_for("invalid")
    with pytest.raises(ValueError, match="Unknown profile"):
        window._switch_profile("invalid")
    with pytest.raises(TypeError, match="must construct a QWidget"):
        window_module._require_widget(object(), "BrokenPanel")

    assert window_module._prompt_profile_name is not None
    window.close()
