from pathlib import Path

from pdf_renamer import config


class DummyKeyring:
    def __init__(self) -> None:
        self.saved: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> str | None:
        return self.saved.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self.saved[(service, username)] = password


def test_get_api_key_prefers_environment(monkeypatch) -> None:
    keyring = DummyKeyring()
    keyring.set_password(config.KEYRING_SERVICE, config.KEYRING_USERNAME, "stored")
    monkeypatch.setattr(config, "_get_keyring", lambda: keyring)
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")

    assert config.get_api_key() == "from-env"


def test_get_api_key_reads_keyring_without_env(monkeypatch) -> None:
    keyring = DummyKeyring()
    keyring.set_password(config.KEYRING_SERVICE, config.KEYRING_USERNAME, "stored")
    monkeypatch.setattr(config, "_get_keyring", lambda: keyring)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    assert config.get_api_key() == "stored"


def test_setup_api_key_interactive_saves_keyring_not_env_file(
    monkeypatch, tmp_path
) -> None:
    keyring = DummyKeyring()
    answers = iter(["y", "secret-key"])
    monkeypatch.setattr(config, "_get_keyring", lambda: keyring)
    monkeypatch.setattr(config, "get_api_key", lambda: None)
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)

    assert config.setup_api_key_interactive() is True
    assert keyring.get_password(config.KEYRING_SERVICE, config.KEYRING_USERNAME) == (
        "secret-key"
    )
    assert not (tmp_path / ".env").exists()
    assert not (tmp_path / ".pdf_renamer" / ".env").exists()


def test_setup_api_key_interactive_without_keyring_does_not_write_env(
    monkeypatch, tmp_path
) -> None:
    answers = iter(["y", "secret-key"])
    monkeypatch.setattr(config, "_get_keyring", lambda: None)
    monkeypatch.setattr(config, "get_api_key", lambda: None)
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)

    assert config.setup_api_key_interactive() is False
    assert not (tmp_path / ".env").exists()
    assert not (tmp_path / ".pdf_renamer" / ".env").exists()
