"""Tests for config_loader - Configuration loading utilities.

These tests verify the configuration loading functions using
Design by Contract principles.
"""

import json


class TestConfigLoaderContract:
    """Design by Contract tests for ConfigLoader class."""

    def test_load_returns_dict(self, tmp_path):
        """Postcondition: load() returns a dictionary."""
        from utils.config_loader import ConfigLoader

        config_file = tmp_path / "config.json"
        config_file.write_text('{"key": "value"}')

        loader = ConfigLoader(config_file)
        result = loader.load()
        assert isinstance(result, dict)

    def test_get_returns_value_or_default(self, tmp_path):
        """Postcondition: get() returns value or default."""
        from utils.config_loader import ConfigLoader

        config_file = tmp_path / "config.json"
        config_file.write_text('{"existing": "value"}')

        loader = ConfigLoader(config_file)
        loader.load()

        assert loader.get("existing") == "value"
        assert loader.get("missing", "default") == "default"

    def test_save_returns_bool(self, tmp_path):
        """Postcondition: save() returns boolean."""
        from utils.config_loader import ConfigLoader

        config_file = tmp_path / "config.json"
        loader = ConfigLoader(config_file, default_config={"key": "value"})
        loader.load()

        result = loader.save()
        assert isinstance(result, bool)


class TestConfigLoader:
    """Functional tests for ConfigLoader class."""

    def test_loads_existing_config(self, tmp_path):
        """Test loading existing configuration file."""
        from utils.config_loader import ConfigLoader

        config_data = {"database": {"host": "localhost", "port": 5432}}
        config_file = tmp_path / "app.json"
        config_file.write_text(json.dumps(config_data))

        loader = ConfigLoader(config_file)
        config = loader.load()

        assert config == config_data

    def test_uses_default_when_file_missing(self, tmp_path):
        """Test using default config when file is missing."""
        from utils.config_loader import ConfigLoader

        default = {"theme": "dark", "language": "en"}
        config_file = tmp_path / "nonexistent.json"

        loader = ConfigLoader(config_file, default_config=default)
        config = loader.load()

        assert config == default

    def test_caches_loaded_config(self, tmp_path):
        """Test that config is cached after loading."""
        from utils.config_loader import ConfigLoader

        config_file = tmp_path / "config.json"
        config_file.write_text('{"cached": true}')

        loader = ConfigLoader(config_file)
        first_load = loader.load()
        second_load = loader.load()

        assert first_load is second_load  # Same object (cached)

    def test_reload_forces_refresh(self, tmp_path):
        """Test that reload forces refresh from file."""
        from utils.config_loader import ConfigLoader

        config_file = tmp_path / "config.json"
        config_file.write_text('{"version": 1}')

        loader = ConfigLoader(config_file)
        loader.load()

        # Modify file
        config_file.write_text('{"version": 2}')

        # Normal load should return cached
        assert loader.load()["version"] == 1

        # Reload should get new version
        assert loader.reload()["version"] == 2

    def test_get_with_dot_notation(self, tmp_path):
        """Test getting nested values with dot notation."""
        from utils.config_loader import ConfigLoader

        config_data = {"database": {"connection": {"host": "db.example.com"}}}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        loader = ConfigLoader(config_file)
        loader.load()

        assert loader.get("database.connection.host") == "db.example.com"

    def test_get_returns_default_for_missing_nested(self, tmp_path):
        """Test get returns default for missing nested key."""
        from utils.config_loader import ConfigLoader

        config_file = tmp_path / "config.json"
        config_file.write_text('{"shallow": "value"}')

        loader = ConfigLoader(config_file)
        loader.load()

        result = loader.get("deep.nested.key", default="fallback")
        assert result == "fallback"

    def test_set_creates_nested_keys(self, tmp_path):
        """Test set creates nested keys."""
        from utils.config_loader import ConfigLoader

        config_file = tmp_path / "config.json"
        config_file.write_text("{}")

        loader = ConfigLoader(config_file)
        loader.load()
        loader.set("new.nested.key", "new_value")

        assert loader.get("new.nested.key") == "new_value"

    def test_save_writes_to_file(self, tmp_path):
        """Test save writes configuration to file."""
        from utils.config_loader import ConfigLoader

        config_file = tmp_path / "output.json"
        loader = ConfigLoader(config_file, default_config={"initial": True})
        loader.load()
        loader.set("added", "value")

        result = loader.save()
        assert result is True

        # Verify file content
        saved_data = json.loads(config_file.read_text())
        assert saved_data["initial"] is True
        assert saved_data["added"] == "value"

    def test_save_returns_false_when_not_loaded(self, tmp_path):
        """Test save returns False when config not loaded."""
        from utils.config_loader import ConfigLoader

        config_file = tmp_path / "config.json"
        loader = ConfigLoader(config_file)

        result = loader.save()
        assert result is False


class TestLoadConfigContract:
    """Design by Contract tests for load_config function."""

    def test_returns_dict(self, tmp_path):
        """Postcondition: Returns a dictionary."""
        from utils.config_loader import load_config

        config_file = tmp_path / "config.json"
        config_file.write_text('{"key": "value"}')

        result = load_config(config_file)
        assert isinstance(result, dict)


class TestLoadConfig:
    """Functional tests for load_config convenience function."""

    def test_loads_config_from_file(self, tmp_path):
        """Test loading config from file."""
        from utils.config_loader import load_config

        config_data = {"api_key": "secret", "debug": True}
        config_file = tmp_path / "settings.json"
        config_file.write_text(json.dumps(config_data))

        result = load_config(config_file)
        assert result == config_data

    def test_uses_default_when_missing(self, tmp_path):
        """Test using default when file missing."""
        from utils.config_loader import load_config

        default = {"fallback": True}
        result = load_config(tmp_path / "missing.json", default_config=default)
        assert result == default

    def test_accepts_string_path(self, tmp_path):
        """Test accepting string path."""
        from utils.config_loader import load_config

        config_file = tmp_path / "config.json"
        config_file.write_text('{"string_path": true}')

        result = load_config(str(config_file))
        assert result["string_path"] is True
