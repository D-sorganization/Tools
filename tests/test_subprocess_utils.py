"""Tests for subprocess_utils - Subprocess execution utilities.

These tests verify the subprocess utility functions using
Design by Contract principles.
"""

import subprocess
import sys
from unittest.mock import patch

import pytest


class TestRunCommandContract:
    """Design by Contract tests for run_command function."""

    def test_returns_completed_process(self):
        """Postcondition: Returns CompletedProcess object."""
        from utils.subprocess_utils import run_command

        result = run_command([sys.executable, "--version"])
        assert isinstance(result, subprocess.CompletedProcess)

    def test_raises_on_missing_cwd(self, tmp_path):
        """Precondition: Raises FileNotFoundError for missing cwd."""
        from utils.subprocess_utils import run_command

        with pytest.raises(FileNotFoundError):
            run_command(["echo", "test"], cwd=tmp_path / "nonexistent")


class TestRunCommand:
    """Functional tests for run_command."""

    def test_runs_simple_command(self):
        """Test running simple command."""
        from utils.subprocess_utils import run_command

        result = run_command([sys.executable, "--version"])
        assert result.returncode == 0
        assert "Python" in result.stdout or "python" in result.stdout.lower()

    def test_captures_stdout(self):
        """Test capturing stdout."""
        from utils.subprocess_utils import run_command

        result = run_command([sys.executable, "-c", "print('hello')"])
        assert "hello" in result.stdout

    def test_captures_stderr(self):
        """Test capturing stderr."""
        from utils.subprocess_utils import run_command

        result = run_command(
            [sys.executable, "-c", "import sys; sys.stderr.write('error\\n')"]
        )
        assert "error" in result.stderr

    def test_respects_cwd(self, tmp_path):
        """Test respecting working directory."""
        from utils.subprocess_utils import run_command

        result = run_command(
            [sys.executable, "-c", "import os; print(os.getcwd())"], cwd=tmp_path
        )
        assert str(tmp_path) in result.stdout or tmp_path.name in result.stdout

    def test_raises_on_timeout(self):
        """Test raising on timeout."""
        from utils.subprocess_utils import run_command

        with pytest.raises(subprocess.TimeoutExpired):
            run_command(
                [sys.executable, "-c", "import time; time.sleep(10)"],
                timeout=1,
            )

    def test_check_raises_on_failure(self):
        """Test check=True raises on failure."""
        from utils.subprocess_utils import run_command

        with pytest.raises(subprocess.CalledProcessError):
            run_command([sys.executable, "-c", "exit(1)"], check=True)

    def test_check_false_does_not_raise(self):
        """Test check=False does not raise on failure."""
        from utils.subprocess_utils import run_command

        result = run_command([sys.executable, "-c", "exit(1)"], check=False)
        assert result.returncode == 1


class TestRunPythonScriptContract:
    """Design by Contract tests for run_python_script function."""

    def test_raises_on_missing_script(self, tmp_path):
        """Precondition: Raises FileNotFoundError for missing script."""
        from utils.subprocess_utils import run_python_script

        with pytest.raises(FileNotFoundError):
            run_python_script(tmp_path / "nonexistent.py")


class TestRunPythonScript:
    """Functional tests for run_python_script."""

    def test_runs_script(self, tmp_path):
        """Test running Python script."""
        from utils.subprocess_utils import run_python_script

        script = tmp_path / "test_script.py"
        script.write_text("print('script output')")

        result = run_python_script(script)
        assert result.returncode == 0
        assert "script output" in result.stdout

    def test_passes_arguments(self, tmp_path):
        """Test passing arguments to script."""
        from utils.subprocess_utils import run_python_script

        script = tmp_path / "args_script.py"
        script.write_text("import sys; print(sys.argv[1:])")

        result = run_python_script(script, args=["arg1", "arg2"])
        assert "arg1" in result.stdout
        assert "arg2" in result.stdout

    def test_respects_cwd(self, tmp_path):
        """Test respecting working directory."""
        from utils.subprocess_utils import run_python_script

        script = tmp_path / "cwd_script.py"
        script.write_text("import os; print(os.getcwd())")

        work_dir = tmp_path / "workdir"
        work_dir.mkdir()

        result = run_python_script(script, cwd=work_dir)
        assert "workdir" in result.stdout

    def test_respects_timeout(self, tmp_path):
        """Test respecting timeout."""
        from utils.subprocess_utils import run_python_script

        script = tmp_path / "slow_script.py"
        script.write_text("import time; time.sleep(10)")

        with pytest.raises(subprocess.TimeoutExpired):
            run_python_script(script, timeout=1)


class TestRunPipCommandContract:
    """Design by Contract tests for run_pip_command function."""

    def test_raises_without_packages_or_requirements(self):
        """Precondition: Raises ValueError for install without packages."""
        from utils.subprocess_utils import run_pip_command

        with pytest.raises(ValueError, match="packages or requirements"):
            run_pip_command("install")


class TestRunPipCommand:
    """Functional tests for run_pip_command."""

    def test_runs_pip_list(self):
        """Test running pip list."""
        from utils.subprocess_utils import run_pip_command

        # Mock to avoid actual pip operations
        with patch("utils.subprocess_utils.run_command") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="pip list output", stderr=""
            )
            result = run_pip_command("list", packages=["dummy"])
            assert mock_run.called

    def test_adds_upgrade_flag(self):
        """Test adding upgrade flag."""
        from utils.subprocess_utils import run_pip_command

        with patch("utils.subprocess_utils.run_command") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="", stderr=""
            )
            run_pip_command("install", packages=["pytest"], upgrade=True)

            call_args = mock_run.call_args[0][0]
            assert "--upgrade" in call_args

    def test_uses_requirements_file(self, tmp_path):
        """Test using requirements file."""
        from utils.subprocess_utils import run_pip_command

        req_file = tmp_path / "requirements.txt"
        req_file.write_text("pytest")

        with patch("utils.subprocess_utils.run_command") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="", stderr=""
            )
            run_pip_command("install", requirements_file=req_file)

            call_args = mock_run.call_args[0][0]
            assert "-r" in call_args
            assert str(req_file) in call_args


class TestCheckCommandAvailableContract:
    """Design by Contract tests for check_command_available function."""

    def test_returns_bool(self):
        """Postcondition: Returns a boolean."""
        from utils.subprocess_utils import check_command_available

        result = check_command_available("python")
        assert isinstance(result, bool)


class TestCheckCommandAvailable:
    """Functional tests for check_command_available."""

    def test_python_is_available(self):
        """Test that python command is available."""
        from utils.subprocess_utils import check_command_available

        # Python should always be available since we're running tests
        result = check_command_available("python")
        assert result is True

    def test_nonexistent_command_not_available(self):
        """Test that nonexistent command is not available."""
        from utils.subprocess_utils import check_command_available

        result = check_command_available("definitely_not_a_real_command_xyz123")
        assert result is False


class TestGetCommandOutputContract:
    """Design by Contract tests for get_command_output function."""

    def test_returns_string(self):
        """Postcondition: Returns a string."""
        from utils.subprocess_utils import get_command_output

        result = get_command_output([sys.executable, "--version"])
        assert isinstance(result, str)


class TestGetCommandOutput:
    """Functional tests for get_command_output."""

    def test_returns_stdout(self):
        """Test returning stdout."""
        from utils.subprocess_utils import get_command_output

        result = get_command_output([sys.executable, "-c", "print('output')"])
        assert result == "output"

    def test_strips_whitespace(self):
        """Test stripping whitespace."""
        from utils.subprocess_utils import get_command_output

        result = get_command_output([sys.executable, "-c", "print('  padded  ')"])
        assert result == "padded"

    def test_raises_on_failure(self):
        """Test raising on command failure."""
        from utils.subprocess_utils import get_command_output

        with pytest.raises(subprocess.CalledProcessError):
            get_command_output([sys.executable, "-c", "exit(1)"])

    def test_respects_timeout(self):
        """Test respecting timeout."""
        from utils.subprocess_utils import get_command_output

        with pytest.raises(subprocess.TimeoutExpired):
            get_command_output(
                [sys.executable, "-c", "import time; time.sleep(10)"],
                timeout=1,
            )
