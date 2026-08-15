"""Tests for the interactive scripting environment (MATLAB-like CLI backend)."""

import os
import tempfile
import unittest

from rotation_converter import modern_robotics as mr
from rotation_converter.converter import Rotation
from rotation_converter.rigid_transform import RigidTransform
from shared.python.scripting.scripting_env import ConsoleEnvironment


class TestConsoleEnvironment(unittest.TestCase):
    def setUp(self) -> None:
        initial_ns = {
            "mr": mr,
            "Rotation": Rotation,
            "RigidTransform": RigidTransform,
        }
        for name in dir(mr):
            if not name.startswith("_") and callable(getattr(mr, name)):
                initial_ns[name] = getattr(mr, name)

        self.env = ConsoleEnvironment(
            default_namespace=initial_ns,
            user_lib_path="~/.rotation_converter_test_funcs.py",
        )

    def test_initial_namespace(self) -> None:
        """Test that modern_robotics functions and useful classes are loaded."""
        ns = self.env.namespace

        # Verify modern robotics is loaded
        self.assertIn("mr", ns)
        self.assertIn("VecToso3", ns["mr"].__dict__)

        # modern robotics functions directly injected for Matlab-like feel:
        self.assertIn("VecToso3", ns)

        # Verify classes
        self.assertIn("Rotation", ns)
        self.assertIn("RigidTransform", ns)
        self.assertIn("np", ns)

    def test_execute_code(self) -> None:
        """Test executing simple Python code."""
        out, err = self.env.execute("a = 2 + 2\nprint(a)")
        self.assertIn("4", out)
        self.assertEqual(err, "")
        self.assertEqual(self.env.namespace["a"], 4)

    def test_execute_expression(self) -> None:
        """Test evaluating a single expression to mimic a REPL."""
        out, err = self.env.execute("2 + 2")
        self.assertIn("4", out)
        self.assertEqual(err, "")

    def test_execute_error(self) -> None:
        """Test executing erroneous code."""
        out, err = self.env.execute("1/0")
        self.assertIn("ZeroDivisionError: division by zero", err)

    def test_execute_none_source_raises_value_error(self) -> None:
        """None source is programmer error even when Python assertions are disabled."""
        with self.assertRaisesRegex(ValueError, "source must be provided"):
            self.env.execute(None)

    def test_execute_propagates_unexpected_custom_exception(self) -> None:
        """Unexpected custom exceptions should not be silently swallowed."""
        source = (
            "class UnexpectedConsoleFailure(Exception):\n"
            "    pass\n"
            "raise UnexpectedConsoleFailure('boom')"
        )

        with self.assertRaisesRegex(Exception, "boom"):
            self.env.execute(source)

    def test_user_functions(self) -> None:
        """Test saving and loading user-defined functions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            user_lib_path = os.path.join(tmpdir, "user_libs.py")
            self.env.set_user_library_path(user_lib_path)

            # Save a function
            self.env.save_user_code("def custom_func(x):\n    return x * 2\n")

            # Load it into namespace
            self.env.refresh_user_functions()

            # Call it
            out, err = self.env.execute("result = custom_func(21)\nprint(result)")
            self.assertEqual(self.env.namespace["result"], 42)

    def test_refresh_user_functions_keyboard_interrupt_propagates(self) -> None:
        """Test that KeyboardInterrupt during user function loading propagates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            user_lib_path = os.path.join(tmpdir, "user_libs.py")
            self.env.set_user_library_path(user_lib_path)

            # Save code that raises KeyboardInterrupt
            self.env.save_user_code("raise KeyboardInterrupt()")

            # Should propagate instead of being caught and logged
            with self.assertRaises(KeyboardInterrupt):
                self.env.refresh_user_functions()

    def test_refresh_user_functions_system_exit_propagates(self) -> None:
        """Test that SystemExit during user function loading propagates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            user_lib_path = os.path.join(tmpdir, "user_libs.py")
            self.env.set_user_library_path(user_lib_path)

            # Save code that raises SystemExit
            # `import sys` is blocked by the scripting sandbox, so importing it to
            # reach sys.exit never raises. SystemExit is a builtin, so raise it
            # directly to test what this case is actually about: propagation.
            self.env.save_user_code("raise SystemExit(1)")

            # Should propagate instead of being caught and logged
            with self.assertRaises(SystemExit):
                self.env.refresh_user_functions()

    def test_refresh_user_functions_handles_user_code_errors(self) -> None:
        """Test that user code errors are logged to stderr but not propagated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            user_lib_path = os.path.join(tmpdir, "user_libs.py")
            self.env.set_user_library_path(user_lib_path)

            # Save code that raises a user-level error
            self.env.save_user_code("raise ValueError('User code error')")

            # Should not raise, but should print traceback
            try:
                self.env.refresh_user_functions()
            except Exception:
                self.fail(
                    "refresh_user_functions should not propagate user-code errors"
                )


if __name__ == "__main__":
    unittest.main()
