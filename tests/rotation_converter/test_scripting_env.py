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


if __name__ == "__main__":
    unittest.main()
