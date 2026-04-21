"""Tests for SignalLoader format detection and error handling.

Covers issue #664: resolve NotImplementedError-style stubs in
signal_toolkit.io by verifying:
  - FileNotFoundError for missing files
  - ValueError for unsupported extensions (with helpful message)
  - AssertionError for internal invariant violations
  - Correct docstring claims about supported formats

Note: We load ``signal_toolkit.io`` via ``importlib.util`` to avoid
pulling in ``signal_toolkit.__init__`` which imports scipy (the calculus
sub-module).  This side-steps a numpy/scipy version-compatibility issue
present in some CI environments.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Load signal_toolkit.core first (needed by io), then io itself, without
# triggering signal_toolkit/__init__.py (which imports scipy via calculus).
# ---------------------------------------------------------------------------
_SHARED = Path(__file__).resolve().parents[4] / "src" / "shared" / "python"

_core_spec = importlib.util.spec_from_file_location(
    "signal_toolkit.core",
    _SHARED / "signal_toolkit" / "core.py",
)
_core_mod = importlib.util.module_from_spec(_core_spec)
sys.modules["signal_toolkit.core"] = _core_mod
_core_spec.loader.exec_module(_core_mod)

_io_spec = importlib.util.spec_from_file_location(
    "signal_toolkit.io",
    _SHARED / "signal_toolkit" / "io.py",
)
_io_mod = importlib.util.module_from_spec(_io_spec)
sys.modules["signal_toolkit.io"] = _io_mod
_io_spec.loader.exec_module(_io_mod)

SignalLoader = _io_mod.SignalLoader


class TestSignalLoaderPreconditions:
    """Verify Design-by-Contract preconditions on SignalLoader.load."""

    def test_file_not_found_raises(self) -> None:
        """load() must raise FileNotFoundError for a non-existent path."""
        fake_path = Path("/tmp/_nonexistent_signal_file_.csv")
        assert not fake_path.exists()

        with pytest.raises(FileNotFoundError, match="does not exist"):
            SignalLoader.load(fake_path)

    def test_unsupported_extension_raises_valueerror(self) -> None:
        """load() must raise ValueError for an unrecognised extension."""
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as f:
            tmp = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                SignalLoader.load(tmp)
        finally:
            tmp.unlink(missing_ok=True)

    def test_unsupported_extension_lists_supported(self) -> None:
        """The error message should enumerate the supported extensions."""
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            tmp = Path(f.name)

        try:
            with pytest.raises(ValueError, match=r"\.csv") as exc_info:
                SignalLoader.load(tmp)
            # Should also mention other supported extensions
            assert ".json" in str(exc_info.value)
            assert ".npz" in str(exc_info.value)
        finally:
            tmp.unlink(missing_ok=True)


class TestSignalLoaderSupportedFormats:
    """Verify that SUPPORTED_EXTENSIONS covers the documented formats."""

    EXPECTED_EXTENSIONS = {".csv", ".txt", ".tsv", ".json", ".npz", ".npy", ".mat"}

    def test_all_documented_extensions_present(self) -> None:
        """Every extension mentioned in the docstring must be in the dict."""
        actual = set(SignalLoader.SUPPORTED_EXTENSIONS.keys())
        assert self.EXPECTED_EXTENSIONS == actual

    def test_csv_loads_correctly(self) -> None:
        """Smoke-test: .csv files load through SignalLoader.load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "data.csv"
            csv_path.write_text("time,value\n0.0,1.0\n1.0,2.0\n2.0,3.0\n")

            signal = SignalLoader.load(csv_path)

            # from_csv returns a single Signal when only one value column
            assert hasattr(signal, "time")
            assert len(signal.time) == 3

    def test_json_loads_correctly(self) -> None:
        """Smoke-test: .json files load through SignalLoader.load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "data.json"
            data = {"time": [0.0, 1.0, 2.0], "values": [1.0, 2.0, 3.0]}
            json_path.write_text(json.dumps(data))

            signal = SignalLoader.load(json_path)
            assert hasattr(signal, "time")
            assert len(signal.time) == 3

    def test_npy_1d_loads_correctly(self) -> None:
        """Smoke-test: 1-D .npy files assume uniform time sampling."""
        with tempfile.TemporaryDirectory() as tmpdir:
            npy_path = Path(tmpdir) / "data.npy"
            np.save(npy_path, np.array([10.0, 20.0, 30.0]))

            signal = SignalLoader.load(npy_path)
            assert np.array_equal(signal.values, [10.0, 20.0, 30.0])
            # Time should be 0, 1, 2 (uniform)
            assert np.array_equal(signal.time, [0, 1, 2])

    def test_npy_2d_loads_correctly(self) -> None:
        """Smoke-test: 2-D .npy files use column 0 as time."""
        with tempfile.TemporaryDirectory() as tmpdir:
            npy_path = Path(tmpdir) / "data.npy"
            arr = np.column_stack(
                [np.array([0.0, 1.0, 2.0]), np.array([5.0, 6.0, 7.0])]
            )
            np.save(npy_path, arr)

            signal = SignalLoader.load(npy_path)
            assert np.allclose(signal.time, [0.0, 1.0, 2.0])
            assert np.allclose(signal.values, [5.0, 6.0, 7.0])

    def test_npz_loads_correctly(self) -> None:
        """Smoke-test: .npz files load through SignalLoader.load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = Path(tmpdir) / "data.npz"
            np.savez(
                npz_path,
                time=np.array([0.0, 1.0, 2.0]),
                values=np.array([10.0, 20.0, 30.0]),
            )

            signal = SignalLoader.load(npz_path)
            assert np.allclose(signal.time, [0.0, 1.0, 2.0])
            assert np.allclose(signal.values, [10.0, 20.0, 30.0])

    def test_tsv_uses_tab_delimiter(self) -> None:
        """TSV files should auto-detect tab delimiter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tsv_path = Path(tmpdir) / "data.tsv"
            tsv_path.write_text("time\tvalue\n0.0\t1.0\n1.0\t2.0\n")

            signal = SignalLoader.load(tsv_path)
            assert hasattr(signal, "time")
            assert len(signal.time) == 2


class TestSignalLoaderInternalInvariant:
    """Test the AssertionError safeguard for future-proofing."""

    def test_unhandled_format_tag_raises_assertion(self) -> None:
        """If a new format tag is added without a handler, AssertionError fires."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a file with a supported-looking extension
            fake_path = Path(tmpdir) / "data.csv"
            fake_path.write_text("time,value\n0,1\n")

            # Temporarily inject a bogus format tag
            original = SignalLoader.SUPPORTED_EXTENSIONS.copy()
            try:
                SignalLoader.SUPPORTED_EXTENSIONS[".csv"] = "bogus_format"
                with pytest.raises(AssertionError, match="Internal error"):
                    SignalLoader.load(fake_path)
            finally:
                SignalLoader.SUPPORTED_EXTENSIONS.clear()
                SignalLoader.SUPPORTED_EXTENSIONS.update(original)
