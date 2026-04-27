"""Targeted coverage tests for signal_toolkit.core, io, filters, noise, and calculus.

These tests exercise edge-case branches not covered by the module-specific
test suites: 2-D signals, single-sample signals, CSV edge cases, no-header
imports, and multi-output filter paths.  They are purely behavioural — each
test documents a pre-condition and asserts a meaningful post-condition.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from contracts import PreconditionError
from signal_toolkit.core import Signal
from signal_toolkit.io import SignalImporter

# ─────────────────────── Signal core edge cases ──────────────────────────────


class TestSignal2D:
    """Signal created from a 2-D values array."""

    def test_2d_signal_shape(self) -> None:
        """2-D values: shape[0] must match time length."""
        t = np.linspace(0, 1, 10)
        v = np.random.default_rng(0).random((10, 3))
        sig = Signal(time=t, values=v)
        assert sig.values.shape == (10, 3)

    def test_2d_signal_n_samples(self) -> None:
        t = np.linspace(0, 1, 20)
        v = np.zeros((20, 2))
        sig = Signal(time=t, values=v)
        assert sig.n_samples == 20

    def test_2d_shape_mismatch_raises(self) -> None:
        """2-D values whose first dim does not match time must raise."""
        t = np.linspace(0, 1, 5)
        v = np.zeros((6, 2))
        with pytest.raises(PreconditionError):
            Signal(time=t, values=v)

    def test_invalid_ndim_raises(self) -> None:
        """3-D values must raise a contract error."""
        t = np.linspace(0, 1, 4)
        v = np.zeros((4, 2, 2))
        with pytest.raises(PreconditionError):
            Signal(time=t, values=v)


class TestSignalSingleSample:
    """Signal with one sample exercises the < 2 guard in fs/dt/duration."""

    def _make_single(self) -> Signal:
        return Signal(time=np.array([0.0]), values=np.array([1.0]))

    def test_fs_single_sample(self) -> None:
        sig = self._make_single()
        assert sig.fs == 1.0

    def test_dt_single_sample(self) -> None:
        sig = self._make_single()
        assert sig.dt == 1.0

    def test_duration_single_sample(self) -> None:
        sig = self._make_single()
        assert sig.duration == 0.0


class TestSignalCopy:
    """Signal.copy returns an independent deep copy."""

    def test_copy_is_independent(self) -> None:
        t = np.linspace(0, 1, 5)
        v = np.ones(5)
        sig = Signal(time=t, values=v, name="orig", units="m")
        c = sig.copy()
        c.values[0] = 99.0
        assert sig.values[0] != 99.0

    def test_copy_preserves_metadata(self) -> None:
        t = np.linspace(0, 1, 5)
        v = np.ones(5)
        sig = Signal(time=t, values=v, metadata={"k": "v"})
        c = sig.copy()
        assert c.metadata["k"] == "v"


# ──────────────────── SignalImporter I/O edge cases ──────────────────────────


class TestSignalImporterCSV:
    """CSV import edge cases: no header, empty file, column selection."""

    def _write_csv(self, content: str) -> Path:
        tmp = Path(tempfile.mktemp(suffix=".csv"))
        tmp.write_text(content)
        return tmp

    def test_empty_csv_raises(self) -> None:
        p = self._write_csv("")
        try:
            with pytest.raises(ValueError, match="Empty CSV"):
                SignalImporter.from_csv(p)
        finally:
            p.unlink(missing_ok=True)

    def test_no_header_csv(self) -> None:
        """CSV without header should auto-assign column indices as names."""
        p = self._write_csv("0.0,1.0\n0.1,2.0\n0.2,3.0\n")
        try:
            sig = SignalImporter.from_csv(p, skip_header=False)
            assert sig.n_samples == 3
        finally:
            p.unlink(missing_ok=True)

    def test_named_column_selection(self) -> None:
        """Select a single value column by name."""
        p = self._write_csv("time,a,b\n0.0,1.0,2.0\n0.1,3.0,4.0\n")
        try:
            sig = SignalImporter.from_csv(p, time_column="time", value_columns="a")
            assert sig.n_samples == 2
        finally:
            p.unlink(missing_ok=True)

    def test_list_column_selection(self) -> None:
        """Select multiple value columns returns a list of Signal objects."""
        p = self._write_csv("time,a,b\n0.0,1.0,2.0\n0.1,3.0,4.0\n")
        try:
            result = SignalImporter.from_csv(
                p, time_column="time", value_columns=["a", "b"]
            )
            assert isinstance(result, list)
            assert len(result) == 2
        finally:
            p.unlink(missing_ok=True)

    def test_unknown_column_raises(self) -> None:
        p = self._write_csv("time,a,b\n0.0,1.0,2.0\n")
        try:
            with pytest.raises(ValueError, match="not found"):
                SignalImporter.from_csv(p, time_column="time", value_columns="x")
        finally:
            p.unlink(missing_ok=True)

    def test_time_scale_applied(self) -> None:
        """time_scale=0.001 should convert ms to seconds."""
        p = self._write_csv("time,v\n0,1.0\n100,2.0\n200,3.0\n")
        try:
            sig = SignalImporter.from_csv(p, time_scale=0.001)
            assert sig.time[-1] == pytest.approx(0.2, rel=1e-6)
        finally:
            p.unlink(missing_ok=True)


class TestSignalImporterFromDict:
    """SignalImporter.from_dict covers the dictionary import path."""

    def test_basic_from_dict(self) -> None:
        data = {"time": [0.0, 0.1, 0.2], "values": [1.0, 2.0, 3.0]}
        sig = SignalImporter.from_dict(data)
        assert sig.n_samples == 3

    def test_from_dict_preserves_name(self) -> None:
        data = {"time": [0.0, 0.1], "values": [5.0, 6.0], "name": "test_sig"}
        sig = SignalImporter.from_dict(data)
        assert sig.name == "test_sig"

    def test_from_dict_preserves_units(self) -> None:
        data = {"time": [0.0, 0.1], "values": [5.0, 6.0], "units": "m/s"}
        sig = SignalImporter.from_dict(data)
        assert sig.units == "m/s"


class TestSignalImporterFromNumpy:
    """SignalImporter.from_numpy — thin wrapper around Signal constructor."""

    def test_roundtrip(self) -> None:
        t = np.linspace(0, 1, 10)
        v = np.sin(t)
        sig = SignalImporter.from_numpy(t, v, name="sine", units="rad")
        np.testing.assert_array_equal(sig.time, t)
        np.testing.assert_array_equal(sig.values, v)
        assert sig.name == "sine"
        assert sig.units == "rad"


class TestSignalImporterFromJson:
    """SignalImporter.from_json covers JSON import path."""

    def test_json_roundtrip(self, tmp_path: Path) -> None:
        payload = {
            "time": [0.0, 0.1, 0.2],
            "values": [1.0, 2.0, 3.0],
            "name": "json_sig",
            "units": "V",
        }
        p = tmp_path / "sig.json"
        p.write_text(json.dumps(payload))
        sig = SignalImporter.from_json(p)
        assert sig.n_samples == 3
        assert sig.name == "json_sig"
        assert sig.units == "V"

    def test_json_metadata_preserved(self, tmp_path: Path) -> None:
        payload = {
            "time": [0.0, 0.1],
            "values": [1.0, 2.0],
            "metadata": {"sensor": "A"},
        }
        p = tmp_path / "sig.json"
        p.write_text(json.dumps(payload))
        sig = SignalImporter.from_json(p)
        assert sig.metadata["sensor"] == "A"
        assert "source_file" in sig.metadata
