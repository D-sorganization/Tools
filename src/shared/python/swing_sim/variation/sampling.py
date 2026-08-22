"""Deterministic eager and bounded-chunk sampling for variation plans."""

from __future__ import annotations

import zlib
from collections.abc import Iterator

import numpy as np

from shared.python.contracts import require

from .group_spec import PerturbationGroup
from .spec import NoiseSpec, VariationPlan

SAMPLING_ALGORITHM_ID = "numpy-pcg64-canonical-rowwise-psd"
SAMPLING_ALGORITHM_VERSION = 2
SAMPLING_STREAM_DERIVATION_ID = "numpy-seedsequence-safe-seed-crc32-utf8-spec-id"
SAMPLING_STREAM_DERIVATION_VERSION = 1


def _clip(values: np.ndarray, spec: NoiseSpec) -> np.ndarray:
    """Apply one specification's deterministic absolute bounds."""
    lower = -np.inf if spec.lower is None else spec.lower
    upper = np.inf if spec.upper is None else spec.upper
    result: np.ndarray = np.clip(values, lower, upper)
    return result


def _covariance_factor(covariance: np.ndarray) -> np.ndarray:
    """Return the deterministic positive-semidefinite square root."""
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    clipped = np.maximum(eigenvalues, 0.0)
    result: np.ndarray = eigenvectors @ np.diag(np.sqrt(clipped)) @ eigenvectors.T
    return result


class _PlanSampler:
    """Stateful subset-stable streams that emit one bounded row block."""

    def __init__(self, plan: VariationPlan) -> None:
        self._plan = plan
        self._base = plan.resolved_base()
        self._specs = self._specs_by_id(plan)
        self._streams = {
            spec_id: np.random.default_rng(
                [plan.seed, zlib.crc32(spec_id.encode("utf-8"))]
            )
            for spec_id in self._specs
        }
        self._grouped_ids = frozenset(
            spec_id for group in plan.groups for spec_id in group.spec_ids
        )

    @staticmethod
    def _specs_by_id(plan: VariationPlan) -> dict[str, NoiseSpec]:
        result: dict[str, NoiseSpec] = {}
        for spec in plan.noise:
            assert spec.spec_id is not None
            result[spec.spec_id] = spec
        return result

    def next(self, row_count: int) -> np.ndarray:
        """Return the next canonical sampled-input rows."""
        sampled: dict[str, np.ndarray] = {}
        for group in self._plan.groups:
            sampled.update(self._sample_group(group, row_count))
        for spec_id in self._specs:
            if spec_id not in self._grouped_ids:
                sampled[spec_id] = self._sample_independent(spec_id, row_count)
        columns = []
        for spec in self._plan.noise:
            assert spec.spec_id is not None
            columns.append(sampled[spec.spec_id])
        result: np.ndarray = np.column_stack(columns)
        return result

    def _sample_group(
        self, group: PerturbationGroup, row_count: int
    ) -> dict[str, np.ndarray]:
        specs = tuple(self._specs[spec_id] for spec_id in group.spec_ids)
        covariance = group.covariance_matrix([spec.scale for spec in specs])
        independent = np.column_stack(
            [
                self._streams[spec_id].standard_normal(row_count)
                for spec_id in group.spec_ids
            ]
        )
        factor = _covariance_factor(covariance)
        deviations = np.sum(
            independent[:, :, np.newaxis] * factor.T[np.newaxis, :, :], axis=1
        )
        return {
            spec_id: _clip(self._base[spec.variable_key] + deviations[:, index], spec)
            for index, (spec_id, spec) in enumerate(
                zip(group.spec_ids, specs, strict=True)
            )
        }

    def _sample_independent(self, spec_id: str, row_count: int) -> np.ndarray:
        spec = self._specs[spec_id]
        center = self._base[spec.variable_key]
        stream = self._streams[spec_id]
        if spec.distribution == "normal":
            values = stream.normal(center, spec.scale, row_count)
        elif spec.distribution == "uniform":
            values = stream.uniform(center - spec.scale, center + spec.scale, row_count)
        else:
            values = stream.triangular(
                center - spec.scale, center, center + spec.scale, row_count
            )
        return _clip(values, spec)


def sample_input_chunks(
    plan: VariationPlan, *, chunk_size: int, start_index: int = 0
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield deterministic sampled rows while retaining at most one chunk.

    Resume regenerates and discards the deterministic prefix without retaining
    it. This performs no solver work and preserves the exact canonical stream.
    """
    require(isinstance(plan, VariationPlan), "plan must be a VariationPlan")
    require(
        type(chunk_size) is int and chunk_size > 0,
        "chunk_size must be a positive integer",
    )
    require(
        type(start_index) is int and 0 <= start_index <= plan.n_runs,
        "start_index must lie within the plan",
    )
    return _iter_input_chunks(plan, chunk_size, start_index)


def _iter_input_chunks(
    plan: VariationPlan, chunk_size: int, start_index: int
) -> Iterator[tuple[int, np.ndarray]]:
    sampler = _PlanSampler(plan)
    for block_start in range(0, plan.n_runs, chunk_size):
        row_count = min(chunk_size, plan.n_runs - block_start)
        block = sampler.next(row_count)
        block_stop = block_start + row_count
        if block_stop <= start_index:
            continue
        offset = max(0, start_index - block_start)
        result = np.array(block[offset:], dtype=float, copy=True)
        result.setflags(write=False)
        yield max(block_start, start_index), result


def sample_inputs(plan: VariationPlan) -> np.ndarray:
    """Return the canonical eager matrix through the same sampling authority."""
    _, block = next(sample_input_chunks(plan, chunk_size=plan.n_runs))
    result: np.ndarray = np.array(block, dtype=float, copy=True)
    return result


def sample_input_block(
    plan: VariationPlan, *, start_index: int, row_count: int
) -> np.ndarray:
    """Regenerate one exact bounded canonical block for validation or replay."""
    require(
        type(row_count) is int and row_count > 0,
        "row_count must be a positive integer",
    )
    require(
        type(start_index) is int
        and 0 <= start_index < plan.n_runs
        and start_index + row_count <= plan.n_runs,
        "requested block must lie within the plan",
    )
    result: np.ndarray = np.empty((row_count, len(plan.noise)), dtype=float)
    written = 0
    for _, values in sample_input_chunks(
        plan, chunk_size=row_count, start_index=start_index
    ):
        copied = min(row_count - written, len(values))
        result[written : written + copied] = values[:copied]
        written += copied
        if written == row_count:
            break
    require(written == row_count, "sampled-input block is incomplete")
    result.setflags(write=False)
    return result


__all__ = [
    "SAMPLING_ALGORITHM_ID",
    "SAMPLING_ALGORITHM_VERSION",
    "SAMPLING_STREAM_DERIVATION_ID",
    "SAMPLING_STREAM_DERIVATION_VERSION",
    "sample_input_block",
    "sample_input_chunks",
    "sample_inputs",
]
