"""Bounded deterministic work sources for simulation ensembles."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from rate_of_closure.simulation import SimulationConfig
from shared.python.contracts import require
from shared.python.swing_sim.variation import VariationPlan, sample_input_chunks

ConfigurationFactory = Callable[[np.ndarray], SimulationConfig]


@dataclass(frozen=True, slots=True)
class EnsembleWorkChunk:
    """One bounded contiguous block of inputs and complete configurations."""

    start_index: int
    sampled_inputs: np.ndarray = field(repr=False)
    configs: tuple[SimulationConfig, ...]

    def __post_init__(self) -> None:
        require(
            type(self.start_index) is int and self.start_index >= 0,
            "start_index must be a non-negative integer",
        )
        inputs = np.array(self.sampled_inputs, dtype=float, copy=True)
        configs = tuple(self.configs)
        require(inputs.ndim == 2, "sampled_inputs must be a matrix")
        require(
            inputs.shape[0] == len(configs) and len(configs) > 0,
            "work inputs and configurations must have equal non-zero rows",
        )
        require(bool(np.all(np.isfinite(inputs))), "sampled_inputs must be finite")
        require(
            all(isinstance(config, SimulationConfig) for config in configs),
            "configs must contain only SimulationConfig values",
        )
        inputs.setflags(write=False)
        object.__setattr__(self, "sampled_inputs", inputs)
        object.__setattr__(self, "configs", configs)


@runtime_checkable
class SimulationEnsembleSource(Protocol):
    """Deterministic bounded source consumed by ensemble executors."""

    @property
    def plan(self) -> VariationPlan: ...

    def reference_config(self) -> SimulationConfig: ...

    def work_chunks(
        self, *, chunk_size: int, start_index: int = 0
    ) -> Iterator[EnsembleWorkChunk]: ...


@dataclass(frozen=True, slots=True)
class LazySimulationEnsembleSource:
    """Generate sampled rows and configurations without retaining a roster."""

    plan: VariationPlan
    _config_factory: ConfigurationFactory = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        require(isinstance(self.plan, VariationPlan), "plan must be a VariationPlan")
        require(callable(self._config_factory), "config_factory must be callable")

    def reference_config(self) -> SimulationConfig:
        """Return the first complete configuration without retaining all trials."""
        _, inputs = next(sample_input_chunks(self.plan, chunk_size=1))
        config = self._config_factory(inputs[0])
        require(
            isinstance(config, SimulationConfig),
            "config_factory must return SimulationConfig",
        )
        return config

    def work_chunks(
        self, *, chunk_size: int, start_index: int = 0
    ) -> Iterator[EnsembleWorkChunk]:
        """Yield bounded canonical rows and their complete configurations."""
        chunks = sample_input_chunks(
            self.plan, chunk_size=chunk_size, start_index=start_index
        )
        return self._map_chunks(chunks)

    def _map_chunks(
        self, chunks: Iterator[tuple[int, np.ndarray]]
    ) -> Iterator[EnsembleWorkChunk]:
        for start_index, inputs in chunks:
            configs = tuple(self._config_factory(row) for row in inputs)
            yield EnsembleWorkChunk(start_index, inputs, configs)


__all__ = [
    "EnsembleWorkChunk",
    "LazySimulationEnsembleSource",
    "SimulationEnsembleSource",
]
