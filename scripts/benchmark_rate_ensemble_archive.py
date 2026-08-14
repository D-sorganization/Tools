"""Measure result-sink RSS after the explicitly eager request baseline."""

from __future__ import annotations

import argparse
import gc
import json
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import psutil

from rate_of_closure.club import get_club
from rate_of_closure.model import ImpactScenario
from rate_of_closure.simulation import ContactMode, SimulationConfig
from rate_of_closure.simulation.contact import assess_fixed_contact
from rate_of_closure.variation.ensemble_archive import DurableEnsembleArchiveSink
from rate_of_closure.variation.ensemble_chunks import (
    EnsembleStreamHeader,
    SimulationResultChunk,
)
from rate_of_closure.variation.ensemble_request_identity import request_identity_sha256
from rate_of_closure.variation.ensemble_trace_authority import (
    ChunkTraceAuthority,
    EnsembleAuthorityLayout,
    event_for_grid,
)
from rate_of_closure.variation.simulation_types import (
    ALL_OUTPUT_NAMES,
    APP_FRAME_ID,
    EVALUATED_NO_IMPACT,
    SimulationEnsembleRequest,
    SimulationTrialOutcome,
)
from shared.python.swing_sim.variation import NoiseSpec, VariationPlan


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=int, required=True)
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--samples", type=int, default=128)
    return parser.parse_args()


class _RssSampler:
    def __init__(self) -> None:
        self._process = psutil.Process()
        self._stop = threading.Event()
        self.peak = self._process.memory_info().rss
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(0.001):
            self.peak = max(self.peak, self._process.memory_info().rss)

    def __enter__(self) -> _RssSampler:
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self._process.memory_info().rss)


def _request(trials: int) -> SimulationEnsembleRequest:
    plan = VariationPlan(
        mode="delivery",
        # A variable the request's per-row config binding can verify; the
        # benchmark measures archive RSS scaling, not any one variable.
        noise=(NoiseSpec("swing_sim.impact.delivery.impact_offset_toe_mm", scale=1.0),),
        n_runs=trials,
        seed=17,
    )
    config = SimulationConfig(
        scenario=ImpactScenario(clubhead_speed_mph=100.0),
        club=get_club("Driver 10.5°"),
        source_kind="manual",
        contact_mode=ContactMode.FIXED_BALL_CONTACT,
    )
    # A synthetic all-zero fixture, not the plan's RNG draws, so it declares
    # that provenance rather than claiming reproducibility from the seed.
    return SimulationEnsembleRequest(
        plan,
        np.zeros((trials, 1)),
        (config,) * trials,
        sample_provenance="explicit_design",
    )


def _header(request: SimulationEnsembleRequest, samples: int) -> EnsembleStreamHeader:
    return EnsembleStreamHeader(
        request.plan,
        request.sampled_inputs,
        np.linspace(0.0, 0.03, samples),
        ("swing.clubhead.reference",),
        APP_FRAME_ID,
        EnsembleAuthorityLayout((), (), ()),
        request_identity_sha256(request),
    )


def _outcome(index: int, event) -> SimulationTrialOutcome:  # type: ignore[no-untyped-def]
    values = dict.fromkeys(ALL_OUTPUT_NAMES)
    values["candidate_time_s"] = event.candidate_time_s
    values["closest_approach_m"] = event.closest_approach_m
    values["contact_margin_m"] = event.contact_margin_m
    return SimulationTrialOutcome(index, EVALUATED_NO_IMPACT, values)


def _chunk(
    header: EnsembleStreamHeader, start: int, rows: int
) -> SimulationResultChunk:
    samples = header.sample_times_s.size
    poses = np.broadcast_to(np.eye(4), (rows, samples, 4, 4)).copy()
    positions = np.zeros((rows, samples, 1, 3))
    contact = assess_fixed_contact(
        header.sample_times_s,
        np.full((samples, 3), 1.0),
        np.zeros(3),
        0.02135,
    )
    events = tuple(
        event_for_grid(start + row, contact, header.sample_times_s)
        for row in range(rows)
    )
    authority = ChunkTraceAuthority(
        poses,
        np.zeros((rows, samples, 6)),
        np.empty((rows, samples, 0)),
        np.empty((rows, samples, 0)),
        np.ones((rows, samples), dtype=bool),
        events,
    )
    return SimulationResultChunk(
        start,
        header.sampled_inputs[start : start + rows],
        tuple(_outcome(start + row, contact) for row in range(rows)),
        positions,
        np.ones((rows, samples), dtype=bool),
        np.full(rows, -1, dtype=int),
        authority,
    )


def main() -> None:
    args = _arguments()
    trials = args.chunks * args.rows
    request = _request(trials)
    header = _header(request, args.samples)
    gc.collect()
    process = psutil.Process()
    baseline = process.memory_info().rss
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as directory, _RssSampler() as sampler:
        sink = DurableEnsembleArchiveSink(Path(directory) / "archive")
        sink.begin(header)
        for start in range(0, trials, args.rows):
            chunk = _chunk(header, start, args.rows)
            sink.accept(chunk)
            del chunk
        sink.commit(time.perf_counter() - started)
    print(
        json.dumps(
            {
                "chunks": args.chunks,
                "trials": trials,
                "samples": args.samples,
                "baseline_rss_bytes": baseline,
                "peak_rss_bytes": sampler.peak,
                "peak_delta_bytes": max(0, sampler.peak - baseline),
                "elapsed_s": time.perf_counter() - started,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
