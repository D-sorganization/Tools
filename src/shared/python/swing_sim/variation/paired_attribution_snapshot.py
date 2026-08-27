"""Bounded exact snapshots for paired localized attribution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import cast

import numpy as np

from shared.python.contracts import require

from .paired_attribution_record import PairedAttributionRecord
from .paired_attribution_types import (
    MAX_ARCHIVE_BYTES,
    MAX_OBSERVATIONS,
    MAX_PAIRS,
    AttributionPair,
    PairedAttributionContract,
    PairedAttributionInput,
)

_SNAPSHOT_SCHEMA_ID = "swing-sim/paired-attribution-snapshot"
_SNAPSHOT_SCHEMA_VERSION = 1


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


@dataclass(frozen=True)
class PairedAttributionSnapshot:
    """Bounded exact prefix for deterministic attribution resume."""

    contract_sha256: str
    accepted_pairs: int
    pairs: tuple[AttributionPair, ...]

    def __post_init__(self) -> None:
        require(
            len(self.contract_sha256) == 64
            and set(self.contract_sha256) <= set("0123456789abcdef"),
            "contract_sha256 must be a lowercase SHA-256",
        )
        pairs = tuple(self.pairs)
        require(self.accepted_pairs == len(pairs), "accepted pair count mismatch")
        require(len(pairs) <= MAX_PAIRS, "snapshot pair count exceeds resource cap")
        require(
            len({pair.pair_id for pair in pairs}) == len(pairs),
            "snapshot pair IDs must be unique",
        )
        object.__setattr__(self, "pairs", pairs)


class PairedAttributionAccumulator:
    """Deterministic bounded accumulator for pair chunks and resume snapshots."""

    def __init__(
        self,
        contract: PairedAttributionContract,
        snapshot: PairedAttributionSnapshot | None = None,
    ) -> None:
        require(isinstance(contract, PairedAttributionContract), "invalid contract")
        self._contract = contract
        self._pairs: list[AttributionPair] = []
        if snapshot is not None:
            from .paired_attribution import attribution_contract_fingerprint

            require(
                snapshot.contract_sha256 == attribution_contract_fingerprint(contract),
                "snapshot contract mismatch",
            )
            self._pairs.extend(snapshot.pairs)

    def accept(self, chunk: PairedAttributionInput) -> None:
        """Append one exact contract-matched chunk in caller-declared order."""
        require(isinstance(chunk, PairedAttributionInput), "invalid attribution chunk")
        require(
            chunk.contract_without_pairs() == self._contract, "chunk contract mismatch"
        )
        combined = self._pairs + list(chunk.pairs)
        require(len(combined) <= MAX_PAIRS, "pair count exceeds resource cap")
        require(
            len(combined) * len(self._contract.targets) <= MAX_OBSERVATIONS,
            "pair-target matrix exceeds resource cap",
        )
        require(
            len({pair.pair_id for pair in combined}) == len(combined),
            "pair IDs must be unique",
        )
        self._pairs = combined

    def snapshot(self) -> PairedAttributionSnapshot:
        """Return an immutable replayable exact-prefix snapshot."""
        from .paired_attribution import attribution_contract_fingerprint

        return PairedAttributionSnapshot(
            attribution_contract_fingerprint(self._contract),
            len(self._pairs),
            tuple(self._pairs),
        )

    def finalize(self) -> PairedAttributionRecord:
        """Materialize the final immutable record once at the analysis boundary."""
        from .paired_attribution import compute_paired_attribution

        field_input = PairedAttributionInput(
            self._contract.source,
            self._contract.targets,
            tuple(self._pairs),
            self._contract.context,
            self._contract.context,
            self._contract.source_sha256,
        )
        return compute_paired_attribution(field_input)


def _nullable_numbers(values: np.ndarray) -> list[float | None]:
    return [float(value) if np.isfinite(value) else None for value in values]


def _pair_payload(pair: AttributionPair) -> dict[str, object]:
    return {
        "pair_id": pair.pair_id,
        "baseline_trial_id": pair.baseline_trial_id,
        "perturbed_trial_id": pair.perturbed_trial_id,
        "baseline_status": pair.baseline_status,
        "perturbed_status": pair.perturbed_status,
        "baseline_source_value": pair.baseline_source_value,
        "perturbed_source_value": pair.perturbed_source_value,
        "baseline_values": _nullable_numbers(pair.baseline_values),
        "perturbed_values": _nullable_numbers(pair.perturbed_values),
        "baseline_value_states": list(pair.baseline_value_states),
        "perturbed_value_states": list(pair.perturbed_value_states),
    }


def snapshot_to_json(snapshot: PairedAttributionSnapshot) -> str:
    """Serialize one strict bounded snapshot without non-standard NaN tokens."""
    require(isinstance(snapshot, PairedAttributionSnapshot), "invalid snapshot")
    payload = {
        "schema_id": _SNAPSHOT_SCHEMA_ID,
        "schema_version": _SNAPSHOT_SCHEMA_VERSION,
        "contract_sha256": snapshot.contract_sha256,
        "accepted_pairs": snapshot.accepted_pairs,
        "pairs": [_pair_payload(pair) for pair in snapshot.pairs],
    }
    encoded = _canonical_json(payload)
    require(
        len(encoded.encode("utf-8")) <= MAX_ARCHIVE_BYTES, "snapshot exceeds byte cap"
    )
    return encoded


def _numbers_from_json(value: object, label: str) -> np.ndarray:
    require(isinstance(value, list), f"{label} must be an array")
    items = cast(list[object], value)
    require(
        all(item is None or isinstance(item, (int, float)) for item in items),
        f"invalid {label}",
    )
    return np.asarray(
        [np.nan if item is None else float(cast(int | float, item)) for item in items]
    )


def _pair_from_payload(value: object) -> AttributionPair:
    require(isinstance(value, dict), "snapshot pair must be an object")
    item = cast(dict[str, object], value)
    expected = {
        "pair_id",
        "baseline_trial_id",
        "perturbed_trial_id",
        "baseline_status",
        "perturbed_status",
        "baseline_source_value",
        "perturbed_source_value",
        "baseline_values",
        "perturbed_values",
        "baseline_value_states",
        "perturbed_value_states",
    }
    require(set(item) == expected, "snapshot pair fields mismatch")
    return AttributionPair(
        pair_id=cast(str, item["pair_id"]),
        baseline_trial_id=cast(str, item["baseline_trial_id"]),
        perturbed_trial_id=cast(str, item["perturbed_trial_id"]),
        baseline_status=cast(str, item["baseline_status"]),
        perturbed_status=cast(str, item["perturbed_status"]),
        baseline_source_value=cast(float, item["baseline_source_value"]),
        perturbed_source_value=cast(float, item["perturbed_source_value"]),
        baseline_values=_numbers_from_json(item["baseline_values"], "baseline values"),
        perturbed_values=_numbers_from_json(
            item["perturbed_values"], "perturbed values"
        ),
        baseline_value_states=tuple(cast(list[str], item["baseline_value_states"])),
        perturbed_value_states=tuple(cast(list[str], item["perturbed_value_states"])),
    )


def snapshot_from_json(text: str) -> PairedAttributionSnapshot:
    """Parse a strict snapshot; malformed or oversized archives fail closed."""
    require(isinstance(text, str), "snapshot must be text")
    require(len(text.encode("utf-8")) <= MAX_ARCHIVE_BYTES, "snapshot exceeds byte cap")
    raw = json.loads(text)
    require(isinstance(raw, dict), "snapshot must be an object")
    item = cast(dict[str, object], raw)
    require(
        set(item)
        == {
            "schema_id",
            "schema_version",
            "contract_sha256",
            "accepted_pairs",
            "pairs",
        },
        "snapshot fields mismatch",
    )
    require(item["schema_id"] == _SNAPSHOT_SCHEMA_ID, "snapshot schema drift")
    require(
        item["schema_version"] == _SNAPSHOT_SCHEMA_VERSION, "snapshot version drift"
    )
    require(isinstance(item["pairs"], list), "snapshot pairs must be an array")
    return PairedAttributionSnapshot(
        cast(str, item["contract_sha256"]),
        cast(int, item["accepted_pairs"]),
        tuple(_pair_from_payload(value) for value in cast(list[object], item["pairs"])),
    )


__all__ = [
    "PairedAttributionAccumulator",
    "PairedAttributionSnapshot",
    "snapshot_from_json",
    "snapshot_to_json",
]
