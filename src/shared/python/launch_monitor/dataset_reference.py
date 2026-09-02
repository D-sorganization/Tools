"""Stable facade for immutable, aggregate-only private-dataset jobs.

Ported from UpstreamDrift ``src/shared/python/launch_monitor/dataset_reference.py``
(35 lines) under ADR-0046 Stage 1 — step **P20** of the ADR-0046 G1 port plan
(UpstreamDrift ``docs/adr/0048-launch-monitor-port-plan.md``), the final row of
the ladder. The implementation is UpstreamDrift's, carried over unchanged rather
than reimplemented; its authors retain authorship. This module is
**AST-identical** to UpstreamDrift's modulo this docstring and the plan's
``src.shared.python.launch_monitor.X`` to ``shared.python.launch_monitor.X``
import rewrite.

The plan's inventory classifies this file a pure facade over the three modules
below it (``dataset_reference_contract``, ``dataset_reference_operations``,
``dataset_reference_verification``), which is exactly what it is: the seam the
API service and the contract generator import, so that the split between
contract, verification and execution is not part of the consumer surface.
"""

from shared.python.launch_monitor.dataset_reference_contract import (
    DATASET_JOB_CONTRACT_VERSION,
    MAX_PAGE_SIZE,
    DatasetJobRequestV1,
    DatasetOperationV1,
    DatasetReferenceV1,
    DatasetUnavailableError,
    DatasetUnavailableStateV1,
    dataset_job_contract_json_schema,
)
from shared.python.launch_monitor.dataset_reference_operations import (
    execute_dataset_operation,
)
from shared.python.launch_monitor.dataset_reference_verification import (
    VerifiedDataset,
    dataset_content_sha256,
    verify_dataset_reference,
)

__all__ = [
    "DATASET_JOB_CONTRACT_VERSION",
    "MAX_PAGE_SIZE",
    "DatasetJobRequestV1",
    "DatasetOperationV1",
    "DatasetReferenceV1",
    "DatasetUnavailableError",
    "DatasetUnavailableStateV1",
    "VerifiedDataset",
    "dataset_content_sha256",
    "dataset_job_contract_json_schema",
    "execute_dataset_operation",
    "verify_dataset_reference",
]
