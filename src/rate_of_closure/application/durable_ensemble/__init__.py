"""Single-authority contracts and lifecycle for durable ensembles."""

from .client import (
    DurableEnsembleAuthorityClient,
    DurableEnsembleAuthorityHttpError,
    DurableEnsembleCapability,
)
from .contracts import (
    DURABLE_ENSEMBLE_JOB_SCHEMA_ID,
    DURABLE_ENSEMBLE_REQUEST_SCHEMA_ID,
    DURABLE_ENSEMBLE_SCOPE,
    DurableEnsembleAuthorityRequest,
    DurableEnsembleJobEnvelope,
    durable_ensemble_request_document,
    parse_durable_ensemble_job,
    parse_durable_ensemble_request,
)
from .registry import (
    DurableEnsembleExecutionService,
    DurableEnsembleJobRegistry,
    DurableEnsembleRegistryOptions,
)
from .router import create_durable_ensemble_router
from .service import EvidenceSink, RateDurableEnsembleService

__all__ = [
    "DURABLE_ENSEMBLE_JOB_SCHEMA_ID",
    "DURABLE_ENSEMBLE_REQUEST_SCHEMA_ID",
    "DURABLE_ENSEMBLE_SCOPE",
    "DurableEnsembleAuthorityRequest",
    "DurableEnsembleAuthorityClient",
    "DurableEnsembleAuthorityHttpError",
    "DurableEnsembleCapability",
    "DurableEnsembleJobEnvelope",
    "DurableEnsembleExecutionService",
    "DurableEnsembleJobRegistry",
    "DurableEnsembleRegistryOptions",
    "EvidenceSink",
    "RateDurableEnsembleService",
    "durable_ensemble_request_document",
    "create_durable_ensemble_router",
    "parse_durable_ensemble_request",
    "parse_durable_ensemble_job",
]
