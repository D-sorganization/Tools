"""Lazy single-authority contracts and lifecycle for durable ensembles."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULE = {
    "DURABLE_ENSEMBLE_JOB_SCHEMA_ID": ".contracts",
    "DURABLE_ENSEMBLE_REQUEST_SCHEMA_ID": ".contracts",
    "DURABLE_ENSEMBLE_SCOPE": ".contracts",
    "DurableEnsembleAuthorityRequest": ".contracts",
    "DurableEnsembleJobEnvelope": ".contracts",
    "durable_ensemble_request_document": ".contracts",
    "parse_durable_ensemble_job": ".contracts",
    "parse_durable_ensemble_request": ".contracts",
    "DurableEnsembleAuthorityClient": ".client",
    "DurableEnsembleAuthorityHttpError": ".client",
    "DurableEnsembleCapability": ".client",
    "DurableEnsembleExecutionService": ".registry",
    "DurableEnsembleJobRegistry": ".registry",
    "DurableEnsembleRegistryOptions": ".registry",
    "create_durable_ensemble_router": ".router",
    "EvidenceSink": ".service",
    "RateDurableEnsembleService": ".service",
}


def __getattr__(name: str) -> Any:
    """Load server dependencies only when their public export is requested."""
    module_name = _EXPORT_MODULE.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value

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
