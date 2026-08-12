"""Authenticated local authority boundary for the Rate of Closure web client."""

from rate_of_closure.application.regional_ground_authority_status import (
    AUTHORITY_JOB_STATUS_SCHEMA_VERSION,
)

from .api import create_authority_app
from .capability import AUTHORITY_CAPABILITY_SCHEMA_VERSION, AuthorityCapability
from .jobs import AuthorityJobManager
from .production_runner import (
    ProductionRunnerPreflightReason,
    RegionalGroundProductionPreflightError,
    preflight_regional_ground_production_job,
    run_regional_ground_production_job,
)

__all__ = [
    "AUTHORITY_CAPABILITY_SCHEMA_VERSION",
    "AUTHORITY_JOB_STATUS_SCHEMA_VERSION",
    "AuthorityCapability",
    "AuthorityJobManager",
    "ProductionRunnerPreflightReason",
    "RegionalGroundProductionPreflightError",
    "create_authority_app",
    "preflight_regional_ground_production_job",
    "run_regional_ground_production_job",
]
