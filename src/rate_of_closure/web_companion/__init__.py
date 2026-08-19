"""Same-origin source production companion for Rate of Closure."""

from .app import create_companion_app
from .bundle import CompanionWebBundle, build_companion_bundle
from .contracts import (
    CompanionRequest,
    CompanionRequestRejected,
    CompanionRoute,
    CompanionRouteKind,
    classify_companion_request,
)
from .runtime import CompanionRuntime, start_companion
from .supervisor import AuthoritySupervisor

__all__ = [
    "CompanionRequest",
    "CompanionRequestRejected",
    "CompanionRoute",
    "CompanionRouteKind",
    "CompanionRuntime",
    "CompanionWebBundle",
    "AuthoritySupervisor",
    "build_companion_bundle",
    "classify_companion_request",
    "create_companion_app",
    "start_companion",
]
