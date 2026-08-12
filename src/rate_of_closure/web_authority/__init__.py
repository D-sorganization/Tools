"""Authenticated local authority boundary for the Rate of Closure web client."""

from .api import create_authority_app
from .capability import AUTHORITY_CAPABILITY_SCHEMA_VERSION, AuthorityCapability

__all__ = [
    "AUTHORITY_CAPABILITY_SCHEMA_VERSION",
    "AuthorityCapability",
    "create_authority_app",
]
