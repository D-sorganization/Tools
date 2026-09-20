"""Source-backed catalog for convention-aware launch-monitor calculations."""

from __future__ import annotations

from functools import lru_cache

from ._catalog_data import (
    FRAME,
    IDENTITIES,
    POLICIES,
    RETRIEVED,
)
from .registry import (
    ConventionRegistry,
    ParameterDefinition,
)


@lru_cache(maxsize=1)
def convention_registry() -> ConventionRegistry:
    """Return the immutable source-backed foundation registry."""
    definitions = []
    for convention, policies in POLICIES.items():
        for parameter, identity in IDENTITIES.items():
            policy = policies[parameter]
            definitions.append(
                ParameterDefinition(
                    convention_id=convention,
                    parameter_id=parameter,
                    label=identity.label,
                    source_url=policy.source,
                    retrieved_on=RETRIEVED,
                    reference_point=policy.reference,
                    event_time=policy.event,
                    frame_id=FRAME,
                    geometry_contract=identity.geometry,
                    sign_rule=policy.sign or identity.sign,
                    unit=identity.unit,
                    quantity_status=policy.status,
                    availability=policy.availability,
                )
            )
    return ConventionRegistry(tuple(definitions))
