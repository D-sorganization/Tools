/** Fail-closed ClubAssembly binding boundary for scalar-only web impact. */

import { type ClubSpec } from "./club";
import {
  assertBindingMatchesSpec,
  type ClubAssemblyBinding,
} from "./clubAssemblyBinding";

export type SimulationCapabilityStatus =
  "available" | "unavailable" | "not_used";

export interface SimulationCapabilityUse {
  status: SimulationCapabilityStatus;
  consumed: boolean;
  reason: string;
}

export interface ClubAssemblyImpactInputs {
  headMassKg: number;
  headInertiaTensorAppKgM2: null;
  headInertia: SimulationCapabilityUse;
  headCenterOfMass: SimulationCapabilityUse;
  assemblyMassProperties: SimulationCapabilityUse;
}

const unavailable = (reason: string): SimulationCapabilityUse => ({
  status: "unavailable",
  consumed: false,
  reason,
});

/** Describe the fail-closed boundary when no selected-spec binding exists. */
export function unboundClubAssemblyImpact(
  headMassKg: number,
): ClubAssemblyImpactInputs {
  if (!Number.isFinite(headMassKg) || headMassKg <= 0) {
    throw new Error("selected club head mass must be finite and positive");
  }
  const reason =
    "unavailable because no validated ClubAssembly binding is configured";
  return {
    headMassKg,
    headInertiaTensorAppKgM2: null,
    headInertia: unavailable(reason),
    headCenterOfMass: unavailable(reason),
    assemblyMassProperties: unavailable(reason),
  };
}

/** Resolve only properties the current browser impact solver can consume. */
export function adaptClubAssemblyForImpact(
  spec: ClubSpec,
  binding?: ClubAssemblyBinding,
): ClubAssemblyImpactInputs {
  if (!binding) {
    return unboundClubAssemblyImpact(spec.headMassKg);
  }
  assertBindingMatchesSpec(binding, spec);
  return {
    headMassKg: binding.headPropertiesInSelectedFrame.mass_kg,
    headInertiaTensorAppKgM2: null,
    headInertia: unavailable(
      "web impact solver is scalar-MOI-only and cannot consume a full head-CG tensor",
    ),
    headCenterOfMass: unavailable(
      "web impact solver does not accept a full head-CG vector with its declared datum",
    ),
    assemblyMassProperties: unavailable(
      "impact solver requires head properties; must not substitute assembled-club mass, CG, or inertia",
    ),
  };
}

/** Mark every bound property unconsumed when contact detection reports a miss. */
export function withoutClubBallImpact(
  adapted: ClubAssemblyImpactInputs,
): ClubAssemblyImpactInputs {
  const notUsed: SimulationCapabilityUse = {
    status: "not_used",
    consumed: false,
    reason: "not consumed because no club-ball impact occurred",
  };
  return {
    ...adapted,
    headInertiaTensorAppKgM2: null,
    headInertia: notUsed,
    headCenterOfMass: notUsed,
    assemblyMassProperties: notUsed,
  };
}
