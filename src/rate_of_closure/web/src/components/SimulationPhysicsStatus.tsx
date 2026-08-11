/** Capability-honest selected-club and assembly status for simulation. */

import { type ClubSpec } from "../model/club";
import { type ClubAssemblyBinding } from "../model/clubAssemblyBinding";
import { DEFAULT_IMPACT_CLUB, type SimulationRunTs } from "../model/simulation";

interface Props {
  clubSpec: ClubSpec | null;
  assemblyBinding?: ClubAssemblyBinding;
  run: SimulationRunTs | null;
  runIsStale: boolean;
}

function clubGuidance(clubSpec: ClubSpec | null): string {
  if (!clubSpec) {
    return `No selected club specification was provided. Impact physics uses the default driver: ${DEFAULT_IMPACT_CLUB.headMassKg.toFixed(3)} kg head mass, ${DEFAULT_IMPACT_CLUB.moiAboutShaftKgM2.toExponential(2)} kg m² MOI, and ${DEFAULT_IMPACT_CLUB.coefficientOfRestitution.toFixed(2)} COR.`;
  }
  return `Impact physics uses ${clubSpec.name}: ${clubSpec.headMassKg.toFixed(3)} kg head mass, ${clubSpec.moiAboutShaftKgM2.toExponential(2)} kg m² MOI, and ${clubSpec.loftDeg.toFixed(1)}° nominal loft. COR uses the ${DEFAULT_IMPACT_CLUB.coefficientOfRestitution.toFixed(2)} driver default because the club library does not yet define measured COR.`;
}

function assemblyGuidance({
  assemblyBinding,
  run,
  runIsStale,
}: Omit<Props, "clubSpec">): string {
  if (run && !runIsStale) {
    const usage = run.clubAssemblyUsage;
    return `Head inertia: ${usage.headInertia.status} — ${usage.headInertia.reason} Head CG: ${usage.headCenterOfMass.status} — ${usage.headCenterOfMass.reason} Assembly properties: ${usage.assemblyMassProperties.status} — ${usage.assemblyMassProperties.reason}`;
  }
  if (assemblyBinding) {
    return `Assembly binding ${assemblyBinding.assemblyIdentity.assemblyId} is loaded for this exact club. The browser impact solver consumes the validated head mass only; its scalar-MOI path cannot consume the full head tensor or CG, and assembled-club mass properties are never substituted.`;
  }
  return "No validated assembly binding is active. Complete head CG/tensor and assembled-club mass properties remain unavailable.";
}

export function SimulationPhysicsStatus(props: Props) {
  const noteClass =
    "mb-3 rounded-lg border border-slate-700/80 bg-slate-950/50 px-3 " +
    "py-2 text-xs leading-relaxed text-slate-400";
  return (
    <>
      <p
        role="note"
        aria-label="Impact club physics"
        title={clubGuidance(props.clubSpec)}
        className={noteClass}
      >
        {clubGuidance(props.clubSpec)}
      </p>
      <p
        role="note"
        aria-label="Club assembly simulation binding"
        className={noteClass}
      >
        {assemblyGuidance(props)}
      </p>
    </>
  );
}
