import type { ClubSpec } from "../model/club";
import type { ImpactScenario } from "../model/impact";
import { impactKinematics } from "../model/impactKinematics";
import type { SimulationRunTs } from "../model/simulation";

interface Props {
  run: SimulationRunTs;
  scenario: ImpactScenario;
  club: ClubSpec;
}

const number = (value: number | null, unit: string, decimals = 2) =>
  value === null ? "Unavailable" : `${value.toFixed(decimals)} ${unit}`;

export function ImpactKinematicsPanel({ run, scenario, club }: Props) {
  const metrics = impactKinematics(run, scenario, club);
  const entries = [
    ["Reference-Point AoA", number(metrics.referenceAoaDeg, "°")],
    ["Contact-Point AoA", number(metrics.contactAoaDeg, "°")],
    ["Without Shaft Rotation", number(metrics.withoutShaftAoaDeg, "°")],
    ["Shaft AoA Contribution", number(metrics.shaftAoaContributionDeg, "°")],
    ["Shaft Rotation Rate", number(metrics.shaftRotationRateDps, "°/s", 1)],
    ["Shaft-Induced Vertical Velocity", number(metrics.shaftVerticalVelocityMps, "m/s", 3)],
    ["Face-Normal 3D Rate", number(metrics.faceNormalRateDps, "°/s", 1)],
  ];
  return (
    <aside aria-label="Impact Kinematics Engineering Readout"
      className="mb-3 rounded-lg border border-cyan-400/30 bg-cyan-950/10 p-3">
      <div className="mb-2 flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="font-semibold text-cyan-200">{metrics.eventLabel} Kinematics</h3>
        <span className="text-xs tabular-nums text-cyan-300/80">
          {metrics.eventTimeS.toFixed(3)} s · app frame: x target, y up, z right
        </span>
      </div>
      <dl className="grid gap-2 text-sm sm:grid-cols-2 xl:grid-cols-4">
        {entries.map(([label, value]) => <div key={label}
          className="rounded border border-slate-700/70 bg-slate-900/60 p-2">
          <dt className="text-xs text-slate-400">{label}</dt>
          <dd className="font-mono text-slate-100">{value}</dd>
        </div>)}
      </dl>
      <p className="mt-2 text-xs text-slate-400">
        <b className="text-slate-300">Geometry Basis:</b> {metrics.geometryBasis}.{" "}
        <b className="text-slate-300">Model Boundary:</b> {metrics.modelLimitations}
      </p>
    </aside>
  );
}
