import type { ClubSpec } from "../model/club";
import type { ImpactScenario } from "../model/impact";
import type { SimulationRunTs } from "../model/simulation";
import { wedgeGroundClearance, type ContactSequence } from "../model/wedgeGroundClearance";

interface Props {
  run: SimulationRunTs;
  scenario: ImpactScenario;
  club: ClubSpec;
}

const sequenceLabels: Record<ContactSequence, string> = {
  ball_first: "Ball First",
  ground_first: "Ground First",
  simultaneous: "Simultaneous",
  ball_only: "Ball Only",
  ground_only_miss: "Ground Only — Ball Missed",
  no_contact_miss: "No Contact — Ball Missed",
};

const sequenceColors: Record<ContactSequence, string> = {
  ball_first: "border-emerald-400/60 bg-emerald-500/15 text-emerald-200",
  ball_only: "border-emerald-400/60 bg-emerald-500/15 text-emerald-200",
  ground_first: "border-rose-400/60 bg-rose-500/15 text-rose-200",
  simultaneous: "border-amber-400/60 bg-amber-500/15 text-amber-200",
  ground_only_miss: "border-slate-500 bg-slate-700/40 text-slate-200",
  no_contact_miss: "border-slate-500 bg-slate-700/40 text-slate-200",
};

const metric = (value: number | null, scale: number, unit: string, decimals: number) =>
  value === null ? "Unavailable" : `${(value * scale).toFixed(decimals)} ${unit}`;
const titleCaseFeature = (value: string) => value
  .split("_").map((word) => word[0].toUpperCase() + word.slice(1)).join(" ");

export function WedgeGroundClearancePanel({ run, scenario, club }: Props) {
  const result = wedgeGroundClearance(run, scenario, club);
  if (result === null) return null;
  const entries = [
    ["Leading-Edge Clearance at Ball", metric(result.metrics.leadingEdgeClearanceAtBallM, 1000, "mm", 2)],
    ["Sole-Entry Margin", metric(result.metrics.soleEntryMarginM, 1000, "mm", 2)],
    ["Minimum Pre-Ball Clearance", metric(result.metrics.minimumPreBallClearanceM, 1000, "mm", 2)],
    ["Ground-Contact Lead / Lag", metric(result.metrics.groundAfterBallTimeMarginS, 1000, "ms", 2)],
    ["Delivered Bounce", metric(result.metrics.deliveredBounceDegAtBall, 1, "°", 2)],
    ["Path-Projected Effective Bounce", metric(result.metrics.pathProjectedEffectiveBounceDegAtBall, 1, "°", 2)],
    ["Reference-Point AoA", metric(result.metrics.referenceAoaDegAtBall, 1, "°", 2)],
    ["Bounce-Utilization Margin", metric(result.metrics.bounceUtilizationMarginDeg, 1, "°", 2)],
  ];
  const contact = result.firstGroundContact;
  return (
    <aside aria-label="Wedge Ground-Clearance Engineering Readout"
      className="mb-3 rounded-lg border border-emerald-400/30 bg-emerald-950/10 p-3">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="font-semibold text-emerald-200">Wedge Ground-Clearance Envelope</h3>
          <p className="text-xs text-slate-400">Swept rigid-head geometry · app frame: x target, y up, z right</p>
        </div>
        <span role="status" aria-label="Wedge contact sequence"
          className={`rounded-full border px-3 py-1 text-xs font-semibold ${sequenceColors[result.sequence]}`}>
          {sequenceLabels[result.sequence]}
        </span>
      </div>
      <div aria-label="Contact ordering" className="mb-3 flex items-center gap-2 rounded border border-slate-700/70 bg-slate-950/60 px-3 py-2 text-xs">
        <span className="font-semibold text-cyan-200">Ball {result.ballContactTimeS === null ? "Missed" : `${result.ballContactTimeS.toFixed(3)} s`}</span>
        <span aria-hidden="true" className="h-px flex-1 bg-gradient-to-r from-cyan-400 via-slate-500 to-emerald-400" />
        <span className="text-right font-semibold text-emerald-200">
          {contact === null ? "No Ground Contact" : `${titleCaseFeature(contact.feature)} ${contact.timeS.toFixed(3)} s`}
        </span>
      </div>
      <dl className="grid gap-2 text-sm sm:grid-cols-2 xl:grid-cols-4">
        {entries.map(([label, value]) => <div key={label}
          className="rounded border border-slate-700/70 bg-slate-900/60 p-2">
          <dt className="text-xs text-slate-400">{label}</dt>
          <dd className="font-mono text-slate-100">{value}</dd>
        </div>)}
      </dl>
      <p className="mt-2 text-xs leading-relaxed text-slate-400">
        <b className="text-slate-300">Geometry Basis:</b> {titleCaseFeature(result.geometryBasis)}. {result.provenance}{" "}
        <b className="text-slate-300">Model Boundary:</b> {result.limitations}
      </p>
    </aside>
  );
}
