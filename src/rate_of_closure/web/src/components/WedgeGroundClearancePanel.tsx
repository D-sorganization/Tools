import { useMemo } from "react";

import type { ClubSpec } from "../model/club";
import type { ImpactScenario } from "../model/impact";
import type { SimulationRunTs } from "../model/simulation";
import {
  wedgeDeliveryMetricsForRun,
  type MetricExplainerTs,
  type WedgeDeliveryMetricsTs,
} from "../model/wedgeDeliveryMetrics";
import type {
  ContactSequence,
  WedgeGroundClearancePayloadTs,
} from "../model/wedgeGroundClearance";

interface Props {
  result: WedgeGroundClearancePayloadTs | null;
  run?: SimulationRunTs | null;
  scenario?: ImpactScenario;
  club?: ClubSpec;
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

const titleCaseFeature = (value: string) =>
  value
    .split("_")
    .map((word) => word[0].toUpperCase() + word.slice(1))
    .join(" ");

export function WedgeGroundClearancePanel({ result, run, scenario, club }: Props) {
  const delivery: WedgeDeliveryMetricsTs | null = useMemo(() => {
    if (!result || !run || !scenario || !club) return null;
    return wedgeDeliveryMetricsForRun(run, scenario, club, result);
  }, [run, scenario, club, result]);

  // Combined explainers if delivery metrics are available (deduplicated by key)
  const allCards: MetricExplainerTs[] = useMemo(() => {
    if (!result) return [];
    const clearanceExplainers: MetricExplainerTs[] = [
      {
        key: "leading_edge_clearance",
        label: "Leading-Edge Clearance at Ball",
        value: metric(result.metrics.leadingEdgeClearanceAtBallM, 1000, "mm", 2),
        units: "mm",
        equation: "argmin_{le} (world_le · ground_normal)",
        frame: result.frameId,
        assumptions: "Shortest vertical distance from leading edge to ground at ball contact.",
        availability: result.metrics.leadingEdgeClearanceAtBallM !== null ? "available" : "undefined",
      },
      {
        key: "sole_entry_margin",
        label: "Sole-Entry Margin",
        value: metric(result.metrics.soleEntryMarginM, 1000, "mm", 2),
        units: "mm",
        equation: "clearance(trailing_sole) - clearance(leading_edge)",
        frame: result.frameId,
        assumptions: "Elevation difference between trailing edge and leading edge.",
        availability: result.metrics.soleEntryMarginM !== null ? "available" : "undefined",
      },
      {
        key: "min_pre_ball_clearance",
        label: "Minimum Pre-Ball Clearance",
        value: metric(result.metrics.minimumPreBallClearanceM, 1000, "mm", 2),
        units: "mm",
        equation: "min_{t <= t_ball} clearance(t)",
        frame: result.frameId,
        assumptions: "Smallest distance from sole envelope to ground prior to ball impact.",
        availability: result.metrics.minimumPreBallClearanceM !== null ? "available" : "undefined",
      },
      {
        key: "ground_lead_lag",
        label: "Ground-Contact Lead / Lag",
        value: metric(result.metrics.groundAfterBallTimeMarginS, 1000, "ms", 2),
        units: "ms",
        equation: "t_ground_first - t_ball",
        frame: result.frameId,
        assumptions: "Positive indicates clean ball-first impact; negative indicates turf-first impact.",
        availability: result.metrics.groundAfterBallTimeMarginS !== null ? "available" : "undefined",
      },
      {
        key: "delivered_bounce",
        label: "Delivered Bounce",
        value: metric(result.metrics.deliveredBounceDegAtBall, 1, "°", 2),
        units: "°",
        equation: "atan2(world_sole · up, |world_sole,horizontal|)",
        frame: result.frameId,
        assumptions: "Orientation angle of central sole above ground plane at impact.",
        availability: result.metrics.deliveredBounceDegAtBall !== null ? "available" : "undefined",
      },
      {
        key: "path_projected_bounce",
        label: "Path-Projected Effective Bounce",
        value: metric(result.metrics.pathProjectedEffectiveBounceDegAtBall, 1, "°", 2),
        units: "°",
        equation: "atan2(sole_vertical, sole_trailing_along_path)",
        frame: result.frameId,
        assumptions: "Effective bounce projected into instantaneous swing path direction.",
        availability: result.metrics.pathProjectedEffectiveBounceDegAtBall !== null ? "available" : "undefined",
      },
      {
        key: "reference_aoa",
        label: "Reference-Point AoA",
        value: metric(result.metrics.referenceAoaDegAtBall, 1, "°", 2),
        units: "°",
        equation: "atan2(v_ref · up, |v_ref,horizontal|)",
        frame: result.frameId,
        assumptions: "Attack angle of the head reference datum.",
        availability: result.metrics.referenceAoaDegAtBall !== null ? "available" : "undefined",
      },
      {
        key: "bounce_utilization_margin",
        label: "Bounce-Utilization Margin",
        value: metric(result.metrics.bounceUtilizationMarginDeg, 1, "°", 2),
        units: "°",
        equation: "effective_bounce + reference_aoa",
        frame: result.frameId,
        assumptions: "Angle margin preventing leading-edge digging before sole skid.",
        availability: result.metrics.bounceUtilizationMarginDeg !== null ? "available" : "undefined",
      },
    ];
    if (!delivery) return clearanceExplainers;
    const seen = new Set<string>();
    const cards: MetricExplainerTs[] = [];
    for (const card of [...clearanceExplainers, ...delivery.explainers]) {
      if (!seen.has(card.key)) {
        seen.add(card.key);
        cards.push(card);
      }
    }
    return cards;
  }, [result, delivery]);

  if (result === null) return null;

  const contact = result.firstGroundContact;
  const wf = delivery?.waterfall;

  return (
    <aside
      aria-label="Wedge Ground-Clearance Engineering Readout"
      className="mb-3 rounded-lg border border-emerald-400/30 bg-emerald-950/10 p-3"
    >
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="font-semibold text-emerald-200">
            Wedge Ground-Clearance & Delivery Kinematics
          </h3>
          <p className="text-xs text-slate-400">
            Swept rigid-head geometry · app frame: x target, y up, z right
          </p>
        </div>
        <span
          role="status"
          aria-label="Wedge contact sequence"
          className={`rounded-full border px-3 py-1 text-xs font-semibold ${sequenceColors[result.sequence]}`}
        >
          {sequenceLabels[result.sequence]}
        </span>
      </div>

      <div
        aria-label="Contact ordering"
        className="mb-3 flex items-center gap-2 rounded border border-slate-700/70 bg-slate-950/60 px-3 py-2 text-xs"
      >
        <span className="font-semibold text-cyan-200">
          Ball {result.ballContactTimeS === null ? "Missed" : `${result.ballContactTimeS.toFixed(3)} s`}
        </span>
        <span aria-hidden="true" className="h-px flex-1 bg-gradient-to-r from-cyan-400 via-slate-500 to-emerald-400" />
        <span className="text-right font-semibold text-emerald-200">
          {contact === null
            ? "No Ground Contact"
            : `${titleCaseFeature(contact.feature)} ${contact.timeS.toFixed(3)} s`}
        </span>
      </div>

      {/* Synchronized Metric Cards with Interactive Explainers */}
      <div className="grid gap-2 text-sm sm:grid-cols-2 xl:grid-cols-4">
        {allCards.map((entry) => (
          <details
            key={entry.key}
            className="group rounded border border-slate-700/70 bg-slate-900/60 p-2 transition-colors open:border-emerald-400/50 hover:border-slate-500"
          >
            <summary className="cursor-pointer list-none focus-visible:outline focus-visible:outline-2 focus-visible:outline-emerald-400">
              <div className="flex items-center justify-between gap-2 text-xs text-slate-400">
                <span>{entry.label}</span>
                <span aria-hidden="true" className="text-emerald-400 transition-transform group-open:rotate-90">
                  ›
                </span>
              </div>
              <p className="font-mono text-slate-100">
                {entry.value !== null && typeof entry.value === "number"
                  ? `${entry.value.toFixed(2)} ${entry.units}`
                  : String(entry.value ?? "Unavailable")}
              </p>
              <span className="text-[10px] font-medium uppercase tracking-wide text-emerald-400/80">
                Click for Definition
              </span>
            </summary>
            <div className="mt-2 border-t border-slate-700 pt-2 text-xs leading-relaxed text-slate-300">
              <p>
                <b>Equation:</b> <code>{entry.equation}</code>
              </p>
              <p className="mt-1">
                <b>Frame:</b> {entry.frame}
              </p>
              <p className="mt-1">
                <b>Assumptions:</b> {entry.assumptions}
              </p>
            </div>
          </details>
        ))}
      </div>

      {/* Linear-Velocity Contribution Waterfall */}
      {wf && (
        <div
          aria-label="Linear-Velocity Contribution Waterfall"
          className="mt-3 rounded border border-emerald-500/30 bg-slate-950/70 p-3"
        >
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <h4 className="text-xs font-semibold uppercase tracking-wider text-emerald-300">
              Linear-Velocity Contribution Waterfall
            </h4>
            <span className="text-[11px] text-slate-400">
              v_contact = v_axis + v_shaft + v_other
            </span>
          </div>

          <p className="mt-1 text-xs text-slate-400">
            Linear velocity components are strictly additive in 3D Euclidean space. Angles of attack are nonlinear (<code>atan2</code>) and reported below as non-additive counterfactual deltas and Shapley values; never implied as additive Euler angles.
          </p>

          <div className="mt-3 overflow-x-auto">
            <table className="w-full text-left text-xs text-slate-300" role="table">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400">
                  <th scope="col" className="py-1">Contribution Term</th>
                  <th scope="col" className="py-1 text-right">Downrange (X)</th>
                  <th scope="col" className="py-1 text-right">Vertical (Y)</th>
                  <th scope="col" className="py-1 text-right">Lateral (Z)</th>
                  <th scope="col" className="py-1 text-right">Speed</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-mono">
                <tr>
                  <td className="py-1.5 font-sans font-medium text-emerald-200">1. Shaft-Axis Translation (v_axis)</td>
                  <td className="py-1.5 text-right">{wf.baseAxis.downrangeMps.toFixed(2)} m/s</td>
                  <td className="py-1.5 text-right">{wf.baseAxis.verticalMps.toFixed(2)} m/s</td>
                  <td className="py-1.5 text-right">{wf.baseAxis.lateralMps.toFixed(2)} m/s</td>
                  <td className="py-1.5 text-right">{wf.baseAxis.totalSpeedMps.toFixed(2)} m/s</td>
                </tr>
                <tr>
                  <td className="py-1.5 font-sans font-medium text-cyan-200">+ 2. Shaft-Rotation Velocity (v_shaft)</td>
                  <td className="py-1.5 text-right">{wf.shaftRotation.downrangeMps.toFixed(2)} m/s</td>
                  <td className="py-1.5 text-right">{wf.shaftRotation.verticalMps.toFixed(2)} m/s</td>
                  <td className="py-1.5 text-right">{wf.shaftRotation.lateralMps.toFixed(2)} m/s</td>
                  <td className="py-1.5 text-right">{wf.shaftRotation.totalSpeedMps.toFixed(2)} m/s</td>
                </tr>
                <tr>
                  <td className="py-1.5 font-sans font-medium text-amber-200">+ 3. Other-Rotation Velocity (v_other)</td>
                  <td className="py-1.5 text-right">{wf.otherRotation.downrangeMps.toFixed(2)} m/s</td>
                  <td className="py-1.5 text-right">{wf.otherRotation.verticalMps.toFixed(2)} m/s</td>
                  <td className="py-1.5 text-right">{wf.otherRotation.lateralMps.toFixed(2)} m/s</td>
                  <td className="py-1.5 text-right">{wf.otherRotation.totalSpeedMps.toFixed(2)} m/s</td>
                </tr>
                <tr className="border-t border-emerald-500/40 bg-emerald-950/20 font-semibold text-emerald-100">
                  <td className="py-1.5 font-sans">= Total Contact Velocity (v_contact)</td>
                  <td className="py-1.5 text-right">{wf.totalContact.downrangeMps.toFixed(2)} m/s</td>
                  <td className="py-1.5 text-right">{wf.totalContact.verticalMps.toFixed(2)} m/s</td>
                  <td className="py-1.5 text-right">{wf.totalContact.lateralMps.toFixed(2)} m/s</td>
                  <td className="py-1.5 text-right">{wf.totalContact.totalSpeedMps.toFixed(2)} m/s</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="mt-2.5 flex flex-wrap gap-2 text-[11px] text-slate-300">
            <span className="rounded bg-slate-900 px-2 py-0.5 border border-slate-700/60">
              Total AoA: <b>{wf.totalAoaDeg !== null ? `${wf.totalAoaDeg.toFixed(2)}°` : "N/A"}</b>
            </span>
            <span className="rounded bg-slate-900 px-2 py-0.5 border border-slate-700/60">
              Without Shaft AoA: <b>{wf.withoutShaftAoaDeg !== null ? `${wf.withoutShaftAoaDeg.toFixed(2)}°` : "N/A"}</b>
            </span>
            <span className="rounded bg-slate-900 px-2 py-0.5 border border-slate-700/60">
              Shaft AoA Δ: <b>{wf.shaftCounterfactualAoaDeltaDeg !== null ? `${wf.shaftCounterfactualAoaDeltaDeg.toFixed(2)}°` : "N/A"}</b>
            </span>
            <span className="rounded bg-slate-900 px-2 py-0.5 border border-slate-700/60">
              Shaft Shapley AoA: <b>{wf.shaftShapleyAoaDeg !== null ? `${wf.shaftShapleyAoaDeg.toFixed(2)}°` : "N/A"}</b>
            </span>
            <span className="rounded bg-slate-900 px-2 py-0.5 border border-slate-700/60">
              Other Shapley AoA: <b>{wf.otherShapleyAoaDeg !== null ? `${wf.otherShapleyAoaDeg.toFixed(2)}°` : "N/A"}</b>
            </span>
          </div>
        </div>
      )}

      <p className="mt-2 text-xs leading-relaxed text-slate-400">
        <b className="text-slate-300">Geometry Basis:</b> {titleCaseFeature(result.geometryBasis)}. {result.provenance}{" "}
        <b className="text-slate-300">Model Boundary:</b> {result.limitations}
      </p>
    </aside>
  );
}
