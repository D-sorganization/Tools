import { useEffect, useMemo, useRef, useState } from "react";

import {
  buildLocalizedPairedRequest, localizedPairedPlan,
} from "../model/localizedAttributionExecution";
import type { LocalizedPairedExecutionServiceTs } from "../model/localizedAttributionExecutionService";
import type { VariationPlanTs } from "../model/variation";
import { stableSpecId } from "../model/variationSchema";
import { BUTTON_CLASS, INPUT_CLASS, PANEL_CLASS } from "./variationUi";
import { useLocalizedPairedExecution } from "./useLocalizedPairedExecution";

interface Props {
  plan: VariationPlanTs;
  service?: LocalizedPairedExecutionServiceTs;
  onAuthorityChange: (authority: ReturnType<typeof useLocalizedPairedExecution>["authority"]) => void;
}

const capability = (plan: VariationPlanTs): { plan: VariationPlanTs | null; reason: string } => {
  try { return { plan: localizedPairedPlan(plan), reason: "" }; }
  catch (error) { return { plan: null, reason: (error as Error).message }; }
};

export function LocalizedPairedStudyPanel({ plan, service, onAuthorityChange }: Props) {
  const execution = useLocalizedPairedExecution(service);
  const available = useMemo(() => capability(plan), [plan]);
  const [open, setOpen] = useState(false);
  const [stateTime, setStateTime] = useState("0.020");
  const [deltas, setDeltas] = useState<Record<string, string>>({});
  const planIdentity = JSON.stringify(plan);
  const priorPlanIdentity = useRef(planIdentity);

  const { invalidate } = execution;
  useEffect(() => {
    if (priorPlanIdentity.current === planIdentity) return;
    priorPlanIdentity.current = planIdentity;
    invalidate("Variation plan changed; prior paired authority was cleared.");
    setOpen(false); setDeltas({});
  }, [planIdentity, invalidate]);
  useEffect(() => onAuthorityChange(execution.authority),
    [execution.authority, onAuthorityChange]);

  const configure = () => {
    if (!available.plan) return;
    setDeltas(Object.fromEntries(available.plan.noise.map((spec) => [
      stableSpecId(spec), String(spec.scale),
    ])));
    setOpen(true);
  };
  const run = () => {
    if (!available.plan) return;
    try {
      const numeric = Object.fromEntries(Object.entries(deltas).map(([key, value]) =>
        [key, Number(value)]));
      void execution.run(buildLocalizedPairedRequest(
        available.plan, numeric, Number(stateTime),
      ));
    } catch (error) {
      execution.reportFailure((error as Error).message);
    }
  };

  return <section aria-label="Separate localized paired study" className={PANEL_CLASS}>
    <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-400">
      Separate Localized Paired Study
    </h2>
    <p className="mt-2 text-xs text-amber-300">
      Explicit baseline and one-source planted interventions only. This does not reuse
      Monte Carlo scatter and does not establish causality. Global factors stay fixed at
      their declared base values; groups are excluded.
    </p>
    <div className="mt-3 flex flex-wrap gap-2">
      <button type="button" className={BUTTON_CLASS} disabled={!available.plan || execution.busy}
        title={available.reason || "Review exact source loci and configure planted deltas."}
        onClick={configure}>Configure &amp; Run Separate Paired Study…</button>
      <button type="button" className={BUTTON_CLASS} disabled={!execution.busy}
        title="Terminate the current paired Worker without replacing prior authority."
        onClick={execution.cancel}>Cancel Separate Paired Study</button>
    </div>
    {!available.plan && <p role="note" className="mt-2 text-xs text-amber-300">
      Paired study unavailable: {available.reason}
    </p>}
    {open && available.plan && <div className="mt-4 space-y-3 rounded border border-slate-700 p-3">
      {available.plan.noise.map((spec) => <div key={stableSpecId(spec)}
        className="grid min-w-0 gap-2 text-xs md:grid-cols-[minmax(0,1fr)_10rem]">
        <p className="min-w-0 break-all"><strong>{stableSpecId(spec)}</strong><br />{spec.variableKey}<br />
          {spec.pointIds?.[0]} · [{spec.timeWindowS?.[0]}, {spec.timeWindowS?.[1]}) s · N·m</p>
        <label>Planted delta (N·m)
          <input className={INPUT_CLASS} aria-label={`Planted delta ${stableSpecId(spec)}`}
            type="number" step="any" value={deltas[stableSpecId(spec)] ?? ""}
            onChange={(event) => setDeltas({ ...deltas,
              [stableSpecId(spec)]: event.target.value })} /></label>
      </div>)}
      <label className="block text-xs">Exact state sample time (s), 1 ms grid
        <input className={`${INPUT_CLASS} mt-1`} aria-label="Paired state sample time"
          type="number" min="0" max="1.5" step="0.001" value={stateTime}
          onChange={(event) => setStateTime(event.target.value)} /></label>
      <p className="text-xs text-slate-400">State target: swing.clubhead.reference in
        app_frame:x_target,y_up,z_right. Impact and shot targets use typed hit/no-impact/failure availability.</p>
      <button type="button" className={BUTTON_CLASS} disabled={execution.busy}
        onClick={run}>Confirm &amp; Run {available.plan.noise.length * 2} Explicit Trials</button>
    </div>}
    <p role="log" aria-live="polite" aria-label="Paired study status"
      className="mt-3 text-xs text-slate-300">
      {execution.status}
    </p>
    {execution.progress && <progress aria-label="Paired study progress"
      className="mt-2 w-full" value={execution.progress.completedRuns}
      max={execution.progress.totalRuns} />}
  </section>;
}
