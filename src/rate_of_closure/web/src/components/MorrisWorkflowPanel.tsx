import { useMemo, useState } from "react";

import { useMorrisAuthority } from "../hooks/useMorrisAuthority";
import type { MorrisAuthorityClient } from "../model/morrisAuthorityClient";
import {
  suggestedMorrisFactorDrafts,
  type MorrisAuthorityBase,
  type MorrisAuthorityRequest,
  type MorrisFactorDraft,
} from "../model/morrisAuthorityRequest";
import { presentMorrisJob } from "../model/morrisPresentation";
import { BUTTON_CLASS, INPUT_CLASS, PANEL_CLASS } from "./variationUi";
import { MorrisFactorEditor } from "./MorrisFactorEditor";
import { MorrisResults } from "./MorrisResults";
import { DecimalInput } from "./DecimalInput";

interface MorrisWorkflowPanelProps {
  readonly client: MorrisAuthorityClient | null;
  readonly base: MorrisAuthorityBase;
  readonly pollIntervalMs?: number;
}

interface DesignControls {
  readonly trajectories: number;
  readonly levels: number;
  readonly seed: number;
  readonly minimumEffects: number;
  readonly workerCount: number;
}

const DEFAULT_DESIGN: DesignControls = Object.freeze({
  trajectories: 12, levels: 4, seed: 73, minimumEffects: 2, workerCount: 1,
});

const requestId = (): string => {
  const cryptoId = globalThis.crypto?.randomUUID?.();
  return cryptoId === undefined ? `morris-${Date.now()}` : `morris-${cryptoId}`;
};

export function MorrisWorkflowPanel(props: MorrisWorkflowPanelProps) {
  const [drafts, setDrafts] = useState<readonly MorrisFactorDraft[]>(() => suggestedMorrisFactorDrafts(props.base));
  const [design, setDesign] = useState<DesignControls>(DEFAULT_DESIGN);
  const workflow = useMorrisAuthority(props.client, props.pollIntervalMs);
  const jobView = useMemo(() => workflow.state.job === null ? null : presentMorrisJob(workflow.state.job), [workflow.state.job]);
  const busy = workflow.state.submitting || (jobView !== null && !jobView.terminal);
  const available = workflow.state.capability?.available === true;
  const status = props.client === null
    ? "Morris authority is not connected; this static client has no browser physics fallback."
    : workflow.state.checking ? "Checking Morris authority capability…"
      : workflow.state.error ? `Morris authority error: ${workflow.state.error}`
        : !available ? "Morris authority unavailable; screening cannot run in this deployment."
          : jobView?.errorMessage
            ? `${jobView.message}: ${jobView.errorMessage}${jobView.errorCode ? ` (${jobView.errorCode})` : ""}`
            : jobView?.message ?? "Morris authority available. Configure an elementary-effects screening study.";

  const updateDesign = (field: keyof DesignControls, value: number) => {
    if (Object.is(design[field], value)) return;
    workflow.invalidate();
    setDesign((current) => ({ ...current, [field]: value }));
  };
  const updateDrafts = (next: readonly MorrisFactorDraft[]) => {
    const unchanged = drafts.length === next.length && drafts.every((draft, index) => (
      draft.variableKey === next[index]?.variableKey
      && draft.enabled === next[index]?.enabled
      && Object.is(draft.lower, next[index]?.lower)
      && Object.is(draft.upper, next[index]?.upper)
    ));
    if (unchanged) return;
    workflow.invalidate();
    setDrafts(next);
  };
  const run = () => {
    const request: MorrisAuthorityRequest = {
      requestId: requestId(), base: props.base, factors: drafts,
      trajectories: design.trajectories, levels: design.levels, seed: design.seed,
      minimumEffects: design.minimumEffects, workerCount: design.workerCount,
    };
    void workflow.run(request);
  };
  return (
    <div className="space-y-5">
      <section aria-label="Morris screening setup" className={`${PANEL_CLASS} space-y-4`}>
        <div><h2 className="text-xl font-semibold">Morris Elementary-Effects Screening</h2>
          <p className="mt-1 max-w-4xl text-sm text-slate-400">Screen several bounded model inputs against every authority output.
            μ* ranks overall influence; σ flags nonlinearity or interaction. All simulations run in the injected local Python authority.</p></div>
        <details className="rounded border border-slate-800 bg-slate-950/40 p-3 text-xs text-slate-300">
          <summary className="cursor-pointer font-medium text-slate-200">Authority base used by this study</summary>
          <p aria-label="Morris authority base" className="mt-2 leading-5">
            {props.base.clubName}; {props.base.supportMode}
            {props.base.supportMode === "tee" ? ` at ${props.base.teeHeightM} m` : ""}; plane
            ({props.base.planeYawDeg}, {props.base.planeSideTiltDeg}, {props.base.planeForwardTiltDeg}) deg;
            damping ({props.base.dampingShoulder}, {props.base.dampingWrist}) N·m·s; pendulum segment 1
            ({props.base.pendulumM1Kg} kg, {props.base.pendulumL1M}/{props.base.pendulumLc1M} m,
            {props.base.pendulumI1KgM2} kg·m²), segment 2 ({props.base.pendulumM2Kg} kg,
            {props.base.pendulumL2M}/{props.base.pendulumLc2M} m, {props.base.pendulumI2KgM2} kg·m²);
            {props.base.swingDurationS} s swing; {props.base.flightModel} flight; impact offsets
            ({props.base.impactOffsetToeMm}, {props.base.impactOffsetHighMm}) mm. Unswept values remain fixed.
          </p>
        </details>
        <p role="status" aria-live="polite" className={`rounded border p-3 text-sm ${workflow.state.error || !available
          ? "border-amber-500/40 bg-amber-950/20 text-amber-100"
          : "border-emerald-500/30 bg-emerald-950/20 text-emerald-100"}`}>{status}</p>
        {jobView && <div aria-label="Morris progress" className="space-y-1">
          <progress className="h-2 w-full" max={jobView.totalSamples} value={jobView.completedSamples} />
          <p className="text-xs text-slate-400">{jobView.completedSamples} of {jobView.totalSamples} evaluations complete.</p>
        </div>}
        <fieldset disabled={busy} className="space-y-3">
          <legend className="font-semibold text-slate-200">Design controls</legend>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
            {([
              ["trajectories", "Trajectories", 1, undefined], ["levels", "Levels (even)", 4, 2],
              ["seed", "Random seed", 0, undefined], ["minimumEffects", "Minimum effects", 2, undefined],
              ["workerCount", "Workers", 1, undefined],
            ] as const).map(([field, label, min, step]) => <label key={field} className="text-xs text-slate-300">{label}
              <DecimalInput className={`${INPUT_CLASS} mt-1`} min={min} step={step}
                title={`${label} for the Morris elementary-effects design`}
                value={design[field]} onCommit={(value) => updateDesign(field, value)} /></label>)}
          </div>
          <MorrisFactorEditor drafts={drafts} supportMode={props.base.supportMode}
            disabled={busy} onChange={updateDrafts} />
        </fieldset>
        <div className="flex flex-wrap gap-2">
          <button type="button" className={BUTTON_CLASS} title="Submit this validated design to the local Python authority"
            disabled={!available || busy} onClick={run}>Run Morris Screening</button>
          <button type="button" className={BUTTON_CLASS} disabled={!jobView?.canCancel}
            title="Request cancellation of the active authority job"
            onClick={() => void workflow.cancel()}>Cancel Morris Screening</button>
        </div>
      </section>
      {workflow.state.job?.report && <MorrisResults report={workflow.state.job.report} />}
    </div>
  );
}
