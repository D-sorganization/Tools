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
import {
  createMorrisWorkspaceDocument,
  createMorrisWorkspaceFromSetup,
  INVALID_MORRIS_BOUNDS_MESSAGE,
  MAX_MORRIS_EDITOR_BOUND,
  morrisWorkspaceMatchesBase,
  parseMorrisWorkspaceJson,
  workspaceDraftsForEditor,
  type MorrisDesignControls,
  type MorrisWorkspaceSetup,
} from "../model/morrisWorkspaceDocument";
import { BUTTON_CLASS, INPUT_CLASS, PANEL_CLASS } from "./variationUi";
import { MorrisFactorEditor } from "./MorrisFactorEditor";
import { MorrisResults } from "./MorrisResults";
import { DecimalInput } from "./DecimalInput";
import { MorrisWorkspaceActions } from "./MorrisWorkspaceActions";

interface MorrisWorkflowPanelProps {
  readonly client: MorrisAuthorityClient | null;
  readonly base: MorrisAuthorityBase;
  readonly pollIntervalMs?: number;
}

const DEFAULT_DESIGN: MorrisDesignControls = Object.freeze({
  trajectories: 12, levels: 4, seed: 73, minimumEffects: 2, workerCount: 1,
});

const requestId = (): string => {
  const cryptoId = globalThis.crypto?.randomUUID?.();
  return cryptoId === undefined ? `morris-${Date.now()}` : `morris-${cryptoId}`;
};

export function MorrisWorkflowPanel(props: MorrisWorkflowPanelProps) {
  const [drafts, setDrafts] = useState<readonly MorrisFactorDraft[]>(() => suggestedMorrisFactorDrafts(props.base));
  const [design, setDesign] = useState<MorrisDesignControls>(DEFAULT_DESIGN);
  const [importedSetup, setImportedSetup] = useState<MorrisWorkspaceSetup | null>(null);
  const [workspaceMessage, setWorkspaceMessage] = useState<string | null>(null);
  const workflow = useMorrisAuthority(props.client, props.pollIntervalMs);
  const jobView = useMemo(() => workflow.state.job === null ? null : presentMorrisJob(workflow.state.job), [workflow.state.job]);
  const busy = workflow.state.submitting || (jobView !== null && !jobView.terminal);
  const available = workflow.state.capability?.available === true;
  const evidence = workflow.state.job?.status === "completed"
    && workflow.state.submittedRequest !== null
    ? Object.freeze({ request: workflow.state.submittedRequest, job: workflow.state.job })
    : null;
  const workspace = useMemo(() => {
    try {
      return importedSetup === null
        ? createMorrisWorkspaceDocument(props.base, drafts, design, evidence)
        : createMorrisWorkspaceFromSetup(importedSetup, evidence);
    } catch {
      return null;
    }
  }, [design, drafts, evidence, importedSetup, props.base]);
  const archivedInvalidDrafts = importedSetup?.factorDrafts.filter(
    (draft) => draft.validationError !== null,
  ) ?? [];
  const status = props.client === null
    ? "Morris authority is not connected; this static client has no browser physics fallback."
    : workflow.state.checking ? "Checking Morris authority capability…"
      : workflow.state.error ? `Morris authority error: ${workflow.state.error}`
        : !available ? "Morris authority unavailable; screening cannot run in this deployment."
          : jobView?.errorMessage
            ? `${jobView.message}: ${jobView.errorMessage}${jobView.errorCode ? ` (${jobView.errorCode})` : ""}`
            : jobView?.message ?? "Morris authority available. Configure an elementary-effects screening study.";

  const updateDesign = (field: keyof MorrisDesignControls, value: number) => {
    if (Object.is(design[field], value)) return;
    workflow.invalidate();
    setImportedSetup((current) => current === null ? null : Object.freeze({
      ...current, [field]: value,
    }));
    setWorkspaceMessage(null);
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
    setImportedSetup((current) => current === null ? null : Object.freeze({
      ...current,
      factorDrafts: Object.freeze(current.factorDrafts.map((archived) => {
        const index = drafts.findIndex((draft) => draft.variableKey === archived.variableKey);
        const previous = drafts[index]; const replacement = next[index];
        if (previous === undefined || replacement === undefined
            || (previous.enabled === replacement.enabled
              && Object.is(previous.lower, replacement.lower)
              && Object.is(previous.upper, replacement.upper))) return archived;
        const valid = replacement.lower !== null && replacement.upper !== null
          && Number.isFinite(replacement.lower) && Number.isFinite(replacement.upper)
          && Math.abs(replacement.lower) <= MAX_MORRIS_EDITOR_BOUND
          && Math.abs(replacement.upper) <= MAX_MORRIS_EDITOR_BOUND
          && replacement.lower < replacement.upper;
        return Object.freeze({
          variableKey: replacement.variableKey,
          enabled: replacement.enabled,
          lower: String(replacement.lower ?? ""),
          upper: String(replacement.upper ?? ""),
          validationError: replacement.enabled || valid ? null : INVALID_MORRIS_BOUNDS_MESSAGE,
        });
      })),
    }));
    setWorkspaceMessage(null);
    setDrafts(next);
  };
  const run = () => {
    setWorkspaceMessage(null);
    const request: MorrisAuthorityRequest = {
      requestId: requestId(), base: props.base, factors: drafts,
      trajectories: design.trajectories, levels: design.levels, seed: design.seed,
      minimumEffects: design.minimumEffects, workerCount: design.workerCount,
    };
    void workflow.run(request);
  };
  const importWorkspace = (source: string) => {
    try {
      const imported = parseMorrisWorkspaceJson(source);
      if (!morrisWorkspaceMatchesBase(imported, props.base)) {
        throw new RangeError("Imported workspace authority base does not match the current simulation.");
      }
      const nextDesign: MorrisDesignControls = {
        trajectories: imported.setup.trajectories,
        levels: imported.setup.levels,
        seed: imported.setup.seed,
        minimumEffects: imported.setup.minimumEffects,
        workerCount: imported.setup.workerCount,
      };
      const nextDrafts = workspaceDraftsForEditor(imported.setup);
      workflow.installArchivedEvidence(imported.completedEvidence);
      setDesign(nextDesign);
      setDrafts(nextDrafts);
      setImportedSetup(imported.setup);
      setWorkspaceMessage(imported.completedEvidence === null
        ? "Workspace setup imported atomically. No completed evidence was present."
        : "Workspace imported atomically. Results are archived evidence and were not revalidated against a live authority.");
    } catch (error: unknown) {
      setWorkspaceMessage(`Workspace import rejected: ${error instanceof Error ? error.message : "Unknown import error."}`);
    }
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
              ["trajectories", "Trajectories", 2, 5_000, undefined], ["levels", "Levels (even)", 4, 10_000, 2],
              ["seed", "Random seed", 0, 2 ** 31 - 1, undefined], ["minimumEffects", "Minimum effects", 2, 5_000, undefined],
              ["workerCount", "Workers", 1, 32, undefined],
            ] as const).map(([field, label, min, max, step]) => <label key={field} className="text-xs text-slate-300">{label}
              <DecimalInput className={`${INPUT_CLASS} mt-1`} min={min} max={max} step={step}
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
        <MorrisWorkspaceActions workspace={workspace} busy={busy}
          onImportText={importWorkspace}
          onImportError={(message) => setWorkspaceMessage(`Workspace import failed: ${message}`)} />
        {workspace === null && <p className="text-xs text-amber-200">
          Workspace export is unavailable until all enabled bounds and design controls are valid. Import remains available.
        </p>}
        {workspaceMessage && <p role="status" aria-label="Morris workspace status"
          className="rounded border border-sky-500/30 bg-sky-950/20 p-3 text-xs text-sky-100">
          {workspaceMessage}
        </p>}
        {archivedInvalidDrafts.length > 0 && <details className="rounded border border-amber-500/30 p-3 text-xs text-amber-100">
          <summary className="cursor-pointer">Archived disabled drafts retained verbatim</summary>
          <ul className="mt-2 list-disc pl-5">{archivedInvalidDrafts.map((draft) => <li key={draft.variableKey}>
            {draft.variableKey}: lower “{draft.lower}”, upper “{draft.upper}” — {draft.validationError}
          </li>)}</ul>
        </details>}
      </section>
      {workflow.state.job?.report && <MorrisResults report={workflow.state.job.report} />}
    </div>
  );
}
