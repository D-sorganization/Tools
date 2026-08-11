import {
  useEffect,
  useState,
  type Dispatch,
  type SetStateAction,
} from "react";

import { useCapabilityOptimization } from "../hooks/useCapabilityOptimization";
import {
  buildCapabilityWorkflow,
  capabilityWorkflowToJson,
  type CapabilityWorkflowInputs,
} from "../model/capabilityWorkflow";
import type { CapabilityRunner } from "../model/capabilityWorkerClient";
import { CapabilityResults } from "./CapabilityResults";
import { BUTTON_CLASS, INPUT_CLASS, PANEL_CLASS, downloadText } from "./variationUi";

const NUMERIC_FIELDS: ReadonlyArray<{ key: keyof CapabilityWorkflowInputs;
  label: string; unit: string; min?: number; max?: number; step?: number }> = [
  { key: "ballSpeedMps", label: "Ball speed center", unit: "m/s", min: 1, max: 100, step: 0.1 },
  { key: "ballSpeedStdMps", label: "Ball speed standard deviation", unit: "m/s", min: 0, max: 30, step: 0.1 },
  { key: "launchAngleDeg", label: "Launch angle center", unit: "deg", min: -10, max: 45, step: 0.1 },
  { key: "launchAngleStdDeg", label: "Launch angle standard deviation", unit: "deg", min: 0, max: 30, step: 0.1 },
  { key: "launchDirectionDeg", label: "Launch direction center (+ right)", unit: "deg", min: -30, max: 30, step: 0.1 },
  { key: "launchDirectionStdDeg", label: "Direction standard deviation", unit: "deg", min: 0, max: 30, step: 0.1 },
  { key: "totalSpinRpm", label: "Fixed total spin", unit: "rpm", min: 0, max: 20_000, step: 10 },
  { key: "spinAxisTiltDeg", label: "Fixed spin-axis tilt (+ fade/right)", unit: "deg", min: -90, max: 90, step: 0.1 },
  { key: "targetDistanceM", label: "Target distance", unit: "m", min: 0.1, max: 1_000, step: 1 },
  { key: "targetLateralM", label: "Target lateral (+ right)", unit: "m", min: -500, max: 500, step: 1 },
  { key: "targetRadiusM", label: "Target radius", unit: "m", min: 0.1, max: 500, step: 1 },
  { key: "maxTimeS", label: "Maximum flight time", unit: "s", min: 0.001, max: 120, step: 0.1 },
  { key: "trajectorySampleIntervalS", label: "Trajectory sample interval", unit: "s", min: 0.001, max: 0.1, step: 0.001 },
  { key: "candidateBudget", label: "Candidate budget", unit: "candidates", min: 1, max: 100_000, step: 1 },
  { key: "ensembleSize", label: "Trials per candidate", unit: "trials", min: 1, max: 100_000, step: 1 },
  { key: "alternativesCount", label: "Alternatives retained", unit: "shots", min: 1, max: 100_000, step: 1 },
  { key: "seed", label: "Deterministic seed", unit: "int31", min: 0, max: 2 ** 31 - 1, step: 1 },
];
const OBJECTIVES = ["maximize_carry", "minimize_expected_miss", "maximize_target_hold",
  "minimize_variability", "minimize_downside", "distance_control_pareto"] as const;

function NumericInput({ field, value, update }: {
  readonly field: (typeof NUMERIC_FIELDS)[number]; readonly value: number;
  readonly update: (value: number) => void;
}) {
  const [draft, setDraft] = useState(String(value));
  useEffect(() => setDraft(String(value)), [value]);
  const commit = (source: string): void => {
    const parsed = Number(source);
    if (source.trim() && Number.isFinite(parsed)) update(parsed);
    else setDraft(String(value));
  };
  return <input type="text" inputMode="decimal" aria-label={field.label}
    className={INPUT_CLASS} value={draft}
    onFocus={(event) => event.currentTarget.select()}
    onBlur={(event) => commit(event.currentTarget.value)}
    onChange={(event) => {
      const next = event.target.value; setDraft(next); const parsed = Number(next);
      if (next.trim() && !/[eE.+-]$/.test(next) && Number.isFinite(parsed)) update(parsed);
    }} />;
}

function InputsForm({ inputs, update }: { readonly inputs: CapabilityWorkflowInputs;
  readonly update: (key: keyof CapabilityWorkflowInputs, value: string | number) => void }) {
  return <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
    <label className="text-xs text-slate-300">Profile ID<input className={INPUT_CLASS}
      value={inputs.profileId} onChange={(event) => update("profileId", event.target.value)} /></label>
    <label className="text-xs text-slate-300">Club ID<input className={INPUT_CLASS}
      value={inputs.clubId} onChange={(event) => update("clubId", event.target.value)} /></label>
    <label className="text-xs text-slate-300">Objective<select className={INPUT_CLASS}
      value={inputs.objective} onChange={(event) => update("objective", event.target.value)}>{
        OBJECTIVES.map((value) => <option key={value}>{value}</option>)}</select></label>
    {NUMERIC_FIELDS.map((field) => <label key={field.key} className="text-xs text-slate-300">
      <span className="flex justify-between"><span>{field.label}</span><span>{field.unit}</span></span>
      <NumericInput field={field} value={inputs[field.key] as number}
        update={(value) => update(field.key, value)} /></label>)}
  </div>;
}

function WorkflowActions({ state }: {
  readonly state: ReturnType<typeof useCapabilityOptimization>;
}) {
  const [saveError, setSaveError] = useState<string | null>(null);
  const save = (): void => {
    try { downloadText("capability-workflow.json", capabilityWorkflowToJson(
      buildCapabilityWorkflow(state.inputs)), "application/json"); setSaveError(null); }
    catch (reason: unknown) { setSaveError(reason instanceof Error ? reason.message : String(reason)); }
  };
  return <><div className="mt-4 flex flex-wrap items-center gap-2">
    <button className={BUTTON_CLASS} type="button" onClick={state.run}
      disabled={state.running}>Run optimization</button>
    <button className={BUTTON_CLASS} type="button" disabled={!state.running}
      onClick={state.cancel}>Cancel</button>
    <button className={BUTTON_CLASS} type="button" onClick={save}>Save workflow</button>
    <label className={BUTTON_CLASS}>Load workflow<input className="sr-only" type="file"
      accept="application/json,.json" onChange={(event) => {
        const file = event.target.files?.[0]; if (file) void state.load(file); }} /></label>
    <span role="status" className="text-xs text-slate-300">{state.status}{state.progress.total > 0
      ? ` — ${state.progress.completed}/${state.progress.total}` : ""}</span>
  </div>{(state.error || saveError) && <p role="alert"
    className="mt-3 text-xs text-rose-400">{state.error ?? saveError}</p>}</>;
}

export function CapabilityOptimizationPanel({ runner, inputs, onInputsChange }: {
  readonly runner?: CapabilityRunner;
  readonly inputs?: CapabilityWorkflowInputs;
  readonly onInputsChange?: Dispatch<SetStateAction<CapabilityWorkflowInputs>>;
}): JSX.Element {
  if ((inputs === undefined) !== (onInputsChange === undefined)) {
    throw new TypeError("controlled capability inputs require an input change handler");
  }
  const authority = inputs === undefined || onInputsChange === undefined
    ? undefined : { inputs, onInputsChange };
  const state = useCapabilityOptimization(runner, authority);
  return <section className={PANEL_CLASS} aria-label="Shot capability optimizer">
    <h2 className="text-lg font-semibold text-sky-300">Shot Capability Optimizer</h2>
    <p className="mt-1 text-xs text-slate-400">Still-air carry-to-first-ground-crossing model. Fixed spin is explicit and sourced; wind, bounce, roll, and total distance are not included.</p>
    <InputsForm inputs={state.inputs} update={state.update} />
    <WorkflowActions state={state} />
    {state.output && <CapabilityResults output={state.output} />}
  </section>;
}
