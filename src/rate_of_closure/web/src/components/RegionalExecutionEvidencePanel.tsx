import { type ChangeEvent, useRef, useState } from "react";

import type { GroundRegionalMaterialPlanRequest } from "../model/groundRegionalPlan";
import {
  readRegionalExecutionEvidenceFile,
  type RegionalExecutionReadback,
} from "../model/regionalExecutionReadback";

const metric = (value: number | null): string =>
  value === null ? "Unavailable" : `${value.toFixed(3)} m`;

export function RegionalExecutionEvidencePanel(props: {
  readonly currentPlan: () => GroundRegionalMaterialPlanRequest;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [readback, setReadback] = useState<RegionalExecutionReadback | null>(null);
  const [status, setStatus] = useState("No execution evidence loaded.");
  const [error, setError] = useState<string | null>(null);
  const importEvidence = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.currentTarget.files?.[0];
    event.currentTarget.value = "";
    if (file === undefined) return;
    try {
      const loaded = await readRegionalExecutionEvidenceFile(file, props.currentPlan());
      setReadback(loaded.readback);
      setError(null);
      setStatus(`Loaded ${file.name}; no browser physics executed.`);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Evidence import failed");
      setStatus("Import failed; prior accepted execution evidence was preserved.");
    }
  };
  return (
    <section aria-labelledby="regional-execution-evidence-title"
      className="rounded-xl border border-slate-700/80 bg-slate-900/60 p-4">
      <h3 id="regional-execution-evidence-title" className="font-semibold text-slate-200">
        Regional execution evidence
      </h3>
      <p className="mt-1 text-xs text-slate-400">
        Import a canonical Python-produced execution result for this exact plan.
        This browser readback does not execute, approximate, or modify physics.
      </p>
      <input ref={input} type="file" accept=".json,application/json" className="sr-only"
        aria-label="Import regional execution evidence JSON"
        onChange={(event) => { void importEvidence(event); }} />
      <button type="button" onClick={() => input.current?.click()}
        className="mt-3 rounded-md border border-sky-500/60 px-3 py-2 text-sm text-sky-200">
        Import execution evidence
      </button>
      {error !== null && <p role="alert" className="mt-3 text-sm text-rose-200">{error}</p>}
      <p role="status" aria-label="Regional execution evidence status"
        className="mt-3 text-xs text-slate-400">{status}</p>
      {readback !== null && <dl aria-label="Regional execution evidence readback"
        className="mt-3 grid gap-2 text-sm sm:grid-cols-2 xl:grid-cols-4">
        <div><dt className="text-slate-500">Status</dt><dd>{readback.status}</dd></div>
        <div><dt className="text-slate-500">Termination</dt><dd>{readback.terminationReason ?? readback.failureReason ?? "Unavailable"}</dd></div>
        <div><dt className="text-slate-500">Plan / surface</dt><dd>{readback.planId} / {readback.surfaceId}</dd></div>
        <div><dt className="text-slate-500">Model</dt><dd>{readback.modelId} {readback.modelVersion}</dd></div>
        <div><dt className="text-slate-500">Skid</dt><dd>{metric(readback.skidDistanceM)}</dd></div>
        <div><dt className="text-slate-500">Roll</dt><dd>{metric(readback.rollDistanceM)}</dd></div>
        <div><dt className="text-slate-500">Total</dt><dd>{metric(readback.totalDistanceM)}</dd></div>
        <div><dt className="text-slate-500">Surface transitions</dt><dd>{readback.transitionCount}</dd></div>
        <div className="sm:col-span-2 xl:col-span-4">
          <dt className="text-slate-500">Executor provenance</dt>
          <dd className="break-all">{readback.executorSourceRevision} · {readback.executorInputSha256}</dd>
        </div>
        <div className="sm:col-span-2 xl:col-span-4">
          <dt className="text-slate-500">Qualification limits</dt>
          <dd>{readback.limitations.join(" · ")}</dd>
        </div>
      </dl>}
    </section>
  );
}
