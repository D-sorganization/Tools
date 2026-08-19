import { type ChangeEvent, type ReactNode, useRef, useState } from "react";

import type { GroundRegionalMaterialPlanRequest } from "../model/groundRegionalPlan";
import { downloadRegionalExecutionEvidence } from "../model/regionalExecutionFiles";
import {
  readRegionalExecutionEvidenceFile,
  type RegionalExecutionEvidence,
} from "../model/regionalExecutionReadback";
import { RegionalExecutionLedgerTables } from "./RegionalExecutionLedgerTables";

const metric = (value: number | null): string =>
  value === null ? "Unavailable" : `${value.toFixed(3)} m`;
const seconds = (value: number | null): string =>
  value === null ? "Unavailable" : `${value.toFixed(3)} s`;

function ReadbackItem(props: { readonly label: string; readonly children: ReactNode }) {
  return <div><dt className="text-slate-500">{props.label}</dt><dd>{props.children}</dd></div>;
}

export function RegionalExecutionEvidencePanel(props: {
  readonly currentPlan: () => GroundRegionalMaterialPlanRequest;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [evidence, setEvidence] = useState<RegionalExecutionEvidence | null>(null);
  const readback = evidence?.readback ?? null;
  const [status, setStatus] = useState("No execution evidence loaded.");
  const [error, setError] = useState<string | null>(null);
  const importEvidence = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.currentTarget.files?.[0];
    event.currentTarget.value = "";
    if (file === undefined) return;
    try {
      const loaded = await readRegionalExecutionEvidenceFile(file, props.currentPlan());
      setEvidence(loaded);
      setError(null);
      setStatus(`Loaded ${file.name}; no browser physics executed.`);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Evidence import failed");
      setStatus("Import failed; prior accepted execution evidence was preserved.");
    }
  };
  const downloadEvidence = () => {
    if (evidence === null) return;
    try {
      downloadRegionalExecutionEvidence(evidence.result);
      setError(null);
      setStatus("Downloaded canonical evidence; no browser physics executed.");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Evidence download failed");
      setStatus("Download failed; prior accepted execution evidence was preserved.");
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
      <button type="button" disabled={evidence === null} onClick={downloadEvidence}
        aria-label="Download canonical execution evidence JSON"
        className="ml-2 mt-3 rounded-md border border-sky-500/60 px-3 py-2 text-sm text-sky-200 disabled:opacity-50">
        Download canonical evidence
      </button>
      {error !== null && <p role="alert" className="mt-3 text-sm text-rose-200">{error}</p>}
      <p role="status" aria-label="Regional execution evidence status"
        className="mt-3 text-xs text-slate-400">{status}</p>
      {readback !== null && <dl aria-label="Regional execution evidence readback"
        className="mt-3 grid gap-2 text-sm sm:grid-cols-2 xl:grid-cols-4">
        <ReadbackItem label="Status">{readback.status}</ReadbackItem>
        <ReadbackItem label="Termination">
          {readback.terminationReason ?? readback.failureReason ?? "Unavailable"}
        </ReadbackItem>
        <ReadbackItem label="Ground time">{seconds(readback.groundTimeS)}</ReadbackItem>
        <ReadbackItem label="Terminal completion">
          {readback.completed === null ? "Unavailable" : readback.completed
            ? "Completed" : "Observed endpoint"}
        </ReadbackItem>
        <ReadbackItem label="Plan / surface">
          {readback.planId} / {readback.surfaceId}
        </ReadbackItem>
        <ReadbackItem label="Surface provider">
          {readback.surfaceProviderId} {readback.surfaceProviderVersion}
        </ReadbackItem>
        <ReadbackItem label="Model">
          {readback.modelId} {readback.modelVersion}
        </ReadbackItem>
        <ReadbackItem label="Units">{readback.unitSystem}</ReadbackItem>
        <ReadbackItem label="Carry">{metric(readback.carryDistanceM)}</ReadbackItem>
        <ReadbackItem label="Bounce air">{metric(readback.bounceAirDistanceM)}</ReadbackItem>
        <ReadbackItem label="Skid">{metric(readback.skidDistanceM)}</ReadbackItem>
        <ReadbackItem label="Roll">{metric(readback.rollDistanceM)}</ReadbackItem>
        <ReadbackItem label="Surface path">{metric(readback.surfacePathDistanceM)}</ReadbackItem>
        <ReadbackItem label="Total">{metric(readback.totalDistanceM)}</ReadbackItem>
        <ReadbackItem label="Final downrange">{metric(readback.finalDownrangeM)}</ReadbackItem>
        <ReadbackItem label="Final offline">{metric(readback.finalOfflineM)}</ReadbackItem>
        <ReadbackItem label="Bounces">{readback.bounceCount ?? "Unavailable"}</ReadbackItem>
        <ReadbackItem label="Surface transitions">{readback.transitionCount}</ReadbackItem>
        <ReadbackItem label="Calibration">
          {readback.calibrationKind === null ? "Unavailable"
            : `${readback.calibrationKind} · ${readback.calibrationId} · ` +
              `${readback.calibrationSource} · confidence ` +
              `${readback.calibrationConfidence}`}
        </ReadbackItem>
        <ReadbackItem label="Observed phases">
          {readback.observedPhases.length === 0
            ? "Unavailable" : readback.observedPhases.join(" → ")}
        </ReadbackItem>
        <div className="sm:col-span-2 xl:col-span-4">
          <dt className="text-slate-500">Executor provenance</dt>
          <dd className="break-all">{readback.executorSourceRevision} · {readback.executorInputSha256}</dd>
        </div>
        <div className="sm:col-span-2 xl:col-span-4">
          <dt className="text-slate-500">Qualification limits</dt>
          <dd>{readback.limitations.join(" · ")}</dd>
        </div>
        {readback.warnings.length > 0 && <div className="sm:col-span-2 xl:col-span-4">
          <dt className="text-slate-500">Warnings</dt>
          <dd><ul className="list-disc space-y-1 pl-5">
            {readback.warnings.map((warning, index) => <li key={`${warning.code}-${index}`}>
              {warning.code} [{warning.severity}] — {warning.message}
            </li>)}
          </ul></dd>
        </div>}
      </dl>}
      {readback !== null && <RegionalExecutionLedgerTables
        events={readback.events}
        trajectory={evidence?.result.ground_result?.trajectory ?? []}
        transitions={readback.transitions} />}
    </section>
  );
}
