/** Strict result-only ground playback workspace. */

import { useRef, useState } from "react";

import type { FlightToGroundResult, GroundVec3 } from "../model/flightGroundTypes";
import { flightToGroundResultFromJson } from "../model/flightGroundContract";
import {
  GroundPlaybackTimeline,
  GROUND_PLAYBACK_MAX_POINTS,
  groundEvidenceWindow,
  timelineFromRegionalExecution,
} from "../model/groundPlayback";
import {
  groundRegionalExecutionResultFromJson,
  MAX_GROUND_REGIONAL_EXECUTION_WIRE_BYTES,
} from "../model/groundRegionalExecution";
import { GroundPlayback3D } from "./GroundPlayback3D";

const MAX_IMPORT_BYTES = 5 * 1024 * 1024;
type ImportKind = "result" | "regional";

function readFileText(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error("The selected file could not be read."));
    reader.onload = () => resolve(String(reader.result ?? ""));
    reader.readAsText(file, "utf-8");
  });
}

function metricRows(result: FlightToGroundResult): Array<[string, string]> {
  const summary = result.summary;
  if (summary === null) return [];
  const endpoint = result.status === "complete" ? "Total" : "Observed total";
  return [
    ["Carry", `${summary.carry_distance_m.toFixed(3)} m`],
    [endpoint, `${summary.total_distance_m.toFixed(3)} m`],
    ["Bounce air", `${summary.bounce_air_distance_m.toFixed(3)} m`],
    ["Skid", `${summary.skid_distance_m.toFixed(3)} m`],
    ["Roll", `${summary.roll_distance_m.toFixed(3)} m`],
    ["Surface path", `${summary.surface_path_distance_m.toFixed(3)} m`],
  ];
}

function Summary({ result }: { readonly result: FlightToGroundResult }) {
  return (
    <section className="rounded-lg border border-slate-800 bg-slate-950/40 p-3" aria-label="Ground result summary">
      <h3 className="mb-2 font-semibold text-slate-100">Result summary</h3>
      <table className="w-full text-left text-sm">
        <thead><tr><th scope="col">Metric</th><th scope="col">Value</th></tr></thead>
        <tbody>{metricRows(result).map(([label, value]) => (
          <tr key={label}><th scope="row" className="py-1 font-medium">{label}</th><td>{value}</td></tr>
        ))}</tbody>
      </table>
      <dl className="mt-3 grid grid-cols-1 gap-1 text-xs text-slate-300 sm:grid-cols-2">
        <div><dt className="font-semibold">Schema</dt><dd>{result.schema_version}</dd></div>
        <div><dt className="font-semibold">Status</dt><dd>{result.status}</dd></div>
        <div><dt className="font-semibold">Unit system</dt><dd>{result.unit_system}</dd></div>
        <div><dt className="font-semibold">Frame</dt><dd>{result.frame}</dd></div>
        <div><dt className="font-semibold">Surface ID</dt><dd>{result.surface_id}</dd></div>
        <div><dt className="font-semibold">Termination</dt><dd>{result.termination.reason} · completed={String(result.termination.completed)} · {result.termination.time_s.toFixed(6)} s</dd></div>
        <div><dt className="font-semibold">Model</dt><dd>{result.model_id} {result.model_version}</dd></div>
        <div><dt className="font-semibold">Request</dt><dd>{result.request_id}</dd></div>
      </dl>
    </section>
  );
}

function Evidence({ result }: { readonly result: FlightToGroundResult }) {
  return (
    <section className="rounded-lg border border-slate-800 bg-slate-950/40 p-3" aria-label="Ground warnings and provenance">
      <h3 className="mb-2 font-semibold text-slate-100">Warnings, calibration & provenance</h3>
      {result.warnings.length === 0 ? <p className="text-sm text-slate-400">No warnings reported.</p> : (
        <ul className="space-y-2 text-sm">{result.warnings.map((warning, index) => (
          <li key={`${warning.code}-${index}`} className="rounded border border-amber-400/20 p-2">
            <strong>{warning.code}</strong> · {warning.severity}<br />{warning.message}
          </li>
        ))}</ul>
      )}
      <dl className="mt-3 grid gap-2 text-xs sm:grid-cols-2">
        <div><dt className="font-semibold">Producer</dt><dd>{result.provenance.producer} {result.provenance.producer_version}</dd></div>
        <div><dt className="font-semibold">Source revision</dt><dd>{result.provenance.source_revision}</dd></div>
        <div><dt className="font-semibold">Input SHA-256</dt><dd className="break-all">{result.provenance.input_sha256}</dd></div>
        <div><dt className="font-semibold">Calibration ID</dt><dd>{result.calibration.calibration_id}</dd></div>
        <div><dt className="font-semibold">Calibration</dt><dd>{result.calibration.kind} · {result.calibration.source}</dd></div>
        <div><dt className="font-semibold">Confidence</dt><dd>{result.calibration.confidence.toFixed(2)}</dd></div>
      </dl>
    </section>
  );
}

function VectorCells({ vectors }: { readonly vectors: readonly GroundVec3[] }) {
  return <>{vectors.flatMap((vector, group) => vector.map((value, axis) => (
    <td key={`${group}-${axis}`}>{value.toFixed(6)}</td>
  )))}</>;
}

function ResultTables({ result }: { readonly result: FlightToGroundResult }) {
  const start = result.trajectory[0].time_s;
  const trajectory = groundEvidenceWindow(result.trajectory);
  const events = groundEvidenceWindow(result.events);
  return (
    <div className="grid gap-4 xl:grid-cols-2">
      <section className="overflow-x-auto rounded-lg border border-slate-800 p-3">
        <h3 className="mb-2 font-semibold">Trajectory samples</h3>
        {trajectory.disclosure && <p role="status" className="mb-2 text-xs text-amber-200">
          {trajectory.disclosure}
        </p>}
        <table className="min-w-full text-left text-xs" aria-label="Ground trajectory evidence">
          <thead><tr>{["Sample", "Absolute s", "Elapsed s", "Phase", "x m", "y m", "z m", "vx m/s", "vy m/s", "vz m/s", "ωx rad/s", "ωy rad/s", "ωz rad/s"].map(
            (label) => <th scope="col" key={label} className="pr-3">{label}</th>,
          )}</tr></thead>
          <tbody>{trajectory.rows.map((point, index) => (
            <tr key={`${point.time_s}-${index}`}><th scope="row">{index}</th>
              <td>{point.time_s.toFixed(6)}</td><td>{(point.time_s - start).toFixed(6)}</td>
              <td>{point.phase}</td><VectorCells vectors={[
                point.position_m, point.velocity_m_s, point.angular_velocity_rad_s,
              ]} /></tr>
          ))}</tbody>
        </table>
      </section>
      <section className="overflow-x-auto rounded-lg border border-slate-800 p-3">
        <h3 className="mb-2 font-semibold">Event ledger</h3>
        {events.disclosure && <p role="status" className="mb-2 text-xs text-amber-200">
          {events.disclosure}
        </p>}
        <table className="min-w-full text-left text-xs" aria-label="Ground event evidence">
          <thead><tr>{["Sequence", "Event", "Time s", "x m", "y m", "z m", "vx before m/s", "vy before m/s", "vz before m/s", "vx after m/s", "vy after m/s", "vz after m/s", "ωx before rad/s", "ωy before rad/s", "ωz before rad/s", "ωx after rad/s", "ωy after rad/s", "ωz after rad/s"].map(
            (label) => <th scope="col" key={label} className="pr-3">{label}</th>,
          )}</tr></thead>
          <tbody>{events.rows.map((event) => (
            <tr key={event.sequence}><th scope="row">{event.sequence}</th><td>{event.event_type}</td>
              <td>{event.time_s.toFixed(6)}</td><VectorCells vectors={[
                event.position_m,
                event.velocity_before_m_s,
                event.velocity_after_m_s,
                event.angular_velocity_before_rad_s,
                event.angular_velocity_after_rad_s,
              ]} /></tr>
          ))}</tbody>
        </table>
      </section>
    </div>
  );
}

function LoadedResult({ result }: { readonly result: FlightToGroundResult }) {
  const timeline = new GroundPlaybackTimeline(result);
  return <>
    <div className="grid gap-4 lg:grid-cols-[minmax(15rem,22rem)_1fr]">
      <div className="space-y-4"><Summary result={result} /><Evidence result={result} /></div>
      <GroundPlayback3D timeline={timeline} />
    </div>
    <ResultTables result={result} />
  </>;
}

export function GroundPlaybackPanel() {
  const importGeneration = useRef(0);
  const [result, setResult] = useState<FlightToGroundResult | null>(null);
  const [message, setMessage] = useState("No result loaded.");
  const [error, setError] = useState<string | null>(null);

  const importFile = async (file: File | undefined, kind: ImportKind) => {
    if (!file) return;
    const generation = importGeneration.current + 1;
    importGeneration.current = generation;
    try {
      const limit = kind === "regional"
        ? MAX_GROUND_REGIONAL_EXECUTION_WIRE_BYTES
        : MAX_IMPORT_BYTES;
      if (file.size > limit) throw new RangeError("File exceeds the import size limit.");
      const text = await readFileText(file);
      const timeline = kind === "regional"
        ? timelineFromRegionalExecution(groundRegionalExecutionResultFromJson(text))
        : new GroundPlaybackTimeline(flightToGroundResultFromJson(text));
      const parsed = timeline.result;
      if (parsed.trajectory.length > GROUND_PLAYBACK_MAX_POINTS) {
        throw new RangeError("Trajectory exceeds the 100,000 point display limit.");
      }
      if (generation !== importGeneration.current) return;
      setResult(parsed);
      setError(null);
      setMessage(`Loaded ${file.name} — ${parsed.status}; ${parsed.trajectory.length} samples.`);
    } catch (reason) {
      if (generation !== importGeneration.current) return;
      const detail = reason instanceof Error ? reason.message : "Unknown import error.";
      const retained = result === null ? "" : " Last valid result remains loaded.";
      setError(`Could not import ${file.name}: ${detail}${retained}`);
    }
  };

  return (
    <section className="space-y-4" aria-labelledby="ground-playback-heading">
      <header className="rounded-lg border border-sky-500/30 bg-sky-950/20 p-4">
        <h2 id="ground-playback-heading" className="text-lg font-semibold text-slate-100">Ground Playback</h2>
        <p className="mt-1 text-sm text-slate-300">
          Import a strict flight-to-ground-result/v1 JSON or validated
          ground-regional-execution-result/v1 JSON. This browser viewer reuses
          existing evidence and does not execute ground physics.
        </p>
        <p className="mt-1 text-xs text-slate-400">
          Result v1 does not embed surface geometry, so neutral locked-scale axes are shown instead of a claimed terrain plane.
        </p>
      </header>
      <label className="inline-flex cursor-pointer rounded border border-sky-500/60 bg-sky-500/10 px-3 py-2 text-sm font-semibold text-sky-200">
        Import Ground Result JSON…
        <input type="file" accept="application/json,.json" className="sr-only"
          aria-label="Import strict ground result JSON"
          onChange={(event) => {
            void importFile(event.target.files?.[0], "result");
            event.currentTarget.value = "";
          }} />
      </label>
      <label className="ml-2 inline-flex cursor-pointer rounded border border-sky-500/60 bg-sky-500/10 px-3 py-2 text-sm font-semibold text-sky-200">
        Import Regional Execution JSON…
        <input type="file" accept="application/json,.json" className="sr-only"
          aria-label="Import strict regional ground execution JSON"
          onChange={(event) => {
            void importFile(event.target.files?.[0], "regional");
            event.currentTarget.value = "";
          }} />
      </label>
      {error ? <p role="alert" className="rounded border border-red-500/40 bg-red-950/30 p-3 text-sm text-red-200">{error}</p>
        : <p role="status" className="text-sm text-slate-300">{message}</p>}
      {result === null ? (
        <div className="rounded-lg border border-dashed border-slate-700 p-8 text-center text-slate-400">
          Choose an exact result record to enable phase-aware playback and evidence tables.
        </div>
      ) : <LoadedResult result={result} />}
    </section>
  );
}
