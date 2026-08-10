/** Strict result-only ground playback workspace. */

import { useRef, useState } from "react";

import type { FlightToGroundResult, GroundVec3 } from "../model/flightGroundTypes";
import { flightToGroundResultFromJson } from "../model/flightGroundContract";
import { GroundPlaybackTimeline } from "../model/groundPlayback";
import {
  GroundPlaybackComparison,
  groundComparisonCsv,
  groundComparisonJson,
} from "../model/groundPlaybackComparison";
import {
  GROUND_PLAYBACK_WORKSPACE_SCHEMA,
  groundEventCsv,
  groundResultJson,
  groundTrajectoryCsv,
  groundWorkspaceFromJson,
  groundWorkspaceToJson,
  type GroundPlaybackWorkspace,
} from "../model/groundPlaybackWorkspace";
import { GroundPlayback3D, type GroundPlaybackPortableState } from "./GroundPlayback3D";
import { GroundPlaybackComparisonSummary } from "./GroundPlaybackComparisonSummary";
import { downloadText } from "./variationUi";

const MAX_IMPORT_BYTES = 5 * 1024 * 1024;
const MAX_IMPORT_POINTS = 100_000;

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
  return (
    <div className="grid gap-4 xl:grid-cols-2">
      <section className="overflow-x-auto rounded-lg border border-slate-800 p-3">
        <h3 className="mb-2 font-semibold">Trajectory samples</h3>
        <table className="min-w-full text-left text-xs" aria-label="Ground trajectory evidence">
          <thead><tr>{["Sample", "Absolute s", "Elapsed s", "Phase", "x m", "y m", "z m", "vx m/s", "vy m/s", "vz m/s", "ωx rad/s", "ωy rad/s", "ωz rad/s"].map(
            (label) => <th scope="col" key={label} className="pr-3">{label}</th>,
          )}</tr></thead>
          <tbody>{result.trajectory.map((point, index) => (
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
        <table className="min-w-full text-left text-xs" aria-label="Ground event evidence">
          <thead><tr>{["Sequence", "Event", "Time s", "x m", "y m", "z m", "vx before m/s", "vy before m/s", "vz before m/s", "vx after m/s", "vy after m/s", "vz after m/s", "ωx before rad/s", "ωy before rad/s", "ωz before rad/s", "ωx after rad/s", "ωy after rad/s", "ωz after rad/s"].map(
            (label) => <th scope="col" key={label} className="pr-3">{label}</th>,
          )}</tr></thead>
          <tbody>{result.events.map((event) => (
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

function LoadedResult({ result, comparison, showComparison, initialState, onStateChange }: {
  readonly result: FlightToGroundResult;
  readonly comparison: GroundPlaybackComparison | null;
  readonly showComparison: boolean;
  readonly initialState: GroundPlaybackPortableState;
  readonly onStateChange: (state: GroundPlaybackPortableState) => void;
}) {
  const timeline = new GroundPlaybackTimeline(result);
  return <>
    <div className="grid gap-4 lg:grid-cols-[minmax(15rem,22rem)_1fr]">
      <div className="space-y-4"><Summary result={result} /><Evidence result={result} /></div>
      <GroundPlayback3D timeline={timeline}
        comparisonTimeline={comparison?.comparison}
        showComparison={showComparison}
        initialState={initialState} onStateChange={onStateChange} />
    </div>
    {comparison && <GroundPlaybackComparisonSummary comparison={comparison} />}
    <ResultTables result={result} />
  </>;
}

export function GroundPlaybackPanel() {
  const importGeneration = useRef(0);
  const comparisonImportGeneration = useRef(0);
  const portableState = useRef<GroundPlaybackPortableState | null>(null);
  const [loaded, setLoaded] = useState<{
    readonly result: FlightToGroundResult;
    readonly initialState: GroundPlaybackPortableState;
    readonly generation: number;
  } | null>(null);
  const [message, setMessage] = useState("No result loaded.");
  const [error, setError] = useState<string | null>(null);
  const [comparison, setComparison] = useState<GroundPlaybackComparison | null>(null);
  const [showComparison, setShowComparison] = useState(false);
  const [comparisonMessage, setComparisonMessage] = useState("No comparison loaded.");
  const [comparisonError, setComparisonError] = useState<string | null>(null);

  const importFile = async (file: File | undefined) => {
    if (!file) return;
    comparisonImportGeneration.current += 1;
    const generation = importGeneration.current + 1;
    importGeneration.current = generation;
    try {
      if (file.size > MAX_IMPORT_BYTES) throw new RangeError("File exceeds the 5 MiB import limit.");
      const text = await readFileText(file);
      const parsed = flightToGroundResultFromJson(text);
      if (parsed.trajectory.length > MAX_IMPORT_POINTS) {
        throw new RangeError("Trajectory exceeds the 100,000 point display limit.");
      }
      new GroundPlaybackTimeline(parsed);
      if (generation !== importGeneration.current) return;
      const initialState: GroundPlaybackPortableState = {
        playback: { timeS: parsed.trajectory[0].time_s, speed: 1, loop: false },
        view: { yawDeg: -0.65 * 180 / Math.PI, pitchDeg: 0.38 * 180 / Math.PI, zoom: 1 },
      };
      portableState.current = initialState;
      setLoaded({ result: parsed, initialState, generation });
      setComparison(null);
      setShowComparison(false);
      setComparisonError(null);
      setComparisonMessage(comparison === null
        ? "No comparison loaded."
        : "Comparison cleared after the primary result changed.");
      setError(null);
      setMessage(`Loaded ${file.name} — ${parsed.status}; ${parsed.trajectory.length} samples.`);
    } catch (reason) {
      if (generation !== importGeneration.current) return;
      const detail = reason instanceof Error ? reason.message : "Unknown import error.";
      const retained = loaded === null ? "" : " Last valid result remains loaded.";
      setError(`Could not import ${file.name}: ${detail}${retained}`);
    }
  };

  const importWorkspace = async (file: File | undefined) => {
    if (!file) return;
    comparisonImportGeneration.current += 1;
    const generation = importGeneration.current + 1;
    importGeneration.current = generation;
    try {
      if (file.size > MAX_IMPORT_BYTES) throw new RangeError("File exceeds the 5 MiB import limit.");
      const candidate = groundWorkspaceFromJson(await readFileText(file));
      if (candidate.result.trajectory.length > MAX_IMPORT_POINTS) {
        throw new RangeError("Trajectory exceeds the 100,000 point display limit.");
      }
      if (generation !== importGeneration.current) return;
      const initialState = { playback: candidate.playback, view: candidate.view };
      portableState.current = initialState;
      setLoaded({ result: candidate.result, initialState, generation });
      setComparison(null);
      setShowComparison(false);
      setComparisonError(null);
      setComparisonMessage(comparison === null
        ? "No comparison loaded."
        : "Comparison cleared after the primary result changed.");
      setError(null);
      setMessage(`Loaded workspace ${file.name} — ${candidate.result.status}; paused at ${candidate.playback.timeS} s.`);
    } catch (reason) {
      if (generation !== importGeneration.current) return;
      const detail = reason instanceof Error ? reason.message : "Unknown import error.";
      const retained = loaded === null ? "" : " Last valid playback remains loaded.";
      setError(`Could not import ${file.name}: ${detail}${retained}`);
    }
  };

  const importComparison = async (file: File | undefined) => {
    if (!file || loaded === null) return;
    const generation = comparisonImportGeneration.current + 1;
    comparisonImportGeneration.current = generation;
    try {
      if (file.size > MAX_IMPORT_BYTES) throw new RangeError("File exceeds the 5 MiB import limit.");
      const parsed = flightToGroundResultFromJson(await readFileText(file));
      if (parsed.trajectory.length > MAX_IMPORT_POINTS) {
        throw new RangeError("Trajectory exceeds the 100,000 point display limit.");
      }
      const candidate = new GroundPlaybackComparison(
        new GroundPlaybackTimeline(loaded.result), new GroundPlaybackTimeline(parsed),
      );
      if (generation !== comparisonImportGeneration.current) return;
      setComparison(candidate);
      setShowComparison(true);
      setComparisonError(null);
      setComparisonMessage(
        `Loaded comparison ${file.name} — ${parsed.status}; ${parsed.trajectory.length} samples. Deltas are comparison minus primary.`,
      );
    } catch (reason) {
      if (generation !== comparisonImportGeneration.current) return;
      const detail = reason instanceof Error ? reason.message : "Unknown import error.";
      const retained = comparison === null ? "" : " Last valid comparison remains loaded.";
      setComparisonError(`Could not import comparison ${file.name}: ${detail}${retained}`);
    }
  };

  const workspace = (): GroundPlaybackWorkspace => {
    if (loaded === null || portableState.current === null) throw new Error("No result loaded.");
    return {
      schemaVersion: GROUND_PLAYBACK_WORKSPACE_SCHEMA,
      result: loaded.result,
      playback: portableState.current.playback,
      view: portableState.current.view,
    };
  };

  return (
    <section className="space-y-4" aria-labelledby="ground-playback-heading">
      <header className="rounded-lg border border-sky-500/30 bg-sky-950/20 p-4">
        <h2 id="ground-playback-heading" className="text-lg font-semibold text-slate-100">Ground Playback</h2>
        <p className="mt-1 text-sm text-slate-300">
          Import a strict flight-to-ground-result/v1 JSON generated by the Python reference executor.
          This browser viewer does not execute ground physics.
        </p>
        <p className="mt-1 text-xs text-slate-400">
          Result v1 does not embed surface geometry, so neutral locked-scale axes are shown instead of a claimed terrain plane.
        </p>
      </header>
      <div className="flex flex-wrap gap-2">
      <label className="inline-flex cursor-pointer rounded border border-sky-500/60 bg-sky-500/10 px-3 py-2 text-sm font-semibold text-sky-200">
        Import Ground Result JSON…
        <input type="file" accept="application/json,.json" className="sr-only"
          aria-label="Import strict ground result JSON"
          onChange={(event) => {
            void importFile(event.target.files?.[0]);
            event.currentTarget.value = "";
          }} />
      </label>
      <label className="inline-flex cursor-pointer rounded border border-slate-700 px-3 py-2 text-sm font-semibold text-slate-200">
        Import Workspace JSON…
        <input type="file" accept="application/json,.json" className="sr-only"
          aria-label="Import ground playback workspace"
          onChange={(event) => {
            void importWorkspace(event.target.files?.[0]);
            event.currentTarget.value = "";
          }} />
      </label>
      {loaded !== null && <>
        <label className="inline-flex cursor-pointer rounded border border-cyan-500/60 bg-cyan-500/10 px-3 py-2 text-sm font-semibold text-cyan-200">
          Import Comparison JSON…
          <input type="file" accept="application/json,.json" className="sr-only"
            aria-label="Import ground comparison result JSON"
            onChange={(event) => {
              void importComparison(event.target.files?.[0]);
              event.currentTarget.value = "";
            }} />
        </label>
        <button type="button" aria-label="Save ground playback workspace"
          className="rounded border border-slate-700 px-3 py-2 text-sm"
          onClick={() => downloadText("ground-playback-workspace.json", groundWorkspaceToJson(workspace()), "application/json")}>Save Workspace</button>
        <button type="button" aria-label="Export ground result JSON"
          className="rounded border border-slate-700 px-3 py-2 text-sm"
          onClick={() => downloadText("ground-result.json", groundResultJson(loaded.result), "application/json")}>Result JSON</button>
        <button type="button" aria-label="Export ground trajectory CSV"
          className="rounded border border-slate-700 px-3 py-2 text-sm"
          onClick={() => downloadText("ground-trajectory.csv", groundTrajectoryCsv(loaded.result), "text/csv;charset=utf-8")}>Trajectory CSV</button>
        <button type="button" aria-label="Export ground events CSV"
          className="rounded border border-slate-700 px-3 py-2 text-sm"
          onClick={() => downloadText("ground-events.csv", groundEventCsv(loaded.result), "text/csv;charset=utf-8")}>Events CSV</button>
        {comparison && <>
          <label className="inline-flex items-center gap-2 rounded border border-cyan-500/40 px-3 py-2 text-sm">
            <input type="checkbox" aria-label="Show comparison overlay"
              checked={showComparison}
              onChange={(event) => setShowComparison(event.target.checked)} />
            Show comparison
          </label>
          <button type="button" aria-label="Export ground comparison JSON"
            className="rounded border border-cyan-500/40 px-3 py-2 text-sm"
            onClick={() => downloadText("ground-comparison.json", groundComparisonJson(comparison), "application/json")}>Comparison JSON</button>
          <button type="button" aria-label="Export ground comparison CSV"
            className="rounded border border-cyan-500/40 px-3 py-2 text-sm"
            onClick={() => downloadText("ground-comparison.csv", groundComparisonCsv(comparison), "text/csv;charset=utf-8")}>Comparison CSV</button>
        </>}
      </>}
      </div>
      {error ? <p role="alert" className="rounded border border-red-500/40 bg-red-950/30 p-3 text-sm text-red-200">{error}</p>
        : <p role="status" className="text-sm text-slate-300">{message}</p>}
      {loaded !== null && (comparisonError
        ? <p role="alert" className="rounded border border-red-500/40 bg-red-950/30 p-3 text-sm text-red-200">{comparisonError}</p>
        : <p role="status" aria-label="Ground comparison status" className="text-sm text-slate-300">{comparisonMessage}</p>)}
      {loaded === null ? (
        <div className="rounded-lg border border-dashed border-slate-700 p-8 text-center text-slate-400">
          Choose an exact result record to enable phase-aware playback and evidence tables.
        </div>
      ) : <LoadedResult key={loaded.generation} result={loaded.result}
        comparison={comparison} showComparison={showComparison}
        initialState={loaded.initialState}
        onStateChange={(state) => { portableState.current = state; }} />}
    </section>
  );
}
