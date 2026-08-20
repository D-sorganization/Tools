import { useEffect, useMemo, useRef, useState } from "react";

import { analyzeLaunchMonitorData, numericLaunchMonitorColumns } from "../model/launchMonitorAnalysis";
import type { LaunchMonitorAnalysisResult, LaunchMonitorRow } from "../model/launchMonitorAnalysisTypes";
import {
  createAnalysisExportBundle,
  fingerprintLaunchMonitorRows,
  parseLaunchMonitorProject,
  serializeLaunchMonitorProject,
  type LaunchMonitorProject,
} from "../model/launchMonitorWorkspace";

interface Props { rows: LaunchMonitorRow[]; sourceName: string }

const field = "w-full rounded border border-slate-700 bg-slate-950 px-2 py-2 text-sm text-slate-100";

function download(name: string, content: string, type = "application/json") {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  URL.revokeObjectURL(url);
}

export function LaunchMonitorPlayerWorkspace({ rows, sourceName }: Props) {
  const columns = useMemo(() => [...new Set(rows.flatMap((row) => Object.keys(row)))].sort(), [rows]);
  const numeric = useMemo(() => numericLaunchMonitorColumns(rows), [rows]);
  const [identity, setIdentity] = useState("");
  const [attested, setAttested] = useState(false);
  const [x, setX] = useState(numeric.includes("face_angle") ? "face_angle" : numeric[0] ?? "");
  const [y, setY] = useState(numeric.includes("club_path") ? "club_path" : numeric[1] ?? "");
  const [datasetSha, setDatasetSha] = useState("");
  const [result, setResult] = useState<LaunchMonitorAnalysisResult | null>(null);
  const [message, setMessage] = useState("Select and attest an explicit player identity column.");
  const loadInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setDatasetSha(fingerprintLaunchMonitorRows(rows));
    setAttested(false);
    setIdentity("");
    setResult(null);
  }, [rows]);

  const project = (): LaunchMonitorProject => ({
    contractVersion: "2.0.0",
    name: `${sourceName} player covariation`,
    dataset: {
      sourceName, repository: "local-user-data", revision: "unversioned",
      relativePath: sourceName, sha256: datasetSha, rowCount: rows.length,
    },
    playerIdentity: { column: identity, userAttested: attested },
    selection: { x, y, minSamples: 10, confidenceLevel: 0.95 },
  });
  const ready = Boolean(identity && attested && x && y && x !== y && datasetSha);

  const run = () => {
    try {
      const next = analyzeLaunchMonitorData(rows, {
        outcome: y, predictors: [x], analysisMode: "correlation",
        correlationMethod: "pearson", missingPolicy: "pairwise",
        groupBy: identity, confidenceLevel: 0.95, minSamples: 10,
      });
      setResult(next);
      setMessage(`${next.groups.length} player groups analyzed. Associations are not causal; no player identity was inferred.`);
    } catch (caught) {
      setResult(null);
      setMessage(caught instanceof Error ? caught.message : String(caught));
    }
  };

  const load = async (file: File) => {
    try {
      const saved = parseLaunchMonitorProject(await file.text());
      if (saved.dataset.sha256 !== datasetSha) throw new RangeError("Saved project references a different dataset");
      setIdentity(saved.playerIdentity.column);
      setAttested(saved.playerIdentity.userAttested);
      setX(saved.selection.x);
      setY(saved.selection.y);
      setMessage("Saved project settings restored against the matching dataset fingerprint.");
    } catch (caught) {
      setMessage(caught instanceof Error ? caught.message : String(caught));
    }
  };

  const exportBundle = async () => {
    if (!result) return;
    const bundle = await createAnalysisExportBundle(project(), result as unknown as Record<string, unknown>, rows);
    download("launch-monitor-analysis-bundle.json", JSON.stringify(bundle, null, 2));
  };

  return <section aria-label="Player analytics workspace" className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
    <h3 className="font-semibold text-slate-200">Player Covariation Workspace</h3>
    <p className="mt-1 text-xs text-slate-400">Identity is never inferred from session, club, filename, or row order. Per-player estimates delegate to the existing analysis adapter pending the UpstreamDrift v2 endpoint.</p>
    <div className="mt-3 grid gap-3 sm:grid-cols-3">
      <label className="text-sm text-slate-300">Player identity
        <select aria-label="Player identity column" title="Choose a real player identifier supplied by the dataset owner" value={identity}
          onChange={(event) => { setIdentity(event.target.value); setAttested(false); setResult(null); }} className={`${field} mt-1`}>
          <option value="">Select column</option>{columns.map((column) => <option key={column}>{column}</option>)}
        </select>
      </label>
      <label className="text-sm text-slate-300">X variable
        <select aria-label="Player covariation X variable" title="Choose the first covariation variable" value={x}
          onChange={(event) => { setX(event.target.value); setResult(null); }} className={`${field} mt-1`}>
          {numeric.map((column) => <option key={column}>{column}</option>)}
        </select>
      </label>
      <label className="text-sm text-slate-300">Y variable
        <select aria-label="Player covariation Y variable" title="Choose the second covariation variable" value={y}
          onChange={(event) => { setY(event.target.value); setResult(null); }} className={`${field} mt-1`}>
          {numeric.map((column) => <option key={column}>{column}</option>)}
        </select>
      </label>
    </div>
    <label className="mt-3 flex items-start gap-2 text-sm text-amber-100">
      <input type="checkbox" aria-label="I attest this column identifies a player" title="Required identity-safety attestation"
        checked={attested} disabled={!identity} onChange={(event) => { setAttested(event.target.checked); setResult(null); }} />
      I attest this column identifies a player; it was not inferred from session, club, filename, or row order.
    </label>
    <div className="mt-3 flex flex-wrap gap-2">
      <button type="button" disabled={!ready} title="Run player-group covariation through the analysis adapter" onClick={run}
        className="rounded bg-emerald-700 px-3 py-2 text-sm disabled:opacity-40">Run Player Covariation</button>
      <button type="button" disabled={!ready} title="Save a persistent reference-only project that does not embed private rows"
        onClick={() => download("analysis.lmproject.json", serializeLaunchMonitorProject(project()))}
        className="rounded border border-slate-700 px-3 py-2 text-sm disabled:opacity-40">Save Project</button>
      <input ref={loadInput} type="file" accept=".json,application/json" className="hidden" aria-label="Load saved launch-monitor project"
        onChange={(event) => { const file = event.target.files?.[0]; if (file) void load(file); }} />
      <button type="button" title="Load settings from a saved project after fingerprint verification" onClick={() => loadInput.current?.click()}
        className="rounded border border-slate-700 px-3 py-2 text-sm">Load Project</button>
      <button type="button" disabled={!result} title="Export the project, result, manifest, hashes, and explicit backing rows"
        onClick={() => void exportBundle()} className="rounded border border-slate-700 px-3 py-2 text-sm disabled:opacity-40">Export Full Bundle</button>
    </div>
    <p role="status" className="mt-3 text-sm text-slate-400">{message}</p>
  </section>;
}
