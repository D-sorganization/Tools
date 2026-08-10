/** Strict result-only ground playback workspace. */

import { useRef, useState } from "react";

import type { FlightToGroundResult } from "../model/flightGroundTypes";
import { flightToGroundResultFromJson } from "../model/flightGroundContract";
import { GroundPlaybackTimeline } from "../model/groundPlayback";
import { GroundPlaybackComparison } from "../model/groundPlaybackComparison";
import {
  GROUND_PLAYBACK_WORKSPACE_SCHEMA,
  groundWorkspaceFromJson,
  type GroundPlaybackWorkspace,
} from "../model/groundPlaybackWorkspace";
import type { GroundPlaybackPortableState } from "./GroundPlayback3D";
import { GroundPlaybackLoadedResult } from "./GroundPlaybackLoadedResult";
import { GroundPlaybackToolbar } from "./GroundPlaybackToolbar";

const MAX_IMPORT_BYTES = 5 * 1024 * 1024;
const MAX_IMPORT_POINTS = 100_000;

function readFileText(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () =>
      reject(new Error("The selected file could not be read."));
    reader.onload = () => resolve(String(reader.result ?? ""));
    reader.readAsText(file, "utf-8");
  });
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
  const [comparison, setComparison] = useState<GroundPlaybackComparison | null>(
    null,
  );
  const [showComparison, setShowComparison] = useState(false);
  const [comparisonMessage, setComparisonMessage] = useState(
    "No comparison loaded.",
  );
  const [comparisonError, setComparisonError] = useState<string | null>(null);

  const importFile = async (file: File | undefined) => {
    if (!file) return;
    comparisonImportGeneration.current += 1;
    const generation = importGeneration.current + 1;
    importGeneration.current = generation;
    try {
      if (file.size > MAX_IMPORT_BYTES)
        throw new RangeError("File exceeds the 5 MiB import limit.");
      const text = await readFileText(file);
      const parsed = flightToGroundResultFromJson(text);
      if (parsed.trajectory.length > MAX_IMPORT_POINTS) {
        throw new RangeError(
          "Trajectory exceeds the 100,000 point display limit.",
        );
      }
      new GroundPlaybackTimeline(parsed);
      if (generation !== importGeneration.current) return;
      comparisonImportGeneration.current += 1;
      const initialState: GroundPlaybackPortableState = {
        playback: { timeS: parsed.trajectory[0].time_s, speed: 1, loop: false },
        view: {
          yawDeg: (-0.65 * 180) / Math.PI,
          pitchDeg: (0.38 * 180) / Math.PI,
          zoom: 1,
        },
      };
      portableState.current = initialState;
      setLoaded({ result: parsed, initialState, generation });
      setComparison(null);
      setShowComparison(false);
      setComparisonError(null);
      setComparisonMessage(
        comparison === null
          ? "No comparison loaded."
          : "Comparison cleared after the primary result changed.",
      );
      setError(null);
      setMessage(
        `Loaded ${file.name} — ${parsed.status}; ${parsed.trajectory.length} samples.`,
      );
    } catch (reason) {
      if (generation !== importGeneration.current) return;
      const detail =
        reason instanceof Error ? reason.message : "Unknown import error.";
      const retained =
        loaded === null ? "" : " Last valid result remains loaded.";
      setError(`Could not import ${file.name}: ${detail}${retained}`);
    }
  };

  const importWorkspace = async (file: File | undefined) => {
    if (!file) return;
    comparisonImportGeneration.current += 1;
    const generation = importGeneration.current + 1;
    importGeneration.current = generation;
    try {
      if (file.size > MAX_IMPORT_BYTES)
        throw new RangeError("File exceeds the 5 MiB import limit.");
      const candidate = groundWorkspaceFromJson(await readFileText(file));
      if (candidate.result.trajectory.length > MAX_IMPORT_POINTS) {
        throw new RangeError(
          "Trajectory exceeds the 100,000 point display limit.",
        );
      }
      if (generation !== importGeneration.current) return;
      comparisonImportGeneration.current += 1;
      const initialState = {
        playback: candidate.playback,
        view: candidate.view,
      };
      portableState.current = initialState;
      setLoaded({ result: candidate.result, initialState, generation });
      setComparison(null);
      setShowComparison(false);
      setComparisonError(null);
      setComparisonMessage(
        comparison === null
          ? "No comparison loaded."
          : "Comparison cleared after the primary result changed.",
      );
      setError(null);
      setMessage(
        `Loaded workspace ${file.name} — ${candidate.result.status}; paused at ${candidate.playback.timeS} s.`,
      );
    } catch (reason) {
      if (generation !== importGeneration.current) return;
      const detail =
        reason instanceof Error ? reason.message : "Unknown import error.";
      const retained =
        loaded === null ? "" : " Last valid playback remains loaded.";
      setError(`Could not import ${file.name}: ${detail}${retained}`);
    }
  };

  const importComparison = async (file: File | undefined) => {
    if (!file || loaded === null) return;
    const generation = comparisonImportGeneration.current + 1;
    comparisonImportGeneration.current = generation;
    try {
      if (file.size > MAX_IMPORT_BYTES)
        throw new RangeError("File exceeds the 5 MiB import limit.");
      const parsed = flightToGroundResultFromJson(await readFileText(file));
      if (parsed.trajectory.length > MAX_IMPORT_POINTS) {
        throw new RangeError(
          "Trajectory exceeds the 100,000 point display limit.",
        );
      }
      const candidate = new GroundPlaybackComparison(
        new GroundPlaybackTimeline(loaded.result),
        new GroundPlaybackTimeline(parsed),
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
      const detail =
        reason instanceof Error ? reason.message : "Unknown import error.";
      const retained =
        comparison === null ? "" : " Last valid comparison remains loaded.";
      setComparisonError(
        `Could not import comparison ${file.name}: ${detail}${retained}`,
      );
    }
  };

  const workspace = (): GroundPlaybackWorkspace => {
    if (loaded === null || portableState.current === null)
      throw new Error("No result loaded.");
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
        <h2
          id="ground-playback-heading"
          className="text-lg font-semibold text-slate-100"
        >
          Ground Playback
        </h2>
        <p className="mt-1 text-sm text-slate-300">
          Import a strict flight-to-ground-result/v1 JSON generated by the
          Python reference executor. This browser viewer does not execute ground
          physics.
        </p>
        <p className="mt-1 text-xs text-slate-400">
          Result v1 does not embed surface geometry, so neutral locked-scale
          axes are shown instead of a claimed terrain plane.
        </p>
      </header>
      <GroundPlaybackToolbar
        result={loaded?.result ?? null}
        comparison={comparison}
        showComparison={showComparison}
        onShowComparisonChange={setShowComparison}
        onImportResult={(file) => void importFile(file)}
        onImportWorkspace={(file) => void importWorkspace(file)}
        onImportComparison={(file) => void importComparison(file)}
        workspace={workspace}
      />
      {error ? (
        <p
          role="alert"
          className="rounded border border-red-500/40 bg-red-950/30 p-3 text-sm text-red-200"
        >
          {error}
        </p>
      ) : (
        <p role="status" className="text-sm text-slate-300">
          {message}
        </p>
      )}
      {loaded !== null &&
        (comparisonError ? (
          <p
            role="alert"
            className="rounded border border-red-500/40 bg-red-950/30 p-3 text-sm text-red-200"
          >
            {comparisonError}
          </p>
        ) : (
          <p
            role="status"
            aria-label="Ground comparison status"
            className="text-sm text-slate-300"
          >
            {comparisonMessage}
          </p>
        ))}
      {loaded === null ? (
        <div className="rounded-lg border border-dashed border-slate-700 p-8 text-center text-slate-400">
          Choose an exact result record to enable phase-aware playback and
          evidence tables.
        </div>
      ) : (
        <GroundPlaybackLoadedResult
          key={loaded.generation}
          result={loaded.result}
          comparison={comparison}
          showComparison={showComparison}
          initialState={loaded.initialState}
          onStateChange={(state) => {
            portableState.current = state;
          }}
        />
      )}
    </section>
  );
}
