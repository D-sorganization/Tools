/** Import and export controls for the strict ground playback workspace. */

import type { FlightToGroundResult } from "../model/flightGroundTypes";
import {
  groundComparisonCsv,
  groundComparisonJson,
  type GroundPlaybackComparison,
} from "../model/groundPlaybackComparison";
import {
  groundEventCsv,
  groundResultJson,
  groundTrajectoryCsv,
} from "../model/groundPlaybackWorkspace";
import {
  groundWorkspaceV2ToJson,
  type GroundPlaybackWorkspaceV2,
} from "../model/groundPlaybackWorkspaceV2";
import { downloadText } from "./variationUi";

type GroundPlaybackToolbarProps = {
  readonly result: FlightToGroundResult | null;
  readonly comparison: GroundPlaybackComparison | null;
  readonly showComparison: boolean;
  readonly onShowComparisonChange: (show: boolean) => void;
  readonly onImportResult: (file: File | undefined) => void;
  readonly onImportWorkspace: (file: File | undefined) => void;
  readonly onImportComparison: (file: File | undefined) => void;
  readonly workspace: () => GroundPlaybackWorkspaceV2;
};

export function GroundPlaybackToolbar({
  result,
  comparison,
  showComparison,
  onShowComparisonChange,
  onImportResult,
  onImportWorkspace,
  onImportComparison,
  workspace,
}: GroundPlaybackToolbarProps) {
  return (
    <div className="flex flex-wrap gap-2">
      <label className="inline-flex cursor-pointer rounded border border-sky-500/60 bg-sky-500/10 px-3 py-2 text-sm font-semibold text-sky-200">
        Import Ground Result JSON…
        <input
          type="file"
          accept="application/json,.json"
          className="sr-only"
          aria-label="Import strict ground result JSON"
          onChange={(event) => {
            onImportResult(event.target.files?.[0]);
            event.currentTarget.value = "";
          }}
        />
      </label>
      <label className="inline-flex cursor-pointer rounded border border-slate-700 px-3 py-2 text-sm font-semibold text-slate-200">
        Import Workspace JSON…
        <input
          type="file"
          accept="application/json,.json"
          className="sr-only"
          aria-label="Import ground playback workspace"
          onChange={(event) => {
            onImportWorkspace(event.target.files?.[0]);
            event.currentTarget.value = "";
          }}
        />
      </label>
      {result !== null && (
        <>
          <label className="inline-flex cursor-pointer rounded border border-cyan-500/60 bg-cyan-500/10 px-3 py-2 text-sm font-semibold text-cyan-200">
            Import Comparison JSON…
            <input
              type="file"
              accept="application/json,.json"
              className="sr-only"
              aria-label="Import ground comparison result JSON"
              onChange={(event) => {
                onImportComparison(event.target.files?.[0]);
                event.currentTarget.value = "";
              }}
            />
          </label>
          <button
            type="button"
            aria-label="Save ground playback workspace"
            className="rounded border border-slate-700 px-3 py-2 text-sm"
            onClick={() =>
              downloadText(
                "ground-playback-workspace.json",
                groundWorkspaceV2ToJson(workspace()),
                "application/json",
              )
            }
          >
            Save Workspace
          </button>
          <button
            type="button"
            aria-label="Export ground result JSON"
            className="rounded border border-slate-700 px-3 py-2 text-sm"
            onClick={() =>
              downloadText(
                "ground-result.json",
                groundResultJson(result),
                "application/json",
              )
            }
          >
            Result JSON
          </button>
          <button
            type="button"
            aria-label="Export ground trajectory CSV"
            className="rounded border border-slate-700 px-3 py-2 text-sm"
            onClick={() =>
              downloadText(
                "ground-trajectory.csv",
                groundTrajectoryCsv(result),
                "text/csv;charset=utf-8",
              )
            }
          >
            Trajectory CSV
          </button>
          <button
            type="button"
            aria-label="Export ground events CSV"
            className="rounded border border-slate-700 px-3 py-2 text-sm"
            onClick={() =>
              downloadText(
                "ground-events.csv",
                groundEventCsv(result),
                "text/csv;charset=utf-8",
              )
            }
          >
            Events CSV
          </button>
          {comparison && (
            <>
              <label className="inline-flex items-center gap-2 rounded border border-cyan-500/40 px-3 py-2 text-sm">
                <input
                  type="checkbox"
                  aria-label="Show comparison overlay"
                  checked={showComparison}
                  onChange={(event) =>
                    onShowComparisonChange(event.target.checked)
                  }
                />
                Show comparison
              </label>
              <button
                type="button"
                aria-label="Export ground comparison JSON"
                className="rounded border border-cyan-500/40 px-3 py-2 text-sm"
                onClick={() =>
                  downloadText(
                    "ground-comparison.json",
                    groundComparisonJson(comparison),
                    "application/json",
                  )
                }
              >
                Comparison JSON
              </button>
              <button
                type="button"
                aria-label="Export ground comparison CSV"
                className="rounded border border-cyan-500/40 px-3 py-2 text-sm"
                onClick={() =>
                  downloadText(
                    "ground-comparison.csv",
                    groundComparisonCsv(comparison),
                    "text/csv;charset=utf-8",
                  )
                }
              >
                Comparison CSV
              </button>
              <button
                type="button"
                aria-label="Export ground comparison trajectory CSV"
                className="rounded border border-cyan-500/40 px-3 py-2 text-sm"
                onClick={() =>
                  downloadText(
                    "ground-comparison-trajectory.csv",
                    groundTrajectoryCsv(comparison.comparison.result),
                    "text/csv;charset=utf-8",
                  )
                }
              >
                Comparison Trajectory
              </button>
              <button
                type="button"
                aria-label="Export ground comparison events CSV"
                className="rounded border border-cyan-500/40 px-3 py-2 text-sm"
                onClick={() =>
                  downloadText(
                    "ground-comparison-events.csv",
                    groundEventCsv(comparison.comparison.result),
                    "text/csv;charset=utf-8",
                  )
                }
              >
                Comparison Events
              </button>
            </>
          )}
        </>
      )}
    </div>
  );
}
