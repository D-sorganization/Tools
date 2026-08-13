import { datasetToCsv, datasetToJson } from "../model/variationAnalysis";
import { planToJson, type VariationDatasetTs, type VariationPlanTs } from "../model/variation";
import { BUTTON_CLASS, PANEL_CLASS, downloadText, readFileText } from "./variationUi";
import {
  swingEnsembleToJson,
  localizedTorqueSourcesToCsv,
  swingTracesToCsv,
  type SwingVariationResultTs,
} from "../model/variationSwingEnsemble";
import type { VariationExecutionProgress } from "../model/variationExecutionService";
import type { VariationVisualState } from "../model/variationVisualState";

interface VariationActionsProps {
  plan: VariationPlanTs;
  dataset: VariationDatasetTs | null;
  ensemble: SwingVariationResultTs | null;
  status: string;
  busy: boolean;
  progress: VariationExecutionProgress | null;
  visualState: VariationVisualState;
  onRun: () => void;
  onCancel: () => void;
  onImportText: (text: string) => void;
  onImportError: (message: string) => void;
}

export function VariationActions({
  plan,
  dataset,
  ensemble,
  status,
  busy,
  progress,
  visualState,
  onRun,
  onCancel,
  onImportText,
  onImportError,
}: VariationActionsProps): JSX.Element {
  let validPlanJson: string | null = null;
  try {
    validPlanJson = planToJson(plan);
  } catch {
    validPlanJson = null;
  }
  return (
    <div className={PANEL_CLASS} aria-busy={busy}>
      <div className="flex flex-wrap gap-2">
        <button
          type="button"
          disabled={!ensemble?.runs.some((trial) => trial.localizedTorqueCommands.length > 0)}
          onClick={() => ensemble && downloadText(
            "variation_localized_torque_sources.csv",
            localizedTorqueSourcesToCsv(ensemble),
            "text/csv",
          )}
          title="Download trial-local spec IDs, half-open windows, topological joints, torque units, and provenance."
          className={BUTTON_CLASS}
        >
          Localized Torque CSV
        </button>
        <button
          type="button"
          onClick={onRun}
          disabled={busy}
          title="Run only the analyses selected in Analysis Execution."
          className={`${BUTTON_CLASS} border-sky-500/60 text-sky-300`}
        >
          Run Variation Study
        </button>
        <button
          type="button"
          onClick={onCancel}
          disabled={!busy}
          title="Cancel the active worker and discard all partial results."
          className={BUTTON_CLASS}
        >
          Cancel Variation Study
        </button>
        <button
          type="button"
          disabled={!ensemble}
          onClick={() => ensemble && downloadText(
            "variation_swing_traces.csv",
            swingTracesToCsv(ensemble),
            "text/csv",
          )}
          title="Download every trial, time sample, and modeled point in the explicit app frame."
          className={BUTTON_CLASS}
        >
          Swing Traces CSV
        </button>
        <button
          type="button"
          disabled={!ensemble}
          onClick={() => ensemble && downloadText(
            "variation_swing_ensemble.json",
            swingEnsembleToJson(ensemble),
            "application/json",
          )}
          title="Download the complete plan, typed outcomes, scalar results, and swing traces."
          className={BUTTON_CLASS}
        >
          Swing Ensemble JSON
        </button>
        <button
          type="button"
          disabled={!dataset}
          onClick={() => dataset && downloadText(
            "variation_dataset.csv",
            datasetToCsv(dataset),
            "text/csv",
          )}
          title="Download the jointly enabled runs table as CSV."
          className={BUTTON_CLASS}
        >
          Dataset CSV
        </button>
        <button
          type="button"
          disabled={!dataset}
          onClick={() => dataset && downloadText(
            "variation_dataset.json",
            datasetToJson(dataset),
            "application/json",
          )}
          title="Download the jointly enabled dataset and complete v2 plan as JSON."
          className={BUTTON_CLASS}
        >
          Dataset JSON
        </button>
        <button
          type="button"
          disabled={validPlanJson === null}
          onClick={() => validPlanJson !== null && downloadText(
            "variation_plan.json", validPlanJson, "application/json",
          )}
          title="Export the complete v2 physical plan as JSON."
          className={BUTTON_CLASS}
        >
          Export Plan JSON
        </button>
        <label className={`${BUTTON_CLASS} cursor-pointer`} title="Import a v1 or v2 plan JSON file.">
          Import Plan JSON
          <input
            aria-label="Import variation plan JSON"
            type="file"
            accept="application/json"
            className="hidden"
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) {
                void readFileText(file)
                  .then(onImportText)
                  .catch((error: unknown) => onImportError((error as Error).message));
              }
              event.target.value = "";
            }}
          />
        </label>
      </div>
      {progress && (
        <div className="mt-3">
          <progress
            aria-label="Variation execution progress"
            className="h-2 w-full accent-sky-400"
            max={progress.totalRuns}
            value={progress.completedRuns}
          />
          <p className="mt-1 text-xs tabular-nums text-slate-500">
            {progress.completedRuns}/{progress.totalRuns} evaluated runs complete
          </p>
        </div>
      )}
      <p
        role={visualState.announcementRole}
        aria-label="Variation status"
        aria-live={visualState.announcementRole === "alert" ? "assertive" : "polite"}
        className="mt-3 text-xs text-slate-400"
      >
        {status}
      </p>
    </div>
  );
}
