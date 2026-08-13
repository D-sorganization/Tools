import { datasetToCsv, datasetToJson } from "../model/variationAnalysis";
import { planToJson, type VariationDatasetTs, type VariationPlanTs } from "../model/variation";
import { BUTTON_CLASS, PANEL_CLASS, downloadText } from "./variationUi";
import {
  swingEnsembleToJson,
  swingTracesToCsv,
  type SwingVariationResultTs,
} from "../model/variationSwingEnsemble";

interface VariationActionsProps {
  plan: VariationPlanTs;
  dataset: VariationDatasetTs | null;
  ensemble: SwingVariationResultTs | null;
  status: string;
  onRun: () => void;
  onImportText: (text: string) => void;
  onImportError: (message: string) => void;
}

const readFileText = (file: File): Promise<string> =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.addEventListener("load", () => resolve(String(reader.result ?? "")));
    reader.addEventListener("error", () => reject(reader.error ?? new Error("File read failed")));
    reader.readAsText(file);
  });

export function VariationActions({
  plan,
  dataset,
  ensemble,
  status,
  onRun,
  onImportText,
  onImportError,
}: VariationActionsProps): JSX.Element {
  return (
    <div className={PANEL_CLASS}>
      <div className="flex flex-wrap gap-2">
        <button
          type="button"
          onClick={onRun}
          title="Run only the analyses selected in Analysis Execution."
          className={`${BUTTON_CLASS} border-sky-500/60 text-sky-300`}
        >
          Run Variation Study
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
          onClick={() => downloadText(
            "variation_plan.json",
            planToJson(plan),
            "application/json",
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
      <p
        role="status"
        aria-label="Variation status"
        aria-live="polite"
        className="mt-3 text-xs text-slate-400"
      >
        {status}
      </p>
    </div>
  );
}
