/** Versioned simulation run import/export controls with atomic settings import. */

import type { BallSetup } from "../model/ballSetup";
import {
  ballSetupFromSimulationDocument,
  createSimulationRunCsv,
  createSimulationRunDocument,
  spatialTargetFromSimulationDocument,
} from "../model/ballSetupPersistence";
import type { SimulationInput, SimulationRunTs } from "../model/simulation";
import type { SpatialTargetTs } from "../model/spatialTarget";
import {
  manualDeliveryFromSimulationDocument,
  type ManualDelivery,
} from "../model/manualDelivery";

interface ImportedSettings {
  readonly ballSetup: BallSetup;
  readonly spatialTarget: SpatialTargetTs;
  readonly manualDelivery: ManualDelivery;
}

interface Props {
  readonly input: SimulationInput;
  readonly run: SimulationRunTs | null;
  readonly prescribedTorqueProfile: unknown;
  readonly spatialTarget: SpatialTargetTs;
  readonly onImported: (settings: ImportedSettings) => void;
  readonly onImportError: (message: string) => void;
}

const BUTTON_CLASS =
  "rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-sm " +
  "text-slate-300 hover:border-slate-500 disabled:opacity-40";

const readFileText = (file: File): Promise<string> => {
  if (typeof file.text === "function") return file.text();
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result ?? ""));
    reader.onerror = () => reject(reader.error ?? new Error("Could not read file."));
    reader.readAsText(file);
  });
};

function download(filename: string, content: string, type: string): void {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

export function SimulationRunFileControls({
  input,
  run,
  prescribedTorqueProfile,
  spatialTarget,
  onImported,
  onImportError,
}: Props): JSX.Element {
  const exportJson = () => {
    if (!run) return;
    const document = createSimulationRunDocument(
      input,
      run,
      prescribedTorqueProfile,
      spatialTarget,
    );
    download("simulation_run.json", JSON.stringify(document, null, 2), "application/json");
  };
  const exportCsv = () => {
    if (run) download("simulation_run.csv", createSimulationRunCsv(run, spatialTarget), "text/csv");
  };
  const importJson = async (file?: File) => {
    if (!file) return;
    try {
      const document: unknown = JSON.parse(await readFileText(file));
      const settings = {
        ballSetup: ballSetupFromSimulationDocument(document),
        spatialTarget: spatialTargetFromSimulationDocument(document),
        manualDelivery: manualDeliveryFromSimulationDocument(document),
      };
      onImported(settings);
    } catch (error) {
      onImportError(error instanceof Error ? error.message : String(error));
    }
  };

  return (
    <>
      <button type="button" onClick={exportJson} disabled={!run}
        title="Download the full run and canonical spatial target as JSON"
        className={BUTTON_CLASS}>Export JSON</button>
      <button type="button" onClick={exportCsv} disabled={!run}
        title="Download trajectory rows and canonical spatial target metadata as CSV"
        className={BUTTON_CLASS}>Export CSV</button>
      <label
        className={`${BUTTON_CLASS} cursor-pointer`}
        title="Import ball setup, spatial target, and manual delivery from Simulation Settings JSON"
      >
        Import Settings JSON
        <input type="file" accept="application/json,.json"
          aria-label="Import Simulation Settings JSON" className="sr-only"
          onChange={(event) => {
            void importJson(event.target.files?.[0]);
            event.currentTarget.value = "";
          }} />
      </label>
    </>
  );
}
