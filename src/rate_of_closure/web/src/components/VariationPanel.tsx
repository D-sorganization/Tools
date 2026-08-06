import { useState } from "react";

import type { TargetRegionTs } from "../model/targets";
import {
  planFromJson,
  planToJson,
  type VariationDatasetTs,
  type VariationPlanTs,
} from "../model/variation";
import {
  executeVariationAnalyses,
  type VariationAnalysisExecution,
} from "../model/variationAnalysisPolicy";
import type { SensitivityResultTs } from "../model/variationAnalysis";
import { oneAtATimeSensitivity } from "../model/variationAnalysis";
import {
  runSwingVariation,
  type SwingVariationResultTs,
} from "../model/variationSwingEnsemble";
import {
  deleteVariationPlan,
  duplicateVariationPlan,
  loadVariationPlanLibrary,
  saveVariationPlanLibrary,
  upsertVariationPlan,
  type NamedVariationPlan,
} from "../model/variationPlanLibrary";
import { VariationActions } from "./VariationActions";
import { VariationPlanLibraryPanel } from "./VariationPlanLibraryPanel";
import { VariationResults } from "./VariationResults";
import { VariationSetup } from "./VariationSetup";
import { defaultVariationPlan } from "./variationUi";
import { DRIVER_TEE_HEIGHT_M } from "../model/ballSetup";
import { loadBallSetupPreference } from "../model/ballSetupPersistence";

let generatedPlanId = 0;
const createPlanId = (): string => {
  generatedPlanId += 1;
  return `variation-plan-${Date.now()}-${generatedPlanId}`;
};

export interface VariationPanelProps {
  target?: TargetRegionTs;
  distanceUnit?: string;
  /** Injectable persistent storage for tests, embedded hosts, and privacy modes. */
  storage?: Storage;
}

export function VariationPanel({
  target,
  distanceUnit = "yd",
  storage,
}: VariationPanelProps = {}): JSX.Element {
  const [initialLibrary] = useState(() => loadVariationPlanLibrary(storage));
  const [initialBallSetup] = useState(() => loadBallSetupPreference(
    storage,
    { supportMode: "tee", teeHeightM: DRIVER_TEE_HEIGHT_M },
  ).setup);
  const [plan, setPlan] = useState<VariationPlanTs>(() => ({
    ...defaultVariationPlan(),
    ballSetup: initialBallSetup,
  }));
  const [analysisExecution, setAnalysisExecution] =
    useState<VariationAnalysisExecution>("both");
  const [dataset, setDataset] = useState<VariationDatasetTs | null>(null);
  const [sensitivity, setSensitivity] = useState<SensitivityResultTs | null>(null);
  const [ensemble, setEnsemble] = useState<SwingVariationResultTs | null>(null);
  const [library, setLibrary] = useState<NamedVariationPlan[]>(initialLibrary.plans);
  const [selectedId, setSelectedId] = useState("");
  const [planName, setPlanName] = useState("");
  const [status, setStatus] = useState(
    initialLibrary.warnings.length > 0 ? initialLibrary.warnings.join(" ") : "Ready.",
  );

  const clearResults = () => {
    setDataset(null);
    setSensitivity(null);
    setEnsemble(null);
  };

  const persistLibrary = (next: NamedVariationPlan[], message: string) => {
    try {
      saveVariationPlanLibrary(next, storage);
      setLibrary(next);
      setStatus(message);
    } catch (error) {
      setStatus(`Cannot update plan library: ${(error as Error).message}`);
    }
  };

  const run = () => {
    clearResults();
    try {
      const runTogether = analysisExecution !== "individual";
      const traceResult = plan.mode === "swing" && runTogether
        ? runSwingVariation(plan)
        : null;
      const result = executeVariationAnalyses(
        plan,
        analysisExecution,
        traceResult === null
          ? undefined
          : {
              runTogether: () => traceResult.dataset,
              runIndividually: oneAtATimeSensitivity,
            },
      );
      setDataset(result.dataset);
      setSensitivity(result.sensitivity);
      setEnsemble(traceResult);
      if (result.dataset) {
        const succeeded = result.dataset.success.filter(Boolean).length;
        const failed = plan.nRuns - succeeded;
        setStatus(
          `Done: ${succeeded}/${plan.nRuns} joint runs${failed ? ` (${failed} failed)` : ""}` +
            `${result.sensitivity ? "; one-at-a-time analysis also complete" : ""}.`,
        );
      } else {
        setStatus("Done: one-at-a-time analysis complete; joint analysis was not requested.");
      }
    } catch (error) {
      setStatus(`Cannot run: ${(error as Error).message}`);
    }
  };

  const importPlan = (text: string) => {
    try {
      const loaded = planFromJson(text);
      setPlan(loaded);
      clearResults();
      setStatus(`Plan loaded with ${loaded.noise.length} noise rows and ${loaded.groups?.length ?? 0} groups.`);
    } catch (error) {
      setStatus(`Cannot load plan: ${(error as Error).message}`);
    }
  };

  const saveNamedPlan = () => {
    const id = library.some((entry) => entry.id === selectedId)
      ? selectedId
      : createPlanId();
    const next = upsertVariationPlan(library, { id, name: planName.trim(), plan });
    persistLibrary(next, `Saved named plan “${planName.trim()}”.`);
    setSelectedId(id);
  };

  const loadSelectedPlan = () => {
    const selected = library.find((entry) => entry.id === selectedId);
    if (!selected) return;
    setPlan(planFromJson(planToJson(selected.plan)));
    setPlanName(selected.name);
    clearResults();
    setStatus(`Loaded named plan “${selected.name}”.`);
  };

  const duplicateSelectedPlan = () => {
    if (!selectedId) return;
    try {
      const duplicateId = createPlanId();
      const next = duplicateVariationPlan(library, selectedId, duplicateId);
      persistLibrary(next, "Duplicated selected named plan.");
      setSelectedId(duplicateId);
      setPlanName(next[next.length - 1].name);
    } catch (error) {
      setStatus(`Cannot duplicate plan: ${(error as Error).message}`);
    }
  };

  const deleteSelected = () => {
    if (!selectedId) return;
    const next = deleteVariationPlan(library, selectedId);
    persistLibrary(next, "Deleted selected named plan.");
    setSelectedId("");
    setPlanName("");
  };

  const selectLibraryPlan = (id: string) => {
    setSelectedId(id);
    setPlanName(library.find((entry) => entry.id === id)?.name ?? "");
  };

  return (
    <div className="grid gap-6 xl:grid-cols-[420px_1fr]">
      <section aria-label="Variation setup" className="space-y-4">
        <VariationSetup
          plan={plan}
          onPlanChange={setPlan}
          analysisExecution={analysisExecution}
          onAnalysisExecutionChange={(value) => {
            setAnalysisExecution(value);
            clearResults();
          }}
          onConfigurationChange={clearResults}
        />
        <VariationActions
          plan={plan}
          dataset={dataset}
          ensemble={ensemble}
          status={status}
          onRun={run}
          onImportText={importPlan}
          onImportError={(message) => setStatus(`Cannot read plan file: ${message}`)}
        />
        <VariationPlanLibraryPanel
          plans={library}
          selectedId={selectedId}
          name={planName}
          onSelectedIdChange={selectLibraryPlan}
          onNameChange={setPlanName}
          onSave={saveNamedPlan}
          onLoad={loadSelectedPlan}
          onDuplicate={duplicateSelectedPlan}
          onDelete={deleteSelected}
        />
      </section>
      <VariationResults
        dataset={dataset}
        sensitivity={sensitivity}
        target={target}
        distanceUnit={distanceUnit}
        ensemble={ensemble}
      />
    </div>
  );
}
