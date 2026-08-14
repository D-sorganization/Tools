import { useState } from "react";

import type { SpatialTargetTs } from "../model/spatialTarget";
import { spatialTargetForGroundWorkflow } from "../model/spatialTargetWorkflow";
import {
  planFromJson,
  planToJson,
  type VariationDatasetTs,
} from "../model/variation";
import { executeVariationAnalyses } from "../model/variationAnalysisPolicy";
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
import { DRIVER_TEE_HEIGHT_M, type BallSetup } from "../model/ballSetup";
import { loadBallSetupPreference } from "../model/ballSetupPersistence";
import { TEE_HEIGHT_VARIATION_KEY } from "../model/variationRegistry";
import { spatialTargetSummary } from "./spatialTargetPresentation";
import {
  useVariationWorkspace,
  type ControlledVariationWorkspaceProps,
} from "../hooks/useVariationWorkspace";

let generatedPlanId = 0;
const createPlanId = (): string => {
  generatedPlanId += 1;
  return `variation-plan-${Date.now()}-${generatedPlanId}`;
};

export interface VariationPanelProps extends ControlledVariationWorkspaceProps {
  spatialTarget?: SpatialTargetTs;
  distanceUnit?: string;
  ballSetup?: BallSetup;
  /** Injectable persistent storage for tests, embedded hosts, and privacy modes. */
  storage?: Storage;
}

export function VariationPanel({
  spatialTarget,
  distanceUnit = "yd",
  ballSetup,
  storage,
  variationWorkspace,
  onVariationWorkspaceChange,
}: VariationPanelProps = {}): JSX.Element {
  const targetUse = spatialTarget
    ? spatialTargetForGroundWorkflow(spatialTarget, "variation")
    : { targetRegion: null, diagnostic: null };
  const [initialLibrary] = useState(() => loadVariationPlanLibrary(storage));
  const [storedBallSetup] = useState(
    () =>
      loadBallSetupPreference(storage, {
        supportMode: "tee",
        teeHeightM: DRIVER_TEE_HEIGHT_M,
      }).setup,
  );
  const activeBallSetup = ballSetup ?? storedBallSetup;
  const variation = useVariationWorkspace(
    { variationWorkspace, onVariationWorkspaceChange },
    activeBallSetup,
  );
  const {
    plan: persistedPlan,
    analysisExecution,
    selectedOutputMetrics,
  } = variation.state;
  const plan = { ...persistedPlan, ballSetup: activeBallSetup };
  const portablePlan = plan.noise.some(
    (spec) => spec.variableKey === TEE_HEIGHT_VARIATION_KEY,
  )
    ? plan
    : persistedPlan;
  const [dataset, setDataset] = useState<VariationDatasetTs | null>(null);
  const [sensitivity, setSensitivity] = useState<SensitivityResultTs | null>(
    null,
  );
  const [ensemble, setEnsemble] = useState<SwingVariationResultTs | null>(null);
  const [library, setLibrary] = useState<NamedVariationPlan[]>(
    initialLibrary.plans,
  );
  const [selectedId, setSelectedId] = useState("");
  const [planName, setPlanName] = useState("");
  const [status, setStatus] = useState(
    initialLibrary.warnings.length > 0
      ? initialLibrary.warnings.join(" ")
      : "Ready.",
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
      const traceResult =
        plan.mode === "swing" && runTogether ? runSwingVariation(plan) : null;
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
        setStatus(
          "Done: one-at-a-time analysis complete; joint analysis was not requested.",
        );
      }
    } catch (error) {
      setStatus(`Cannot run: ${(error as Error).message}`);
    }
  };

  const importPlan = (text: string) => {
    try {
      const loaded = planFromJson(text);
      variation.setPlan(loaded);
      clearResults();
      setStatus(
        `Plan loaded with ${loaded.noise.length} noise rows and ${loaded.groups?.length ?? 0} groups.`,
      );
    } catch (error) {
      setStatus(`Cannot load plan: ${(error as Error).message}`);
    }
  };

  const saveNamedPlan = () => {
    const id = library.some((entry) => entry.id === selectedId)
      ? selectedId
      : createPlanId();
    const next = upsertVariationPlan(library, {
      id,
      name: planName.trim(),
      plan: portablePlan,
    });
    persistLibrary(next, `Saved named plan “${planName.trim()}”.`);
    setSelectedId(id);
  };

  const loadSelectedPlan = () => {
    const selected = library.find((entry) => entry.id === selectedId);
    if (!selected) return;
    variation.setPlan(planFromJson(planToJson(selected.plan)));
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
        {spatialTarget && (
          <p
            role="status"
            aria-label="Variation current spatial target"
            className={`rounded-lg border p-3 text-xs ${
              targetUse.diagnostic
                ? "border-amber-400/40 bg-amber-950/20 text-amber-200"
                : "border-sky-400/30 bg-sky-950/20 text-sky-200"
            }`}
          >
            Current target: {spatialTargetSummary(spatialTarget)}
            {targetUse.diagnostic ? ` ${targetUse.diagnostic.message}` : ""}
          </p>
        )}
        <VariationSetup
          plan={plan}
          onPlanChange={variation.setPlan}
          analysisExecution={analysisExecution}
          onAnalysisExecutionChange={(value) => {
            variation.setAnalysisExecution(value);
            clearResults();
          }}
          selectedOutputMetrics={selectedOutputMetrics}
          onSelectedOutputMetricsChange={(metrics) => {
            variation.setSelectedOutputMetrics(metrics);
            clearResults();
          }}
          onConfigurationChange={clearResults}
        />
        <VariationActions
          plan={portablePlan}
          dataset={dataset}
          ensemble={ensemble}
          status={status}
          onRun={run}
          onImportText={importPlan}
          onImportError={(message) =>
            setStatus(`Cannot read plan file: ${message}`)
          }
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
        target={targetUse.targetRegion ?? undefined}
        distanceUnit={distanceUnit}
        ensemble={ensemble}
      />
    </div>
  );
}
