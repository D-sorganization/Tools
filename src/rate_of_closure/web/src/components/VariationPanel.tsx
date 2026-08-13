import { useRef, useState } from "react";

import type { SpatialTargetTs } from "../model/spatialTarget";
import { spatialTargetForGroundWorkflow } from "../model/spatialTargetWorkflow";
import {
  planFromJson,
  planToJson,
  type VariationPlanTs,
} from "../model/variation";
import type { VariationAnalysisExecution } from "../model/variationAnalysisPolicy";
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
import type { MorrisAuthorityClient } from "../model/morrisAuthorityClient";
import {
  morrisAuthorityBaseIdentity,
  type MorrisAuthorityBase,
} from "../model/morrisAuthorityRequest";
import { MorrisWorkflowPanel } from "./MorrisWorkflowPanel";
import { VariationSetup } from "./VariationSetup";
import { BUTTON_CLASS, defaultVariationPlan } from "./variationUi";
import { DRIVER_TEE_HEIGHT_M } from "../model/ballSetup";
import { loadBallSetupPreference } from "../model/ballSetupPersistence";
import { spatialTargetSummary } from "./spatialTargetPresentation";
import type { VariationExecutionService } from "../model/variationExecutionService";
import { useVariationExecution } from "./useVariationExecution";
import { scheduleMeaningfulVisualReveal } from "../model/variationVisualProminence";

let generatedPlanId = 0;
const createPlanId = (): string => {
  generatedPlanId += 1;
  return `variation-plan-${Date.now()}-${generatedPlanId}`;
};

export interface VariationPanelProps {
  spatialTarget?: SpatialTargetTs;
  distanceUnit?: string;
  /** Injectable persistent storage for tests, embedded hosts, and privacy modes. */
  storage?: Storage;
  /** App-owned authority dependency. Null keeps static builds fail-closed. */
  morrisClient?: MorrisAuthorityClient | null;
  /** Current simulation context serialized by the strict shared contract. */
  morrisBase?: MorrisAuthorityBase;
  /** Honest fail-closed reason when current app context cannot round-trip. */
  morrisUnavailableReason?: string;
  /** Injectable bounded execution authority for tests and embedded hosts. */
  executionService?: VariationExecutionService;
}

export function VariationPanel({
  spatialTarget,
  distanceUnit = "yd",
  storage,
  morrisClient = null,
  morrisBase,
  morrisUnavailableReason,
  executionService,
}: VariationPanelProps = {}): JSX.Element {
  const [workflow, setWorkflow] = useState<"variation" | "morris">("variation");
  const targetUse = spatialTarget
    ? spatialTargetForGroundWorkflow(spatialTarget, "variation")
    : { targetRegion: null, diagnostic: null };
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
  const [library, setLibrary] = useState<NamedVariationPlan[]>(initialLibrary.plans);
  const [selectedId, setSelectedId] = useState("");
  const [planName, setPlanName] = useState("");
  const actionsRef = useRef<HTMLSpanElement>(null);
  const prominenceRef = useRef<HTMLElement>(null);
  const runButtonRef = useRef<HTMLButtonElement>(null);
  const execution = useVariationExecution(
    plan,
    analysisExecution,
    initialLibrary.warnings.length > 0 ? initialLibrary.warnings.join(" ") : "Ready.",
    executionService,
  );
  const {
    dataset,
    sensitivity,
    ensemble,
    status,
    setStatus,
    busy,
    progress,
    visualState,
    run,
    cancel,
    invalidateResults: clearResults,
  } = execution;

  const selectWorkflow = (value: "variation" | "morris") => {
    if (value !== workflow) clearResults();
    setWorkflow(value);
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
    try {
      const id = library.some((entry) => entry.id === selectedId)
        ? selectedId
        : createPlanId();
      const next = upsertVariationPlan(library, { id, name: planName.trim(), plan });
      persistLibrary(next, `Saved named plan “${planName.trim()}”.`);
      setSelectedId(id);
    } catch (error) {
      setStatus(`Cannot save plan: ${(error as Error).message}`);
    }
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

  if (workflow === "morris" && morrisBase !== undefined) {
    return <div className="space-y-4">
      <VariationWorkflowPicker value={workflow} onChange={selectWorkflow} />
      <MorrisWorkflowPanel key={morrisAuthorityBaseIdentity(morrisBase)}
        client={morrisClient} base={morrisBase} />
    </div>;
  }
  return (
    <div className="space-y-4">
      <VariationWorkflowPicker value={workflow} onChange={selectWorkflow}
        morrisDisabled={morrisBase === undefined} morrisUnavailableReason={morrisUnavailableReason} />
      <div className="grid gap-6 xl:grid-cols-[420px_1fr]">
      <section aria-label="Variation setup" className="order-2 space-y-4 xl:order-none">
        {spatialTarget && (
          <p role="status" aria-label="Variation current spatial target"
            className={`rounded-lg border p-3 text-xs ${targetUse.diagnostic
              ? "border-amber-400/40 bg-amber-950/20 text-amber-200"
              : "border-sky-400/30 bg-sky-950/20 text-sky-200"}`}>
            Current target: {spatialTargetSummary(spatialTarget)}
            {targetUse.diagnostic ? ` ${targetUse.diagnostic.message}` : ""}
          </p>
        )}
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
          busy={busy}
          progress={progress}
          visualState={visualState}
          onRun={(allowAutomaticReveal) => void run().then((outcome) => {
            if (outcome === "accepted" && allowAutomaticReveal) {
              scheduleMeaningfulVisualReveal(() => prominenceRef.current);
            }
          })}
          onCancel={cancel}
          onImportText={importPlan}
          onImportError={(message) => setStatus(`Cannot read plan file: ${message}`)}
          runButtonRef={runButtonRef}
          actionsRef={actionsRef}
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
      <div className="order-first min-w-0 xl:order-none"><VariationResults
        dataset={dataset}
        sensitivity={sensitivity}
        target={targetUse.targetRegion ?? undefined}
        distanceUnit={distanceUnit}
        ensemble={ensemble}
        visualState={visualState}
        visualAnnouncement={status}
        prominenceRef={prominenceRef}
        onReturnToControls={(focusRun) => {
          actionsRef.current?.scrollIntoView({
            behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches
            ? "auto" : "smooth",
            block: "center",
          });
          if (focusRun) runButtonRef.current?.focus({ preventScroll: true });
        }}
      /></div>
      </div>
    </div>
  );
}

function VariationWorkflowPicker(props: {
  readonly value: "variation" | "morris";
  readonly onChange: (value: "variation" | "morris") => void;
  readonly morrisDisabled?: boolean;
  readonly morrisUnavailableReason?: string;
}) {
  return <nav aria-label="Variation workflows" className="flex flex-wrap gap-2">
    <button type="button" aria-pressed={props.value === "variation"} className={BUTTON_CLASS}
      title="Run seeded Monte Carlo variation and one-at-a-time analyses in the browser"
      onClick={() => props.onChange("variation")}>Monte Carlo Variation</button>
    <button type="button" aria-pressed={props.value === "morris"} className={BUTTON_CLASS}
      title={props.morrisUnavailableReason ?? "Run global Morris elementary-effects screening in the local Python authority"}
      disabled={props.morrisDisabled} onClick={() => props.onChange("morris")}>Morris Screening</button>
    {props.morrisDisabled && props.morrisUnavailableReason && <span role="status" aria-label="Morris availability"
      className="self-center text-xs text-amber-300">
      Morris unavailable: {props.morrisUnavailableReason}
    </span>}
  </nav>;
}
