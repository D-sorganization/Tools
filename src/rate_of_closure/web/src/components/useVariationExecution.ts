import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { VariationAnalysisExecution } from "../model/variationAnalysisPolicy";
import type { VariationDatasetTs, VariationPlanTs } from "../model/variation";
import type { SensitivityResultTs } from "../model/variationAnalysis";
import type { SwingVariationResultTs } from "../model/variationSwingEnsemble";
import {
  createVariationExecutionService,
  plannedVariationRuns,
  prepareVariationExecutionRequest,
  type VariationExecutionProgress,
  type VariationExecutionResult,
  type VariationExecutionService,
} from "../model/variationExecutionService";
import {
  MAX_WORKER_ERROR_LENGTH,
  validateExecutionRequest,
  validateResult,
} from "../model/variationExecutionValidation";
import {
  variationExecutionIdentity,
  variationVisualState,
  type VariationVisualState,
} from "../model/variationVisualState";

interface VariationExecutionState {
  dataset: VariationDatasetTs | null;
  sensitivity: SensitivityResultTs | null;
  ensemble: SwingVariationResultTs | null;
  status: string;
  setStatus: (status: string) => void;
  busy: boolean;
  progress: VariationExecutionProgress | null;
  visualState: VariationVisualState;
  run: () => Promise<VariationRunOutcome>;
  cancel: () => void;
  invalidateResults: () => void;
}

export type VariationRunOutcome = "accepted" | "failed" | "stale";

const completionStatus = (
  result: VariationExecutionResult,
  plan: VariationPlanTs,
): string => {
  if (result.dataset === null) {
    return "Done: one-at-a-time analysis complete; joint analysis was not requested.";
  }
  const succeeded = result.dataset.success.filter(Boolean).length;
  const failed = plan.nRuns - succeeded;
  return `Done: ${succeeded}/${plan.nRuns} joint runs${failed ? ` (${failed} failed)` : ""}` +
    `${result.sensitivity ? "; one-at-a-time analysis also complete" : ""}.`;
};

const runningStatus = (progress: VariationExecutionProgress): string => {
  const label = progress.phase === "joint" ? "joint variation" : "one-at-a-time variation";
  return `Running ${label}: ${progress.completedRuns}/${progress.totalRuns} evaluated runs.`;
};

export function useVariationExecution(
  plan: VariationPlanTs,
  analysisExecution: VariationAnalysisExecution,
  initialStatus: string,
  serviceOverride?: VariationExecutionService,
): VariationExecutionState {
  const service = useMemo(
    () => serviceOverride ?? createVariationExecutionService(),
    [serviceOverride],
  );
  const [dataset, setDataset] = useState<VariationDatasetTs | null>(null);
  const [sensitivity, setSensitivity] = useState<SensitivityResultTs | null>(null);
  const [ensemble, setEnsemble] = useState<SwingVariationResultTs | null>(null);
  const [status, setStatus] = useState(initialStatus);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState<VariationExecutionProgress | null>(null);
  const [visualState, setVisualState] = useState(() => variationVisualState("invalidate"));
  const generation = useRef(0);
  const activeController = useRef<AbortController | null>(null);
  const acceptedIdentity = useRef<string | null>(null);
  const serviceIdentity = useRef(service);
  const configurationIdentity = variationExecutionIdentity(plan, analysisExecution);
  const previousConfigurationIdentity = useRef(configurationIdentity);

  const clearResultState = useCallback(() => {
    setDataset(null);
    setSensitivity(null);
    setEnsemble(null);
  }, []);

  const invalidateResults = useCallback(() => {
    // Only announce a discard that actually happened. This runs from an effect
    // on every configuration-identity change, including the one caused by
    // loading a plan, so announcing unconditionally would overwrite the
    // caller's own message -- e.g. the import provenance warning, which is the
    // only place the user is told a plan resolved against the current registry
    // rather than its recorded one. Refs, so the callback identity is stable
    // and the effect that calls this cannot re-trigger itself.
    const discarded =
      acceptedIdentity.current !== null || activeController.current !== null;
    generation.current += 1;
    activeController.current?.abort();
    activeController.current = null;
    setDataset(null);
    setEnsemble(null);
    clearResultState();
    acceptedIdentity.current = null;
    setVisualState(variationVisualState("invalidate"));
    if (discarded) setStatus("Ready: configuration changed; run again.");
  }, [clearResultState]);

  const cancel = useCallback(() => {
    if (activeController.current === null) return;
    const retainsAccepted = acceptedIdentity.current === configurationIdentity;
    const acceptedDataset = dataset;
    const acceptedSensitivity = sensitivity;
    const acceptedEnsemble = ensemble;
    invalidateResults();
    if (retainsAccepted) {
      setDataset(acceptedDataset);
      setSensitivity(acceptedSensitivity);
      setEnsemble(acceptedEnsemble);
      acceptedIdentity.current = configurationIdentity;
    }
    setBusy(false);
    setProgress(null);
    setStatus("Cancelled: no partial variation result was accepted.");
    setVisualState(variationVisualState(
      retainsAccepted ? "cancel-retained" : "cancel-empty",
    ));
  }, [configurationIdentity, dataset, ensemble, invalidateResults, sensitivity]);

  const run = useCallback(async () => {
    const currentGeneration = generation.current + 1;
    generation.current = currentGeneration;
    activeController.current?.abort();
    const controller = new AbortController();
    activeController.current = controller;
    const retainsAccepted = acceptedIdentity.current === configurationIdentity;
    setVisualState(variationVisualState(retainsAccepted ? "start-retained" : "start-empty"));
    setBusy(true);
    const initialProgress: VariationExecutionProgress = {
      completedRuns: 0,
      totalRuns: plannedVariationRuns(plan, analysisExecution),
      phase: analysisExecution === "individual" ? "individual" : "joint",
    };
    setProgress(initialProgress);
    setStatus(runningStatus(initialProgress));
    try {
      // The prepared request carries the execution metadata that both the
      // request and the returned result are validated against.
      const request = prepareVariationExecutionRequest(plan, analysisExecution);
      validateExecutionRequest(request);
      const result = validateResult(await service.execute(
        request,
        {
          signal: controller.signal,
          onProgress: (nextProgress) => {
            if (generation.current !== currentGeneration || controller.signal.aborted) return;
            setProgress(nextProgress);
            setStatus(runningStatus(nextProgress));
          },
        },
      ), request);
      if (generation.current !== currentGeneration || controller.signal.aborted) return "stale";
      setDataset(result.dataset);
      setSensitivity(result.sensitivity);
      setEnsemble(result.ensemble);
      acceptedIdentity.current = configurationIdentity;
      setVisualState(variationVisualState("succeed"));
      setStatus(completionStatus(result, plan));
      return "accepted";
    } catch (error) {
      if (generation.current !== currentGeneration || controller.signal.aborted) return "stale";
      const message = String((error as Error).message ?? error)
        .slice(0, MAX_WORKER_ERROR_LENGTH);
      setStatus(`Cannot run: ${message}`);
      setProgress(null);
      setVisualState(variationVisualState(
        retainsAccepted ? "fail-retained" : "fail-empty",
      ));
      return "failed";
    } finally {
      if (generation.current === currentGeneration) {
        activeController.current = null;
        setBusy(false);
      }
    }
  }, [analysisExecution, configurationIdentity, plan, service]);

  useEffect(() => {
    const configurationChanged = previousConfigurationIdentity.current !== configurationIdentity;
    const serviceChanged = serviceIdentity.current !== service;
    previousConfigurationIdentity.current = configurationIdentity;
    serviceIdentity.current = service;
    if (configurationChanged || serviceChanged) invalidateResults();
  }, [configurationIdentity, invalidateResults, service]);

  useEffect(() => () => {
    generation.current += 1;
    activeController.current?.abort();
    activeController.current = null;
  }, []);

  return {
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
    invalidateResults,
  };
}
