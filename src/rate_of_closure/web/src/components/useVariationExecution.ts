import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { VariationAnalysisExecution } from "../model/variationAnalysisPolicy";
import type { VariationDatasetTs, VariationPlanTs } from "../model/variation";
import type { SensitivityResultTs } from "../model/variationAnalysis";
import type { SwingVariationResultTs } from "../model/variationSwingEnsemble";
import {
  createVariationExecutionService,
  plannedVariationRuns,
  type VariationExecutionProgress,
  type VariationExecutionResult,
  type VariationExecutionService,
} from "../model/variationExecutionService";

interface VariationExecutionState {
  dataset: VariationDatasetTs | null;
  sensitivity: SensitivityResultTs | null;
  ensemble: SwingVariationResultTs | null;
  status: string;
  setStatus: (status: string) => void;
  busy: boolean;
  progress: VariationExecutionProgress | null;
  run: () => Promise<void>;
  cancel: () => void;
  invalidateResults: () => void;
}

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
  const generation = useRef(0);
  const activeController = useRef<AbortController | null>(null);

  const clearResultState = useCallback(() => {
    setDataset(null);
    setSensitivity(null);
    setEnsemble(null);
  }, []);

  const invalidateResults = useCallback(() => {
    generation.current += 1;
    activeController.current?.abort();
    activeController.current = null;
    setBusy(false);
    setProgress(null);
    clearResultState();
    setStatus("Ready: configuration changed; run again.");
  }, [clearResultState]);

  const cancel = useCallback(() => {
    if (activeController.current === null) return;
    invalidateResults();
    setStatus("Cancelled: no partial variation result was accepted.");
  }, [invalidateResults]);

  const run = useCallback(async () => {
    const currentGeneration = generation.current + 1;
    generation.current = currentGeneration;
    activeController.current?.abort();
    const controller = new AbortController();
    activeController.current = controller;
    clearResultState();
    setBusy(true);
    const initialProgress: VariationExecutionProgress = {
      completedRuns: 0,
      totalRuns: plannedVariationRuns(plan, analysisExecution),
      phase: analysisExecution === "individual" ? "individual" : "joint",
    };
    setProgress(initialProgress);
    setStatus(runningStatus(initialProgress));
    try {
      const result = await service.execute(
        { plan, analysisExecution },
        {
          signal: controller.signal,
          onProgress: (nextProgress) => {
            if (generation.current !== currentGeneration || controller.signal.aborted) return;
            setProgress(nextProgress);
            setStatus(runningStatus(nextProgress));
          },
        },
      );
      if (generation.current !== currentGeneration || controller.signal.aborted) return;
      setDataset(result.dataset);
      setSensitivity(result.sensitivity);
      setEnsemble(result.ensemble);
      setStatus(completionStatus(result, plan));
    } catch (error) {
      if (generation.current !== currentGeneration || controller.signal.aborted) return;
      setStatus(`Cannot run: ${(error as Error).message}`);
      setProgress(null);
    } finally {
      if (generation.current === currentGeneration) {
        activeController.current = null;
        setBusy(false);
      }
    }
  }, [analysisExecution, clearResultState, plan, service]);

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
    run,
    cancel,
    invalidateResults,
  };
}
