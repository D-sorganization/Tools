import type { VariationAnalysisExecution } from "./variationAnalysisPolicy";
import { oneAtATimeSensitivity, type SensitivityResultTs } from "./variationAnalysis";
import { runVariation, type VariationDatasetTs, type VariationPlanTs } from "./variation";
import {
  runSwingVariation,
  type SwingVariationResultTs,
} from "./variationSwingEnsemble";

export type VariationExecutionPhase = "joint" | "individual";

export interface VariationExecutionProgress {
  completedRuns: number;
  totalRuns: number;
  phase: VariationExecutionPhase;
}

export interface VariationExecutionRequest {
  plan: VariationPlanTs;
  analysisExecution: VariationAnalysisExecution;
}

export interface VariationExecutionResult {
  dataset: VariationDatasetTs | null;
  sensitivity: SensitivityResultTs | null;
  ensemble: SwingVariationResultTs | null;
}

export interface VariationExecutionControls {
  signal: AbortSignal;
  onProgress: (progress: VariationExecutionProgress) => void;
}

export interface VariationExecutionService {
  execute(
    request: VariationExecutionRequest,
    controls: VariationExecutionControls,
  ): Promise<VariationExecutionResult>;
}

interface WorkerProgressMessage {
  kind: "progress";
  progress: VariationExecutionProgress;
}

interface WorkerResultMessage {
  kind: "result";
  result: VariationExecutionResult;
}

interface WorkerErrorMessage {
  kind: "error";
  message: string;
}

export type VariationWorkerResponse =
  | WorkerProgressMessage
  | WorkerResultMessage
  | WorkerErrorMessage;

export const plannedVariationRuns = (
  plan: VariationPlanTs,
  execution: VariationAnalysisExecution,
): number => {
  const jointRuns = execution === "individual" ? 0 : plan.nRuns;
  const individualRuns = execution === "all_together"
    ? 0
    : plan.nRuns * plan.noise.length;
  return jointRuns + individualRuns;
};

/** Run the exact existing algorithms while exposing only completed evaluations. */
export function executeVariationWork(
  request: VariationExecutionRequest,
  onProgress: (progress: VariationExecutionProgress) => void,
): VariationExecutionResult {
  const { plan, analysisExecution } = request;
  const totalRuns = plannedVariationRuns(plan, analysisExecution);
  let completedRuns = 0;
  const report = (phase: VariationExecutionPhase) => {
    completedRuns += 1;
    onProgress({ completedRuns, totalRuns, phase });
  };
  const runJoint = analysisExecution !== "individual";
  const runIndividual = analysisExecution !== "all_together";
  const ensemble = plan.mode === "swing" && runJoint
    ? runSwingVariation(plan, undefined, () => report("joint"))
    : null;
  const dataset = runJoint
    ? ensemble?.dataset ?? runVariation(plan, () => report("joint"))
    : null;
  const sensitivity = runIndividual
    ? oneAtATimeSensitivity(plan, () => report("individual"))
    : null;
  return { dataset, sensitivity, ensemble };
}

const abortError = (): DOMException =>
  new DOMException("Variation execution was cancelled.", "AbortError");

class InProcessVariationExecutionService implements VariationExecutionService {
  execute(
    request: VariationExecutionRequest,
    controls: VariationExecutionControls,
  ): Promise<VariationExecutionResult> {
    return Promise.resolve().then(() => {
      if (controls.signal.aborted) throw abortError();
      return executeVariationWork(request, controls.onProgress);
    });
  }
}

class WorkerVariationExecutionService implements VariationExecutionService {
  execute(
    request: VariationExecutionRequest,
    controls: VariationExecutionControls,
  ): Promise<VariationExecutionResult> {
    if (controls.signal.aborted) return Promise.reject(abortError());
    const worker = new Worker(
      new URL("./variationExecution.worker.ts", import.meta.url),
      { type: "module", name: "rate-variation-execution" },
    );
    return new Promise((resolve, reject) => {
      const finish = () => {
        controls.signal.removeEventListener("abort", cancel);
        worker.terminate();
      };
      const cancel = () => {
        finish();
        reject(abortError());
      };
      controls.signal.addEventListener("abort", cancel, { once: true });
      worker.onerror = (event) => {
        finish();
        reject(new Error(event.message || "Variation worker failed."));
      };
      worker.onmessage = (event: MessageEvent<VariationWorkerResponse>) => {
        const message = event.data;
        if (message.kind === "progress") {
          controls.onProgress(message.progress);
          return;
        }
        finish();
        if (message.kind === "result") resolve(message.result);
        else reject(new Error(message.message));
      };
      worker.postMessage(request);
    });
  }
}

/** Production browsers use a one-job worker; non-browser tests use the same authority inline. */
export const createVariationExecutionService = (): VariationExecutionService =>
  typeof Worker === "undefined"
    ? new InProcessVariationExecutionService()
    : new WorkerVariationExecutionService();
