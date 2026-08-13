import type { VariationAnalysisExecution } from "./variationAnalysisPolicy";
import { oneAtATimeSensitivity, type SensitivityResultTs } from "./variationAnalysis";
import {
  runVariation,
  type VariationDatasetTs,
  type VariationPlanTs,
} from "./variation";
import {
  runSwingVariation,
  type SwingVariationResultTs,
} from "./variationSwingEnsemble";
import {
  failProtocol,
  isRecord,
  MAX_WORKER_ERROR_LENGTH,
  validateExecutionRequest,
  validateResult,
  workerError,
} from "./variationExecutionValidation";
import {
  makeVariationExecutionMetadata,
  type VariationExecutionMetadataTs,
} from "./variationExecutionMetadata";

export type VariationExecutionPhase = "joint" | "individual";

export interface VariationExecutionProgress {
  completedRuns: number;
  totalRuns: number;
  phase: VariationExecutionPhase;
}

export interface VariationExecutionRequest {
  plan: VariationPlanTs;
  analysisExecution: VariationAnalysisExecution;
  executionMetadata: VariationExecutionMetadataTs;
}

export interface VariationExecutionResult {
  dataset: VariationDatasetTs | null;
  sensitivity: SensitivityResultTs | null;
  ensemble: SwingVariationResultTs | null;
  executionMetadata: VariationExecutionMetadataTs;
}

export const prepareVariationExecutionRequest = (
  plan: VariationPlanTs,
  analysisExecution: VariationAnalysisExecution,
): VariationExecutionRequest => ({
  plan,
  analysisExecution,
  executionMetadata: makeVariationExecutionMetadata(plan),
});

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

export type VariationWorkerFactory = () => Worker;

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
  validateExecutionRequest(request);
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
  return { dataset, sensitivity, ensemble, executionMetadata: request.executionMetadata };
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

class WorkerExecutionJob {
  private readonly totalRuns: number;
  private completedRuns = 0;
  private settled = false;
  private resolve!: (result: VariationExecutionResult) => void;
  private reject!: (error: Error | DOMException) => void;

  constructor(
    private readonly worker: Worker,
    private readonly request: VariationExecutionRequest,
    private readonly controls: VariationExecutionControls,
  ) {
    this.totalRuns = plannedVariationRuns(request.plan, request.analysisExecution);
  }

  start(): Promise<VariationExecutionResult> {
    const promise = new Promise<VariationExecutionResult>((resolve, reject) => {
      this.resolve = resolve;
      this.reject = reject;
    });
    this.controls.signal.addEventListener("abort", this.cancel, { once: true });
    this.worker.onerror = this.handleError;
    this.worker.onmessageerror = this.handleMessageError;
    this.worker.onmessage = this.handleMessage;
    if (this.controls.signal.aborted) this.cancel();
    else this.postRequest();
    return promise;
  }

  private readonly cancel = () => this.rejectOnce(abortError());

  private readonly handleError = (event: ErrorEvent) => {
    this.rejectOnce(new Error(event.message || "Variation worker failed."));
  };

  private readonly handleMessageError = () => {
    this.rejectOnce(new Error("Variation worker response could not be decoded."));
  };

  private readonly handleMessage = (event: MessageEvent<unknown>) => {
    if (this.settled) return;
    try {
      const message = event.data;
      if (!isRecord(message) || typeof message.kind !== "string") {
        return failProtocol("message");
      }
      if (message.kind === "progress") return this.acceptProgress(message.progress);
      if (message.kind === "result") return this.acceptResult(message.result);
      if (message.kind === "error" && typeof message.message === "string") {
        return this.rejectOnce(new Error(message.message.slice(0, MAX_WORKER_ERROR_LENGTH)));
      }
      return failProtocol("message kind");
    } catch (error) {
      this.rejectOnce(error);
    }
  };

  private acceptProgress(value: unknown): void {
    if (!isRecord(value)) return failProtocol("progress message");
    const expectedCompleted = this.completedRuns + 1;
    const jointRuns = this.request.analysisExecution === "individual" ? 0 : this.request.plan.nRuns;
    const expectedPhase = expectedCompleted <= jointRuns ? "joint" : "individual";
    if (value.completedRuns !== expectedCompleted
        || value.totalRuns !== this.totalRuns || value.phase !== expectedPhase) {
      return failProtocol("progress sequence");
    }
    this.completedRuns = expectedCompleted;
    this.controls.onProgress(value as unknown as VariationExecutionProgress);
  }

  private acceptResult(value: unknown): void {
    if (this.completedRuns !== this.totalRuns) return failProtocol("result progress count");
    this.resolveOnce(validateResult(value, this.request));
  }

  private postRequest(): void {
    try {
      this.worker.postMessage(this.request);
    } catch (error) {
      this.rejectOnce(error);
    }
  }

  private resolveOnce(result: VariationExecutionResult): void {
    if (this.settled) return;
    this.settled = true;
    this.cleanup();
    this.resolve(result);
  }

  private rejectOnce(error: unknown): void {
    if (this.settled) return;
    this.settled = true;
    this.cleanup();
    this.reject(workerError(error));
  }

  private cleanup(): void {
    this.controls.signal.removeEventListener("abort", this.cancel);
    this.worker.onmessage = null;
    this.worker.onmessageerror = null;
    this.worker.onerror = null;
    this.worker.terminate();
  }
}

class WorkerVariationExecutionService implements VariationExecutionService {
  constructor(private readonly workerFactory: VariationWorkerFactory) {}

  execute(
    request: VariationExecutionRequest,
    controls: VariationExecutionControls,
  ): Promise<VariationExecutionResult> {
    validateExecutionRequest(request);
    if (controls.signal.aborted) return Promise.reject(abortError());
    try {
      return new WorkerExecutionJob(this.workerFactory(), request, controls).start();
    } catch (error) {
      return Promise.reject(workerError(error));
    }
  }
}

/** Production browsers use a one-job worker; non-browser tests use the same authority inline. */
export const createVariationExecutionService = (
  workerFactory?: VariationWorkerFactory,
): VariationExecutionService => {
  if (workerFactory !== undefined) return new WorkerVariationExecutionService(workerFactory);
  if (typeof Worker === "undefined") return new InProcessVariationExecutionService();
  return new WorkerVariationExecutionService(() => new Worker(
    new URL("./variationExecution.worker.ts", import.meta.url),
    { type: "module", name: "rate-variation-execution" },
  ));
};
