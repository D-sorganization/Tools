/** One-job Worker transport for the separate paired attribution experiment. */

import { attributionAuthorityFromValue, attributionAuthorityToValue } from "./localizedAttribution";
import { isRecord, MAX_WORKER_ERROR_LENGTH, workerError } from "./variationExecutionValidation";
import { resolvedBase } from "./variationSampling";
import { stableSpecId } from "./variationSchema";
import {
  executeLocalizedPairedWork, localizedAuthorityFromEvidence, normalizePairedRequest, pairedDesignIdentity,
  pairedRequestIdentity, pairedSourcePlan, PAIRED_RESULT_SCHEMA_ID,
  type LocalizedPairedProgressTs, type LocalizedPairedRequestTs,
  type LocalizedPairedResultTs,
} from "./localizedAttributionExecution";
import { SWING_VARIATION_OUTPUT_NAMES } from "./variationSwingEnsemble";

export interface LocalizedPairedControlsTs {
  signal: AbortSignal;
  onProgress: (progress: LocalizedPairedProgressTs) => void;
}
export interface LocalizedPairedExecutionServiceTs {
  execute(
    request: LocalizedPairedRequestTs,
    controls: LocalizedPairedControlsTs,
  ): Promise<LocalizedPairedResultTs>;
}
export type LocalizedPairedWorkerFactory = () => Worker;

const abortError = (): DOMException =>
  new DOMException("Localized paired study was cancelled.", "AbortError");
const fail = (label: string): never => { throw new Error(`Invalid localized paired ${label}.`); };

export async function validateLocalizedPairedResult(
  value: unknown,
  request: LocalizedPairedRequestTs,
): Promise<LocalizedPairedResultTs> {
  if (!isRecord(value) || Object.keys(value).sort().join("|") !==
      ["schemaId", "schemaVersion", "requestIdentity", "designIdentity", "authority",
        "explicitRows", "trials"].sort().join("|")) {
    return fail("result fields");
  }
  if (value.schemaId !== PAIRED_RESULT_SCHEMA_ID || value.schemaVersion !== 1 ||
      typeof value.requestIdentity !== "string" || typeof value.designIdentity !== "string") {
    return fail("result schema");
  }
  const normalizedRequest = normalizePairedRequest(request);
  const requestIdentity = await pairedRequestIdentity(normalizedRequest);
  const designIdentity = await pairedDesignIdentity(normalizedRequest, requestIdentity);
  if (value.requestIdentity !== requestIdentity || value.designIdentity !== designIdentity) {
    return fail("result identity");
  }
  const authority = attributionAuthorityFromValue(
    attributionAuthorityToValue(value.authority as never),
  );
  if (authority.authorityId !== `paired-attribution.${designIdentity}`) {
    return fail("authority identity");
  }
  const plan = pairedSourcePlan(normalizedRequest);
  const base = resolvedBase(plan);
  if (!Array.isArray(value.explicitRows) || value.explicitRows.length !== plan.noise.length * 2 ||
      !Array.isArray(value.trials) || value.trials.length !== value.explicitRows.length) {
    return fail("execution evidence shape");
  }
  const explicitRows = value.explicitRows.map((row, rowIndex) => {
    if (!Array.isArray(row) || row.length !== plan.noise.length ||
        row.some((item) => typeof item !== "number" || !Number.isFinite(item))) {
      return fail("explicit row");
    }
    const expected = plan.noise.map((spec, column) => base[spec.variableKey] +
      (rowIndex === column * 2 + 1
        ? normalizedRequest.interventionDeltasNm[stableSpecId(spec)] : 0));
    if (row.some((item, column) => item !== expected[column])) return fail("explicit row binding");
    return Object.freeze([...row]) as readonly number[];
  });
  const statuses = new Set(["evaluated_hit", "evaluated_no_impact", "numerical_failure"]);
  const trials = value.trials.map((trial) => {
    if (!isRecord(trial) || Object.keys(trial).sort().join("|") !==
        ["status", "state", "outputs"].sort().join("|") || !statuses.has(String(trial.status)) ||
        !Array.isArray(trial.outputs) || trial.outputs.length !== SWING_VARIATION_OUTPUT_NAMES.length ||
        trial.outputs.some((item) => item !== null &&
          (typeof item !== "number" || !Number.isFinite(item))) ||
        (trial.state !== null && (!Array.isArray(trial.state) || trial.state.length !== 3 ||
          trial.state.some((item) => typeof item !== "number" || !Number.isFinite(item))))) {
      return fail("trial evidence");
    }
    if (trial.status === "numerical_failure" &&
        (trial.state !== null || trial.outputs.some((item) => item !== null))) {
      return fail("failure evidence");
    }
    return Object.freeze({
      status: trial.status as "evaluated_hit" | "evaluated_no_impact" | "numerical_failure",
      state: trial.state === null ? null : Object.freeze([...(trial.state as number[])]),
      outputs: Object.freeze([...(trial.outputs as (number | null)[])]),
    });
  });
  if (authority.sources.length !== plan.noise.length ||
      authority.pairs.length !== plan.noise.length) return fail("source roster");
  plan.noise.forEach((spec, index) => {
    const specId = stableSpecId(spec);
    const source = authority.sources[index];
    const pair = authority.pairs[index];
    if (source.specId !== specId || source.variableKey !== spec.variableKey ||
        source.timeWindowS[0] !== spec.timeWindowS?.[0] ||
        source.timeWindowS[1] !== spec.timeWindowS?.[1] ||
        pair.sourceSpecId !== specId || pair.baselineTrialIndex !== index * 2 ||
        pair.perturbedTrialIndex !== index * 2 + 1 ||
        pair.baselineSourceValue !== base[spec.variableKey] ||
        pair.perturbedSourceValue !== base[spec.variableKey] +
          normalizedRequest.interventionDeltasNm[specId]) {
      fail("authority binding");
    }
  });
  const expectedAuthority = localizedAuthorityFromEvidence(
    normalizedRequest, plan, explicitRows, trials, designIdentity,
  );
  if (JSON.stringify(attributionAuthorityToValue(authority)) !==
      JSON.stringify(attributionAuthorityToValue(expectedAuthority))) {
    return fail("authority evidence binding");
  }
  return Object.freeze({
    schemaId: PAIRED_RESULT_SCHEMA_ID, schemaVersion: 1,
    requestIdentity, designIdentity, authority, explicitRows, trials,
  });
}

class InlineService implements LocalizedPairedExecutionServiceTs {
  execute(request: LocalizedPairedRequestTs, controls: LocalizedPairedControlsTs) {
    if (controls.signal.aborted) return Promise.reject(abortError());
    return executeLocalizedPairedWork(request, controls.onProgress);
  }
}

class WorkerJob {
  private settled = false;
  private processingResult = false;
  private completedRuns = 0;
  private readonly totalRuns: number;
  private resolve!: (result: LocalizedPairedResultTs) => void;
  private reject!: (error: Error | DOMException) => void;

  constructor(
    private readonly worker: Worker,
    private readonly request: LocalizedPairedRequestTs,
    private readonly controls: LocalizedPairedControlsTs,
  ) { this.totalRuns = pairedSourcePlan(request).noise.length * 2; }

  start(): Promise<LocalizedPairedResultTs> {
    const promise = new Promise<LocalizedPairedResultTs>((resolve, reject) => {
      this.resolve = resolve; this.reject = reject;
    });
    this.controls.signal.addEventListener("abort", this.cancel, { once: true });
    this.worker.onerror = (event) => this.rejectOnce(new Error(event.message || "Paired worker failed."));
    this.worker.onmessageerror = () => this.rejectOnce(new Error("Paired worker response could not be decoded."));
    this.worker.onmessage = this.handleMessage;
    if (this.controls.signal.aborted) this.cancel();
    else {
      try { this.worker.postMessage(this.request); }
      catch (error) { this.rejectOnce(error); }
    }
    return promise;
  }

  private readonly cancel = () => this.rejectOnce(abortError());
  private readonly handleMessage = (event: MessageEvent<unknown>) => {
    if (this.settled || this.processingResult) return;
    try {
      const message = event.data;
      if (!isRecord(message) || typeof message.kind !== "string") return fail("message");
      if (message.kind === "progress") return this.acceptProgress(message.progress);
      if (message.kind === "error" && typeof message.message === "string") {
        return this.rejectOnce(new Error(message.message.slice(0, MAX_WORKER_ERROR_LENGTH)));
      }
      if (message.kind !== "result" || this.completedRuns !== this.totalRuns) {
        return fail("result progress");
      }
      this.processingResult = true;
      void validateLocalizedPairedResult(message.result, this.request).then(
        (result) => this.resolveOnce(result), (error) => this.rejectOnce(error),
      );
    } catch (error) { this.rejectOnce(error); }
  };

  private acceptProgress(value: unknown): void {
    if (!isRecord(value) || value.completedRuns !== this.completedRuns + 1 ||
        value.totalRuns !== this.totalRuns) return fail("progress sequence");
    this.completedRuns += 1;
    this.controls.onProgress({ completedRuns: this.completedRuns, totalRuns: this.totalRuns });
  }

  private resolveOnce(result: LocalizedPairedResultTs): void {
    if (this.settled) return;
    this.settled = true; this.cleanup(); this.resolve(result);
  }
  private rejectOnce(error: unknown): void {
    if (this.settled) return;
    this.settled = true; this.cleanup(); this.reject(workerError(error));
  }
  private cleanup(): void {
    this.controls.signal.removeEventListener("abort", this.cancel);
    this.worker.onmessage = null; this.worker.onmessageerror = null;
    this.worker.onerror = null; this.worker.terminate();
  }
}

class WorkerService implements LocalizedPairedExecutionServiceTs {
  constructor(private readonly factory: LocalizedPairedWorkerFactory) {}
  execute(request: LocalizedPairedRequestTs, controls: LocalizedPairedControlsTs) {
    try {
      const normalized = normalizePairedRequest(request);
      if (controls.signal.aborted) return Promise.reject(abortError());
      return new WorkerJob(this.factory(), normalized, controls).start();
    } catch (error) { return Promise.reject(workerError(error)); }
  }
}

export const createLocalizedPairedExecutionService = (
  factory?: LocalizedPairedWorkerFactory,
): LocalizedPairedExecutionServiceTs => {
  if (factory) return new WorkerService(factory);
  if (typeof Worker === "undefined") return new InlineService();
  return new WorkerService(() => new Worker(
    new URL("./localizedAttributionExecution.worker.ts", import.meta.url),
    { type: "module", name: "rate-localized-paired-execution" },
  ));
};
