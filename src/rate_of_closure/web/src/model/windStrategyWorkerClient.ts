/** Browser-worker boundary for cancellable wind-strategy analysis. */

import type { WindStrategyAnalysis, WindStrategyRequest } from "./windUncertainty";

export interface WindStrategyProgress {
  readonly completed: number;
  readonly total: number;
}

export interface WindStrategyRunController {
  readonly promise: Promise<WindStrategyAnalysis>;
  cancel: () => void;
}

export type WindStrategyRunner = (
  request: WindStrategyRequest,
  onProgress: (progress: WindStrategyProgress) => void,
) => WindStrategyRunController;

export type WindStrategyWorkerRequest = {
  readonly type: "run";
  readonly request: WindStrategyRequest;
};

export type WindStrategyWorkerResponse =
  | ({ readonly type: "progress" } & WindStrategyProgress)
  | { readonly type: "complete"; readonly analysis: WindStrategyAnalysis }
  | { readonly type: "error"; readonly message: string };

type WorkerFactory = () => Worker;

const defaultWorkerFactory: WorkerFactory = () => new Worker(
  new URL("../workers/windStrategy.worker.ts", import.meta.url),
  { type: "module", name: "wind-strategy-analysis" },
);

/** Build a runner with an injectable worker constructor for deterministic tests. */
export function createWindStrategyWorkerRunner(
  workerFactory: WorkerFactory = defaultWorkerFactory,
): WindStrategyRunner {
  return (request, onProgress) => {
    const worker = workerFactory();
    let rejectRun: (reason: unknown) => void = () => undefined;
    let settled = false;
    const finish = (action: () => void): void => {
      if (settled) return;
      settled = true;
      worker.terminate();
      action();
    };
    const promise = new Promise<WindStrategyAnalysis>((resolve, reject) => {
      rejectRun = reject;
      worker.onmessage = (event: MessageEvent<WindStrategyWorkerResponse>) => {
        const message = event.data;
        if (message.type === "progress") {
          onProgress({ completed: message.completed, total: message.total });
        } else if (message.type === "complete") {
          finish(() => resolve(message.analysis));
        } else {
          finish(() => reject(new Error(message.message)));
        }
      };
      worker.onerror = (event) => {
        finish(() => reject(new Error(event.message || "Wind strategy worker failed")));
      };
      worker.postMessage({ type: "run", request } satisfies WindStrategyWorkerRequest);
    });
    return {
      promise,
      cancel: () => finish(() => rejectRun(new DOMException(
        "Wind strategy analysis cancelled", "AbortError",
      ))),
    };
  };
}

export const runWindStrategyInWorker = createWindStrategyWorkerRunner();
