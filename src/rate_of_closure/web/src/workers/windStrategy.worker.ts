/** Dedicated worker entry point for wind-strategy analysis. */

import { analyzeWindStrategies } from "../model/windUncertainty";
import type {
  WindStrategyWorkerRequest,
  WindStrategyWorkerResponse,
} from "../model/windStrategyWorkerClient";

interface WorkerScope {
  addEventListener(type: "message", listener: (event: MessageEvent<WindStrategyWorkerRequest>) => void): void;
  postMessage(message: WindStrategyWorkerResponse): void;
}

const scope = globalThis as unknown as WorkerScope;

scope.addEventListener("message", (event) => {
  if (event.data.type !== "run") return;
  try {
    let lastReported = 0;
    const analysis = analyzeWindStrategies(
      event.data.request,
      (completed: number, total: number) => {
        const reportingInterval = Math.max(1, Math.floor(total / 100));
        if (completed === total || completed - lastReported >= reportingInterval) {
          lastReported = completed;
          scope.postMessage({ type: "progress", completed, total });
        }
      },
    );
    scope.postMessage({ type: "complete", analysis });
  } catch (error: unknown) {
    scope.postMessage({
      type: "error",
      message: error instanceof Error ? error.message : String(error),
    });
  }
});
