/** Dedicated worker entry point for capability optimization. */

import { runCapabilityOptimization } from "../model/capabilityRun";
import type {
  CapabilityWorkerRequest,
  CapabilityWorkerResponse,
} from "../model/capabilityWorkerClient";

interface WorkerScope {
  addEventListener(
    type: "message",
    listener: (event: MessageEvent<CapabilityWorkerRequest>) => void,
  ): void;
  postMessage(message: CapabilityWorkerResponse): void;
}

const scope = globalThis as unknown as WorkerScope;

scope.addEventListener("message", (event) => {
  if (event.data.type !== "run") return;
  try {
    const output = runCapabilityOptimization(
      event.data.document,
      (progress) => scope.postMessage({ type: "progress", ...progress }),
    );
    scope.postMessage({ type: "complete", output });
  } catch (error: unknown) {
    scope.postMessage({ type: "error",
      message: error instanceof Error ? error.message : String(error) });
  }
});
