/** Browser-worker boundary for cancellable capability optimization. */

import { parseOptimizationResult } from "./capabilityContract";
import type { CapabilityCohort } from "./capabilityObservationEnsemble";
import type { CapabilityProgress, CapabilityRunOutput } from "./capabilityRun";
import type { CapabilityWorkflowDocument } from "./capabilityWorkflow";
import {
  createScalarEnsemble,
  type ScalarEnsembleInput,
} from "./scalarEnsembleContract";

export interface CapabilityRunController {
  readonly promise: Promise<CapabilityRunOutput>;
  cancel: () => void;
}

export type CapabilityRunner = (
  document: CapabilityWorkflowDocument,
  onProgress: (progress: CapabilityProgress) => void,
) => CapabilityRunController;

export type CapabilityWorkerRequest = {
  readonly type: "run"; readonly document: CapabilityWorkflowDocument;
};

export type CapabilityWorkerResponse =
  | ({ readonly type: "progress" } & CapabilityProgress)
  | { readonly type: "complete"; readonly output: CapabilityRunOutput }
  | { readonly type: "error"; readonly message: string };

type WorkerFactory = () => Worker;

const defaultWorkerFactory: WorkerFactory = () => new Worker(
  new URL("../workers/capabilityOptimization.worker.ts", import.meta.url),
  { type: "module", name: "capability-optimization" },
);

const validatedOutput = (output: CapabilityRunOutput): CapabilityRunOutput =>
  Object.freeze({
    result: parseOptimizationResult(output.result),
    ensemble: createScalarEnsemble(
      output.ensemble as unknown as ScalarEnsembleInput<CapabilityCohort>,
    ),
  });

export function createCapabilityWorkerRunner(
  workerFactory: WorkerFactory = defaultWorkerFactory,
): CapabilityRunner {
  return (document, onProgress) => {
    const worker = workerFactory();
    let rejectRun: (reason: unknown) => void = () => undefined;
    let settled = false;
    const finish = (action: () => void): void => {
      if (settled) return;
      settled = true;
      worker.terminate();
      action();
    };
    const promise = new Promise<CapabilityRunOutput>((resolve, reject) => {
      rejectRun = reject;
      worker.onmessage = (event: MessageEvent<CapabilityWorkerResponse>) => {
        if (settled) return;
        const message = event.data;
        if (message.type === "progress") {
          onProgress({ completed: message.completed, total: message.total });
        }
        else if (message.type === "complete") {
          try {
            const output = validatedOutput(message.output);
            finish(() => resolve(output));
          } catch (reason: unknown) {
            finish(() => reject(reason));
          }
        } else finish(() => reject(new Error(message.message)));
      };
      worker.onerror = (event) => finish(() => reject(
        new Error(event.message || "Capability optimization worker failed"),
      ));
      worker.postMessage({ type: "run", document } satisfies CapabilityWorkerRequest);
    });
    return { promise, cancel: () => finish(() => rejectRun(
      new DOMException("Capability optimization cancelled", "AbortError"),
    )) };
  };
}

export const runCapabilityInWorker = createCapabilityWorkerRunner();
