/** Typed browser boundary for the transport-only durable-ensemble Worker. */

import type {
  DurableEnsembleCapability,
  DurableEnsembleJob,
  DurableEnsembleRequestDocument,
} from "./durableEnsembleAuthorityContract";

export type DurableWorkerRequest =
  | { readonly type: "capability" }
  | { readonly type: "run"; readonly request: DurableEnsembleRequestDocument }
  | { readonly type: "cancel" };

export type DurableWorkerResponse =
  | { readonly type: "capability"; readonly capability: DurableEnsembleCapability }
  | { readonly type: "job"; readonly job: DurableEnsembleJob }
  | { readonly type: "error"; readonly message: string };

export interface DurableEnsembleRunController {
  readonly promise: Promise<DurableEnsembleJob>;
  cancel(): void;
}

export interface DurableEnsembleRunner {
  capability(): Promise<DurableEnsembleCapability>;
  run(
    request: DurableEnsembleRequestDocument,
    onJob: (job: DurableEnsembleJob) => void,
  ): DurableEnsembleRunController;
  close(): void;
}

type WorkerFactory = () => Worker;
const defaultWorkerFactory: WorkerFactory = () => new Worker(
  new URL("../workers/durableEnsembleAuthority.worker.ts", import.meta.url),
  { type: "module", name: "durable-ensemble-authority" },
);

export function createDurableEnsembleRunner(
  factory: WorkerFactory = defaultWorkerFactory,
): DurableEnsembleRunner {
  let worker: Worker | null = null;
  let capabilityResolve: ((value: DurableEnsembleCapability) => void) | null = null;
  let capabilityReject: ((reason: unknown) => void) | null = null;
  let runResolve: ((value: DurableEnsembleJob) => void) | null = null;
  let runReject: ((reason: unknown) => void) | null = null;
  let onJob: ((job: DurableEnsembleJob) => void) | null = null;
  const ensure = (): Worker => {
    if (worker !== null) return worker;
    worker = factory();
    worker.onmessage = (event: MessageEvent<DurableWorkerResponse>) => {
      const message = event.data;
      if (message.type === "capability") {
        capabilityResolve?.(message.capability); capabilityResolve = null; capabilityReject = null;
      } else if (message.type === "job") {
        onJob?.(message.job);
        if (["completed", "cancelled", "failed"].includes(message.job.status)) {
          runResolve?.(message.job); runResolve = null; runReject = null; onJob = null;
        }
      } else {
        const error = new Error(message.message);
        capabilityReject?.(error); runReject?.(error);
        capabilityResolve = null; capabilityReject = null; runResolve = null; runReject = null;
      }
    };
    worker.onerror = (event) => {
      const error = new Error(event.message || "Durable ensemble Worker failed.");
      capabilityReject?.(error); runReject?.(error);
    };
    return worker;
  };
  return Object.freeze({
    capability: () => new Promise<DurableEnsembleCapability>((resolve, reject) => {
      capabilityResolve = resolve; capabilityReject = reject;
      ensure().postMessage({ type: "capability" } satisfies DurableWorkerRequest);
    }),
    run: (
      request: DurableEnsembleRequestDocument,
      report: (job: DurableEnsembleJob) => void,
    ) => {
      if (runResolve !== null) throw new Error("A durable ensemble run is already active.");
      onJob = report;
      const promise = new Promise<DurableEnsembleJob>((resolve, reject) => {
        runResolve = resolve; runReject = reject;
        ensure().postMessage({ type: "run", request } satisfies DurableWorkerRequest);
      });
      return { promise, cancel: () => worker?.postMessage({ type: "cancel" } satisfies DurableWorkerRequest) };
    },
    close: () => { worker?.terminate(); worker = null; },
  });
}
