/// <reference lib="webworker" />

import { executeLocalizedPairedWork, type LocalizedPairedRequestTs } from "./localizedAttributionExecution";

const context: DedicatedWorkerGlobalScope = self as DedicatedWorkerGlobalScope;
context.onmessage = (event: MessageEvent<LocalizedPairedRequestTs>) => {
  void executeLocalizedPairedWork(event.data, (progress) => {
    context.postMessage({ kind: "progress", progress });
  }).then(
    (result) => context.postMessage({ kind: "result", result }),
    (error) => context.postMessage({
      kind: "error", message: error instanceof Error ? error.message : String(error),
    }),
  );
};

export {};
