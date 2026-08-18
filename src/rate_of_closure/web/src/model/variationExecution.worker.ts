/// <reference lib="webworker" />

import {
  executeVariationWork,
  type VariationExecutionRequest,
  type VariationWorkerResponse,
} from "./variationExecutionService";

const context: DedicatedWorkerGlobalScope = self as DedicatedWorkerGlobalScope;

context.onmessage = (event: MessageEvent<VariationExecutionRequest>) => {
  try {
    const result = executeVariationWork(event.data, (progress) => {
      const message: VariationWorkerResponse = { kind: "progress", progress };
      context.postMessage(message);
    });
    const message: VariationWorkerResponse = { kind: "result", result };
    context.postMessage(message);
  } catch (error) {
    const message: VariationWorkerResponse = {
      kind: "error",
      message: error instanceof Error ? error.message : String(error),
    };
    context.postMessage(message);
  }
};

export {};
