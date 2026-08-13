import { describe, expect, it, vi } from "vitest";

import type { CapabilityWorkflowDocument } from "./capabilityWorkflow";
import { createCapabilityWorkerRunner } from "./capabilityWorkerClient";

const document = {} as CapabilityWorkflowDocument;

class WorkerDouble {
  static instance: WorkerDouble | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: ErrorEvent) => void) | null = null;
  postMessage = vi.fn();
  terminate = vi.fn();

  constructor() { WorkerDouble.instance = this; }
}

describe("capability worker client", () => {
  it("forwards progress and terminates on cancellation", async () => {
    const progress = vi.fn();
    const runner = createCapabilityWorkerRunner(
      () => new WorkerDouble() as unknown as Worker,
    );
    const controller = runner(document, progress);
    const worker = WorkerDouble.instance!;

    worker.onmessage?.({ data: { type: "progress", completed: 2, total: 5 } } as MessageEvent);
    expect(progress).toHaveBeenCalledWith({ completed: 2, total: 5 });

    controller.cancel();
    await expect(controller.promise).rejects.toMatchObject({ name: "AbortError" });
    worker.onmessage?.({ data: { type: "progress", completed: 5, total: 5 } } as MessageEvent);
    expect(progress).toHaveBeenCalledTimes(1);
    expect(worker.postMessage).toHaveBeenCalledWith({ type: "run", document });
    expect(worker.terminate).toHaveBeenCalledOnce();
  });

  it("rejects and terminates when completed worker output is malformed", async () => {
    const runner = createCapabilityWorkerRunner(
      () => new WorkerDouble() as unknown as Worker,
    );
    const controller = runner(document, vi.fn());
    const worker = WorkerDouble.instance!;

    worker.onmessage?.({
      data: { type: "complete", output: { result: {}, ensemble: {} } },
    } as MessageEvent);

    await expect(controller.promise).rejects.toThrow();
    expect(worker.terminate).toHaveBeenCalledOnce();
  });
});
