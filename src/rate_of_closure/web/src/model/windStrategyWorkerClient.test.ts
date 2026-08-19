import { describe, expect, it, vi } from "vitest";

import type { WindStrategyRequest } from "./windUncertainty";
import { createWindStrategyWorkerRunner } from "./windStrategyWorkerClient";

const request = {} as WindStrategyRequest;

class WorkerDouble {
  static instance: WorkerDouble | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: ErrorEvent) => void) | null = null;
  postMessage = vi.fn();
  terminate = vi.fn();

  constructor() {
    WorkerDouble.instance = this;
  }
}

describe("wind strategy worker client", () => {
  it("forwards progress, resolves once, and terminates the worker", async () => {
    const progress = vi.fn();
    const runner = createWindStrategyWorkerRunner(
      () => new WorkerDouble() as unknown as Worker,
    );
    const controller = runner(request, progress);
    const worker = WorkerDouble.instance!;
    const analysis = { schema_version: "wind-strategy-analysis/v2" };

    worker.onmessage?.({ data: { type: "progress", completed: 1, total: 2 } } as MessageEvent);
    worker.onmessage?.({ data: { type: "complete", analysis } } as MessageEvent);

    await expect(controller.promise).resolves.toBe(analysis);
    expect(progress).toHaveBeenCalledWith({ completed: 1, total: 2 });
    expect(worker.postMessage).toHaveBeenCalledWith({ type: "run", request });
    expect(worker.terminate).toHaveBeenCalledOnce();
  });

  it("rejects and terminates when cancelled", async () => {
    const runner = createWindStrategyWorkerRunner(
      () => new WorkerDouble() as unknown as Worker,
    );
    const controller = runner(request, vi.fn());

    controller.cancel();

    await expect(controller.promise).rejects.toMatchObject({ name: "AbortError" });
    expect(WorkerDouble.instance!.terminate).toHaveBeenCalledOnce();
  });
});
