import { describe, expect, it, vi } from "vitest";

import { createDurableEnsembleRunner, type DurableWorkerResponse } from "./durableEnsembleWorkerClient";

class WorkerDouble {
  onmessage: ((event: MessageEvent<DurableWorkerResponse>) => void) | null = null;
  onerror: ((event: ErrorEvent) => void) | null = null;
  postMessage = vi.fn();
  terminate = vi.fn();
}

describe("durable ensemble Worker client", () => {
  it("transports capability and cancellation without doing analysis", async () => {
    const worker = new WorkerDouble();
    const runner = createDurableEnsembleRunner(() => worker as unknown as Worker);
    const capability = runner.capability();
    worker.onmessage?.({ data: { type: "capability", capability: {
      available: true, apiPrefix: "/api/rate-of-closure/v1",
    } } } as MessageEvent<DurableWorkerResponse>);
    await expect(capability).resolves.toMatchObject({ available: true });
    const control = runner.run({} as never, vi.fn());
    control.cancel();
    expect(worker.postMessage).toHaveBeenLastCalledWith({ type: "cancel" });
    runner.close();
    expect(worker.terminate).toHaveBeenCalledOnce();
  });
});

