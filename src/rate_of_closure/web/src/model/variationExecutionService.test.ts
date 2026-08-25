import { describe, expect, it, vi } from "vitest";

import { executeVariationAnalyses } from "./variationAnalysisPolicy";
import {
  executeVariationWork,
  createVariationExecutionService,
  plannedVariationRuns,
  prepareVariationExecutionRequest,
  type VariationExecutionProgress,
  type VariationWorkerFactory,
} from "./variationExecutionService";
import { CATEGORY_LAUNCH, type VariationPlanTs } from "./variation";

const BALL_SPEED = `${CATEGORY_LAUNCH}.ball_speed_mph`;
const LAUNCH_ANGLE = `${CATEGORY_LAUNCH}.launch_angle_deg`;

const plan: VariationPlanTs = {
  mode: "launch",
  baseVariables: {
    [BALL_SPEED]: 154,
    [LAUNCH_ANGLE]: 13,
    [`${CATEGORY_LAUNCH}.launch_azimuth_deg`]: 0,
    [`${CATEGORY_LAUNCH}.spin_rpm`]: 2400,
    [`${CATEGORY_LAUNCH}.spin_axis_deg`]: 0,
  },
  noise: [
    { variableKey: BALL_SPEED, distribution: "normal", scale: 1, lower: null, upper: null },
    { variableKey: LAUNCH_ANGLE, distribution: "normal", scale: 0.5, lower: null, upper: null },
  ],
  nRuns: 4,
  seed: 93,
  flightModel: "waterloo_penner",
};

const jointRequest = () => prepareVariationExecutionRequest(plan, "all_together");

describe("variation execution authority", () => {
  it("preserves the existing deterministic results and reports completed evaluations", () => {
    const progress: VariationExecutionProgress[] = [];
    const request = prepareVariationExecutionRequest(plan, "both");
    const result = executeVariationWork(request, (value) => progress.push(value));
    const replay = executeVariationWork(request, () => undefined);
    const expected = executeVariationAnalyses(plan, "both");

    expect(result.dataset).toEqual(expected.dataset);
    expect(result.sensitivity).toEqual(expected.sensitivity);
    expect(result.ensemble).toBeNull();
    expect(replay).toEqual(result);
    expect(result.executionMetadata).toEqual(request.executionMetadata);
    expect(plannedVariationRuns(plan, "both")).toBe(12);
    expect(progress[progress.length - 1]).toEqual({
      completedRuns: 12,
      totalRuns: 12,
      phase: "individual",
    });
    expect(progress.map(({ completedRuns }) => completedRuns)).toEqual(
      Array.from({ length: 12 }, (_unused, index) => index + 1),
    );
  });

  it("fails before execution when a plan names an unsupported flight model", () => {
    expect(() => executeVariationWork(
      {
        ...jointRequest(),
        plan: { ...plan, flightModel: "custom-flight-model" },
      },
      vi.fn(),
    )).toThrow(/flight model.*custom-flight-model.*waterloo_penner/i);
  });
});

class FakeWorker {
  onerror: ((event: ErrorEvent) => unknown) | null = null;
  onmessage: ((event: MessageEvent) => unknown) | null = null;
  onmessageerror: ((event: MessageEvent) => unknown) | null = null;
  readonly posted: unknown[] = [];
  postError: unknown = null;
  terminateCount = 0;

  postMessage(message: unknown): void {
    if (this.postError !== null) throw this.postError;
    this.posted.push(message);
  }

  terminate(): void {
    this.terminateCount += 1;
  }

  emitMessage(data: unknown): void {
    this.onmessage?.({ data } as MessageEvent);
  }

  emitError(message = "worker exploded"): void {
    this.onerror?.({ message } as ErrorEvent);
  }

  emitMessageError(): void {
    this.onmessageerror?.({ data: null } as MessageEvent);
  }
}

const workerService = (worker: FakeWorker) => {
  const factory: VariationWorkerFactory = () => worker as unknown as Worker;
  return createVariationExecutionService(factory);
};

const validResult = (): ReturnType<typeof executeVariationWork> =>
  executeVariationWork(jointRequest(), () => undefined);

describe("production worker transport", () => {
  it("rejects an unsupported flight model before constructing or posting to a worker", () => {
    const worker = new FakeWorker();

    expect(() => workerService(worker).execute(
      {
        ...jointRequest(),
        plan: { ...plan, flightModel: "custom-flight-model" },
      },
      { signal: new AbortController().signal, onProgress: vi.fn() },
    )).toThrow(/flight model.*custom-flight-model.*waterloo_penner/i);
    expect(worker.posted).toEqual([]);
  });

  it("accepts validated progress and result, then cleans up exactly once", async () => {
    const worker = new FakeWorker();
    const onProgress = vi.fn();
    const controller = new AbortController();
    const pending = workerService(worker).execute(
      jointRequest(),
      { signal: controller.signal, onProgress },
    );

    expect(worker.posted).toEqual([jointRequest()]);
    for (let completedRuns = 1; completedRuns <= plan.nRuns; completedRuns += 1) {
      worker.emitMessage({
        kind: "progress",
        progress: { completedRuns, totalRuns: plan.nRuns, phase: "joint" },
      });
    }
    const result = validResult();
    worker.emitMessage({ kind: "result", result });

    await expect(pending).resolves.toEqual(result);
    expect(onProgress).toHaveBeenCalledTimes(plan.nRuns);
    expect(worker.terminateCount).toBe(1);
    expect(worker.onmessage).toBeNull();
    expect(worker.onerror).toBeNull();
    expect(worker.onmessageerror).toBeNull();
  });

  it("rejects cross-plan metadata before constructing or posting to a worker", () => {
    const worker = new FakeWorker();
    const request = jointRequest();
    request.plan = { ...plan, seed: plan.seed + 1 };

    expect(() => workerService(worker).execute(
      request,
      { signal: new AbortController().signal, onProgress: vi.fn() },
    )).toThrow(/plan digest/i);
    expect(worker.posted).toEqual([]);
  });

  it("rejects unknown inline metadata fields before performing work", () => {
    const request = structuredClone(jointRequest());
    Object.assign(request.executionMetadata, { unexpected: true });

    expect(() => executeVariationWork(request, vi.fn())).toThrow(/metadata fields/i);
  });

  it("rejects result metadata drift after otherwise valid worker progress", async () => {
    const worker = new FakeWorker();
    const pending = workerService(worker).execute(
      jointRequest(),
      { signal: new AbortController().signal, onProgress: vi.fn() },
    );
    for (let completedRuns = 1; completedRuns <= plan.nRuns; completedRuns += 1) {
      worker.emitMessage({
        kind: "progress",
        progress: { completedRuns, totalRuns: plan.nRuns, phase: "joint" },
      });
    }
    const result = structuredClone(validResult());
    (result.executionMetadata.resolvedVariables[0] as { unit: string }).unit = "m/s";
    worker.emitMessage({ kind: "result", result });

    await expect(pending).rejects.toThrow(/execution metadata/i);
  });

  it("rejects unknown Worker result metadata fields", async () => {
    const worker = new FakeWorker();
    const pending = workerService(worker).execute(
      jointRequest(),
      { signal: new AbortController().signal, onProgress: vi.fn() },
    );
    for (let completedRuns = 1; completedRuns <= plan.nRuns; completedRuns += 1) {
      worker.emitMessage({
        kind: "progress",
        progress: { completedRuns, totalRuns: plan.nRuns, phase: "joint" },
      });
    }
    const result = structuredClone(validResult());
    Object.assign(result.executionMetadata, { unexpected: true });
    worker.emitMessage({ kind: "result", result });

    await expect(pending).rejects.toThrow(/execution metadata/i);
  });

  it("aborts, terminates, and ignores late events", async () => {
    const worker = new FakeWorker();
    const onProgress = vi.fn();
    const controller = new AbortController();
    const pending = workerService(worker).execute(
      jointRequest(),
      { signal: controller.signal, onProgress },
    );
    const lateMessage = worker.onmessage;

    controller.abort();
    lateMessage?.({
      data: {
        kind: "progress",
        progress: { completedRuns: 1, totalRuns: plan.nRuns, phase: "joint" },
      },
    } as MessageEvent);

    await expect(pending).rejects.toMatchObject({ name: "AbortError" });
    expect(onProgress).not.toHaveBeenCalled();
    expect(worker.terminateCount).toBe(1);
  });

  it.each([
    ["worker error", (worker: FakeWorker) => worker.emitError(), /worker exploded/i],
    ["message decode error", (worker: FakeWorker) => worker.emitMessageError(), /decode/i],
    ["malformed message", (worker: FakeWorker) => worker.emitMessage(null), /message/i],
    [
      "invalid progress",
      (worker: FakeWorker) => worker.emitMessage({
        kind: "progress",
        progress: { completedRuns: 2, totalRuns: plan.nRuns, phase: "joint" },
      }),
      /progress/i,
    ],
    [
      "invalid result",
      (worker: FakeWorker) => worker.emitMessage({
        kind: "result",
        result: { dataset: null, sensitivity: null, ensemble: null },
      }),
      /result/i,
    ],
  ])("fails closed for %s", async (_label, emit, expectedMessage) => {
    const worker = new FakeWorker();
    const pending = workerService(worker).execute(
      jointRequest(),
      { signal: new AbortController().signal, onProgress: vi.fn() },
    );

    emit(worker);

    await expect(pending).rejects.toThrow(expectedMessage);
    expect(worker.terminateCount).toBe(1);
  });

  it("cleans up when the initial postMessage cannot clone the request", async () => {
    const worker = new FakeWorker();
    worker.postError = new DOMException("cannot clone", "DataCloneError");

    const pending = workerService(worker).execute(
      jointRequest(),
      { signal: new AbortController().signal, onProgress: vi.fn() },
    );

    await expect(pending).rejects.toMatchObject({ name: "DataCloneError" });
    expect(worker.terminateCount).toBe(1);
  });

  it("turns a worker-construction failure into a rejected execution", async () => {
    const constructionError = new DOMException("workers blocked", "SecurityError");
    const factory: VariationWorkerFactory = () => { throw constructionError; };

    const pending = createVariationExecutionService(factory).execute(
      jointRequest(),
      { signal: new AbortController().signal, onProgress: vi.fn() },
    );

    await expect(pending).rejects.toBe(constructionError);
  });

  it("rejects validly shaped results whose field identities cross the request", async () => {
    const worker = new FakeWorker();
    const pending = workerService(worker).execute(
      jointRequest(),
      { signal: new AbortController().signal, onProgress: vi.fn() },
    );
    for (let completedRuns = 1; completedRuns <= plan.nRuns; completedRuns += 1) {
      worker.emitMessage({
        kind: "progress",
        progress: { completedRuns, totalRuns: plan.nRuns, phase: "joint" },
      });
    }
    const crossed = validResult();
    crossed.dataset!.outputNames = crossed.dataset!.outputNames.slice().reverse();

    worker.emitMessage({ kind: "result", result: crossed });

    await expect(pending).rejects.toThrow(/output identity/i);
    expect(worker.terminateCount).toBe(1);
  });
});
