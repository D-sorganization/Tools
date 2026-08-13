import { describe, expect, it, vi } from "vitest";

import localizedPlan from "./__fixtures__/localized_torque_authoring_v1.json";
import pythonGolden from "./__fixtures__/localized_attribution_python_golden_v1.json";
import { planFromJson, planToJson } from "./variation";
import {
  executeLocalizedPairedWork, PAIRED_REQUEST_SCHEMA_ID,
  type LocalizedPairedRequestTs,
} from "./localizedAttributionExecution";
import {
  createLocalizedPairedExecutionService,
  type LocalizedPairedWorkerFactory,
} from "./localizedAttributionExecutionService";

const request = (): LocalizedPairedRequestTs => {
  const plan = planFromJson(JSON.stringify(localizedPlan));
  return {
    schemaId: PAIRED_REQUEST_SCHEMA_ID, schemaVersion: 1,
    designId: "react.paired.fixture", sourcePlanJson: planToJson({ ...plan, groups: [] }),
    interventionDeltasNm: { "shoulder-window": 2, "wrist-window": -1.5 },
    statePointId: "swing.clubhead.reference", stateTimeS: 0.02,
  };
};

class FakeWorker {
  onerror: ((event: ErrorEvent) => unknown) | null = null;
  onmessage: ((event: MessageEvent) => unknown) | null = null;
  onmessageerror: ((event: MessageEvent) => unknown) | null = null;
  posted: unknown[] = [];
  terminateCount = 0;
  postMessage(value: unknown): void { this.posted.push(value); }
  terminate(): void { this.terminateCount += 1; }
  emit(value: unknown): void { this.onmessage?.({ data: value } as MessageEvent); }
}

const service = (worker: FakeWorker) => createLocalizedPairedExecutionService(
  (() => worker as unknown as Worker) as LocalizedPairedWorkerFactory,
);

describe("localized paired Worker protocol", () => {
  it("matches the Python-owned shared semantic golden without claiming solver identity", () => {
    const authored = planFromJson(JSON.stringify(localizedPlan));
    const shoulder = {
      ...authored.noise[0], specId: pythonGolden.source_spec_id,
      timeWindowS: pythonGolden.time_window_s as [number, number],
    };
    const plan = { ...authored, baseVariables: {
      [shoulder.variableKey]: pythonGolden.baseline_source_value,
    }, noise: [shoulder], nRuns: 2, groups: [] };
    const delta = pythonGolden.perturbed_source_value - pythonGolden.baseline_source_value;
    expect({
      source_spec_id: shoulder.specId,
      variable_key: shoulder.variableKey,
      joint_id: shoulder.pointIds?.[0],
      time_window_s: shoulder.timeWindowS,
      baseline_source_value: plan.baseVariables[shoulder.variableKey],
      perturbed_source_value: plan.baseVariables[shoulder.variableKey] + delta,
      pair_trials: [0, 1],
      interpretation: "paired-planted-intervention-noncausal",
      state_target: pythonGolden.state_target,
    }).toEqual({
      source_spec_id: pythonGolden.source_spec_id,
      variable_key: pythonGolden.variable_key,
      joint_id: pythonGolden.joint_id,
      time_window_s: pythonGolden.time_window_s,
      baseline_source_value: pythonGolden.baseline_source_value,
      perturbed_source_value: pythonGolden.perturbed_source_value,
      pair_trials: pythonGolden.pair_trials,
      interpretation: pythonGolden.interpretation,
      state_target: pythonGolden.state_target,
    });
  });
  it("executes deterministic explicit rows with complete typed authority", async () => {
    const progress: number[] = [];
    const first = await executeLocalizedPairedWork(request(),
      (value) => progress.push(value.completedRuns));
    const second = await executeLocalizedPairedWork(request(), () => undefined);

    expect(progress).toEqual([1, 2, 3, 4]);
    expect(first).toEqual(second);
    expect(first.authority.pairs.map((pair) => [
      pair.baselineTrialIndex, pair.perturbedTrialIndex,
    ])).toEqual([[0, 1], [2, 3]]);
    expect(first.authority.targets).toHaveLength(17);
    expect(first.authority.observations).toHaveLength(34);
    expect(first.authority.authorityId).toBe(`paired-attribution.${first.designIdentity}`);
  });

  it("accepts exact progress/result and terminates once", async () => {
    const worker = new FakeWorker();
    const progress = vi.fn();
    const pending = service(worker).execute(request(), {
      signal: new AbortController().signal, onProgress: progress,
    });
    const result = await executeLocalizedPairedWork(request(), () => undefined);
    [1, 2, 3, 4].forEach((completedRuns) => worker.emit({
      kind: "progress", progress: { completedRuns, totalRuns: 4 },
    }));
    worker.emit({ kind: "result", result });

    await expect(pending).resolves.toEqual(result);
    expect(progress).toHaveBeenCalledTimes(4);
    expect(worker.posted).toEqual([request()]);
    expect(worker.terminateCount).toBe(1);
  });

  it("cancels, terminates, and ignores late result", async () => {
    const worker = new FakeWorker();
    const controller = new AbortController();
    const progress = vi.fn();
    const pending = service(worker).execute(request(), {
      signal: controller.signal, onProgress: progress,
    });
    const late = worker.onmessage;
    controller.abort();
    late?.({ data: { kind: "progress", progress: {
      completedRuns: 1, totalRuns: 4,
    } } } as MessageEvent);

    await expect(pending).rejects.toMatchObject({ name: "AbortError" });
    expect(progress).not.toHaveBeenCalled();
    expect(worker.terminateCount).toBe(1);
  });

  it("rejects forged progress and authority binding", async () => {
    const progressWorker = new FakeWorker();
    const progressPending = service(progressWorker).execute(request(), {
      signal: new AbortController().signal, onProgress: vi.fn(),
    });
    progressWorker.emit({ kind: "progress", progress: { completedRuns: 2, totalRuns: 4 } });
    await expect(progressPending).rejects.toThrow(/progress/i);

    const resultWorker = new FakeWorker();
    const resultPending = service(resultWorker).execute(request(), {
      signal: new AbortController().signal, onProgress: vi.fn(),
    });
    const valid = await executeLocalizedPairedWork(request(), () => undefined);
    [1, 2, 3, 4].forEach((completedRuns) => resultWorker.emit({
      kind: "progress", progress: { completedRuns, totalRuns: 4 },
    }));
    resultWorker.emit({ kind: "result", result: {
      ...valid, authority: { ...valid.authority, authorityId: "forged" },
    } });
    await expect(resultPending).rejects.toThrow(/authority identity/i);
  });

  it("rejects forged explicit rows and result message failures", async () => {
    const valid = await executeLocalizedPairedWork(request(), () => undefined);
    const worker = new FakeWorker();
    const pending = service(worker).execute(request(), {
      signal: new AbortController().signal, onProgress: vi.fn(),
    });
    [1, 2, 3, 4].forEach((completedRuns) => worker.emit({
      kind: "progress", progress: { completedRuns, totalRuns: 4 },
    }));
    worker.emit({ kind: "result", result: {
      ...valid, explicitRows: valid.explicitRows.map((row, index) =>
        index === 1 ? [row[0] + 1, row[1]] : row),
    } });
    await expect(pending).rejects.toThrow(/explicit row binding/i);

    const errorWorker = new FakeWorker();
    const errorPending = service(errorWorker).execute(request(), {
      signal: new AbortController().signal, onProgress: vi.fn(),
    });
    errorWorker.emit({ kind: "error", message: "bounded worker failure" });
    await expect(errorPending).rejects.toThrow(/bounded worker failure/i);
    expect(errorWorker.terminateCount).toBe(1);
  });

  it("rejects malformed requests before constructing a Worker", async () => {
    const factory = vi.fn(() => new FakeWorker() as unknown as Worker);
    const invalid = { ...request(), stateTimeS: 1.501 };
    await expect(createLocalizedPairedExecutionService(factory).execute(invalid, {
      signal: new AbortController().signal, onProgress: vi.fn(),
    })).rejects.toThrow(/0\.\.1\.5 s/i);
    expect(factory).not.toHaveBeenCalled();
  });
});
