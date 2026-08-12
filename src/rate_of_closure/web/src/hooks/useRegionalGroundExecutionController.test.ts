import { act, renderHook } from "@testing-library/react";
import { createElement, StrictMode, type ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import jobFixture from "../model/__fixtures__/regional_ground_execution_job_golden_v1.json";
import resultFixture from "../model/__fixtures__/regional_ground_execution_result_golden_v1.json";
import type { RegionalGroundAuthorityCapability } from "../model/regionalGroundAuthority";
import type {
  RegionalGroundAuthorityClient,
  RegionalGroundAuthorityJobStatus,
} from "../model/regionalGroundAuthorityClient";
import { regionalGroundExecutionResultFromJson } from "../model/regionalGroundExecutionResult";
import { parseRegionalGroundExecutionJob } from "../model/regionalGroundExecutionJob";
import { useRegionalGroundExecutionController } from "./useRegionalGroundExecutionController";

const job = parseRegionalGroundExecutionJob(jobFixture.job);
const resultEnvelope = regionalGroundExecutionResultFromJson(
  JSON.stringify(resultFixture.result),
);
const unavailable: RegionalGroundAuthorityCapability = {
  schema_version: "rate-of-closure/regional-ground-authority-capability/v1",
  authority_id: "rate-of-closure-python-authority",
  authority_version: "1",
  available: false,
  regional_ground_execution: false,
  reason_code: "execution_profile_unqualified",
  detail: "Exact execution profile is not qualified.",
};
const status = (
  state: RegionalGroundAuthorityJobStatus["status"],
  completed: number,
): RegionalGroundAuthorityJobStatus => ({
  schema_version: "rate-of-closure/regional-ground-authority-job-status/v1",
  job_id: job.job_id,
  job_sha256: job.job_sha256,
  status: state,
  completed,
  total: job.execution_options.max_trials,
  result_available: state === "succeeded",
  failure: state === "failed" ? { code: "execution_failed", stage: "executor" } : null,
});
const admittedForLifecycleTests = (): boolean => true;
const deferred = <T,>() => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((accept) => { resolve = accept; });
  return { promise, resolve };
};
const client = (overrides: Partial<RegionalGroundAuthorityClient> = {}) => ({
  capability: vi.fn().mockResolvedValue(unavailable),
  submit: vi.fn().mockResolvedValue(status("queued", 0)),
  status: vi.fn().mockResolvedValue(status("running", 1)),
  cancel: vi.fn().mockResolvedValue(status("cancel_requested", 1)),
  result: vi.fn().mockResolvedValue(resultEnvelope),
  ...overrides,
}) satisfies RegionalGroundAuthorityClient;

afterEach(() => vi.useRealTimers());

describe("regional-ground React execution controller", () => {
  it("denies production submission under the exact false-only capability", async () => {
    const authority = client();
    const { result } = renderHook(() => useRegionalGroundExecutionController({
      client: authority,
      capability: unavailable,
    }));

    await expect(result.current.submit(job)).rejects.toThrow(/capability/i);

    expect(authority.submit).not.toHaveBeenCalled();
    expect(result.current.phase).toBe("idle");
    expect(result.current.controls.submitEnabled).toBe(false);
  });

  it("remains live after the development StrictMode effect probe", async () => {
    const authority = client();
    const wrapper = ({ children }: { readonly children: ReactNode }) =>
      createElement(StrictMode, null, children);
    const { result } = renderHook(() => useRegionalGroundExecutionController({
      client: authority,
      capability: unavailable,
      testOnlyAdmission: admittedForLifecycleTests,
    }), { wrapper });

    await act(async () => { await result.current.submit(job); });

    expect(result.current.phase).toBe("queued");
    expect(result.current.progress).toEqual({ completed: 0, total: 4 });
  });

  it("permits one active job, polls serially, and publishes only its complete result", async () => {
    vi.useFakeTimers();
    const firstPoll = deferred<RegionalGroundAuthorityJobStatus>();
    const authority = client({
      status: vi.fn()
        .mockImplementationOnce(() => firstPoll.promise)
        .mockResolvedValueOnce(status("succeeded", 4)),
    });
    const { result } = renderHook(() => useRegionalGroundExecutionController({
      client: authority,
      capability: unavailable,
      pollIntervalMs: 250,
      testOnlyAdmission: admittedForLifecycleTests,
    }));

    await act(async () => { await result.current.submit(job); });
    expect(result.current.progress).toEqual({ completed: 0, total: 4 });
    await expect(result.current.submit(job)).rejects.toThrow(/active/i);
    await act(async () => { vi.advanceTimersByTime(250); });
    expect(authority.status).toHaveBeenCalledTimes(1);
    await act(async () => { vi.advanceTimersByTime(1_000); });
    expect(authority.status).toHaveBeenCalledTimes(1);

    await act(async () => { firstPoll.resolve(status("running", 2)); });
    expect(result.current.progress).toEqual({ completed: 2, total: 4 });
    await act(async () => {
      vi.advanceTimersByTime(250);
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(result.current.phase).toBe("succeeded");

    expect(authority.result).toHaveBeenCalledTimes(1);
    expect(result.current.result).toEqual(resultEnvelope);
    expect(result.current.failure).toBeNull();
  });

  it("posts cancellation and retains exact cancellation progress", async () => {
    vi.useFakeTimers();
    const authority = client({
      cancel: vi.fn().mockResolvedValue(status("cancel_requested", 2)),
      status: vi.fn().mockResolvedValue(status("cancelled", 2)),
    });
    const { result } = renderHook(() => useRegionalGroundExecutionController({
      client: authority,
      capability: unavailable,
      pollIntervalMs: 250,
      testOnlyAdmission: admittedForLifecycleTests,
    }));

    await act(async () => { await result.current.submit(job); });
    await act(async () => { await result.current.cancel(); });
    expect(authority.cancel).toHaveBeenCalledWith(job, expect.any(AbortSignal));
    expect(result.current.progress).toEqual({ completed: 2, total: 4 });
    await act(async () => {
      vi.advanceTimersByTime(250);
      await Promise.resolve();
    });
    expect(result.current.phase).toBe("cancelled");

    expect(authority.result).not.toHaveBeenCalled();
    expect(result.current.progress).toEqual({ completed: 2, total: 4 });
  });

  it("exposes the exact typed terminal failure and never requests a result", async () => {
    const authority = client({ submit: vi.fn().mockResolvedValue(status("failed", 1)) });
    const { result } = renderHook(() => useRegionalGroundExecutionController({
      client: authority,
      capability: unavailable,
      testOnlyAdmission: admittedForLifecycleTests,
    }));

    await act(async () => { await result.current.submit(job); });

    expect(result.current.phase).toBe("failed");
    expect(result.current.failure).toEqual({ code: "execution_failed", stage: "executor" });
    expect(result.current.progress).toEqual({ completed: 1, total: 4 });
    expect(authority.result).not.toHaveBeenCalled();
  });

  it("aborts active polling and suppresses stale publication after reset", async () => {
    vi.useFakeTimers();
    const pending = deferred<RegionalGroundAuthorityJobStatus>();
    let signal: AbortSignal | undefined;
    const authority = client({
      status: vi.fn((_job, requestSignal) => {
        signal = requestSignal;
        return pending.promise;
      }),
    });
    const { result } = renderHook(() => useRegionalGroundExecutionController({
      client: authority,
      capability: unavailable,
      pollIntervalMs: 250,
      testOnlyAdmission: admittedForLifecycleTests,
    }));

    await act(async () => { await result.current.submit(job); });
    await act(async () => { vi.advanceTimersByTime(250); });
    expect(signal?.aborted).toBe(false);
    act(() => result.current.reset());
    expect(signal?.aborted).toBe(true);
    await act(async () => { pending.resolve(status("succeeded", 4)); });

    expect(result.current.phase).toBe("idle");
    expect(result.current.status).toBeNull();
    expect(authority.result).not.toHaveBeenCalled();
  });
});
