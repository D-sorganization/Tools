import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import {
  qualifiedRegionalGroundAuthorityCapability,
  type RegionalGroundAuthorityCapability,
} from "../model/regionalGroundAuthority";
import { STATIC_INSPECTION_WEB_RUNTIME } from "../model/webRuntime";
import { useRegionalGroundAuthority } from "./useRegionalGroundAuthority";

const capability = (detail: string): RegionalGroundAuthorityCapability => ({
  schema_version: "rate-of-closure/regional-ground-authority-capability/v1",
  authority_id: "rate-of-closure-python-authority",
  authority_version: "1",
  available: false,
  regional_ground_execution: false,
  reason_code: "runner_not_started",
  detail,
});

const deferred = <T,>() => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((accept) => { resolve = accept; });
  return { promise, resolve };
};

afterEach(() => {
  vi.useRealTimers();
});

describe("regional-ground authority polling lifecycle", () => {
  it("performs zero authority queries in static-inspection mode", async () => {
    const query = vi.fn().mockResolvedValue(
      qualifiedRegionalGroundAuthorityCapability(),
    );
    const { result } = renderHook(() => useRegionalGroundAuthority({
      query,
      runtime: STATIC_INSPECTION_WEB_RUNTIME,
    }));

    await waitFor(() => expect(result.current.checking).toBe(false));
    expect(query).not.toHaveBeenCalled();
    expect(result.current.capability.reason_code).toBe("static_inspection");
    expect(result.current.controls.submitEnabled).toBe(false);
  });

  it("keeps every execution control disabled while capability is false", async () => {
    const query = vi.fn().mockResolvedValue(capability("Unavailable."));
    const { result } = renderHook(() => useRegionalGroundAuthority({
      query,
      pollIntervalMs: 1_000,
    }));

    await waitFor(() => expect(result.current.checking).toBe(false));
    expect(result.current.controls).toEqual({
      submitEnabled: false,
      statusEnabled: false,
      cancelEnabled: false,
      resultEnabled: false,
    });
  });

  it("enables submission only for exact qualified execution evidence", async () => {
    const query = vi.fn().mockResolvedValue(
      qualifiedRegionalGroundAuthorityCapability(),
    );
    const { result } = renderHook(() => useRegionalGroundAuthority({
      query,
      pollIntervalMs: 1_000,
    }));

    await waitFor(() => expect(result.current.checking).toBe(false));
    expect(result.current.capability.available).toBe(true);
    expect(result.current.capability.reason_code).toBe("qualified_execution_profile");
    expect(result.current.controls).toEqual({
      submitEnabled: true,
      statusEnabled: false,
      cancelEnabled: false,
      resultEnabled: false,
    });
  });

  it("polls one request at a time and aborts the active request on cleanup", async () => {
    vi.useFakeTimers();
    const second = deferred<RegionalGroundAuthorityCapability>();
    const signals: AbortSignal[] = [];
    const query = vi.fn((signal: AbortSignal) => {
      signals.push(signal);
      return signals.length === 1
        ? Promise.resolve(capability("First."))
        : second.promise;
    });
    const { unmount } = renderHook(() => useRegionalGroundAuthority({
      query,
      pollIntervalMs: 1_000,
    }));

    await act(async () => { await Promise.resolve(); });
    await act(async () => { vi.advanceTimersByTime(1_000); });
    expect(query).toHaveBeenCalledTimes(2);
    expect(signals[0].aborted).toBe(false);
    expect(signals[1].aborted).toBe(false);

    unmount();
    expect(signals[1].aborted).toBe(true);
    await act(async () => { vi.advanceTimersByTime(5_000); });
    expect(query).toHaveBeenCalledTimes(2);
  });

  it("never publishes a stale response after the query authority changes", async () => {
    const first = deferred<RegionalGroundAuthorityCapability>();
    const firstQuery = vi.fn(() => first.promise);
    const secondQuery = vi.fn().mockResolvedValue(capability("Current authority."));
    const { result, rerender } = renderHook(
      ({ query }) => useRegionalGroundAuthority({ query, pollIntervalMs: 1_000 }),
      { initialProps: { query: firstQuery } },
    );

    rerender({ query: secondQuery });
    await waitFor(() => expect(result.current.capability.detail).toBe("Current authority."));
    await act(async () => { first.resolve(capability("Stale authority.")); });

    expect(result.current.capability.detail).toBe("Current authority.");
  });
});
