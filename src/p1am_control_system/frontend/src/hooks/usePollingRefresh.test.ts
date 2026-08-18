import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook } from "@testing-library/react";
import { usePollingRefresh } from "./usePollingRefresh";

/**
 * Independent refresh loop (#4042a, #4011).
 *
 * The event log had exactly one writer — `fetchAlarmsAndEvents` — and exactly
 * one call site, inside `handleAcknowledgeAlarm`. The mount effect omitted it
 * and there was no interval, so the Events tab was blank until an unrelated
 * acknowledgement happened and frozen thereafter.
 *
 * The same loop is the alarm list's recovery path: nothing else polled
 * `/api/alarms/active`, so a stream-side alarm-map failure was permanent for
 * the session.
 */

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

describe("usePollingRefresh", () => {
  it("runs once immediately on mount", () => {
    const fn = vi.fn();
    renderHook(() => usePollingRefresh(fn, 1000));
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it("keeps refreshing on the interval", () => {
    const fn = vi.fn();
    renderHook(() => usePollingRefresh(fn, 1000));
    vi.advanceTimersByTime(3000);
    expect(fn).toHaveBeenCalledTimes(4); // mount + 3 ticks
  });

  it("stops when unmounted", () => {
    const fn = vi.fn();
    const { unmount } = renderHook(() => usePollingRefresh(fn, 1000));
    unmount();
    vi.advanceTimersByTime(5000);
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it("does not restart the interval when the callback identity changes", () => {
    // The App's fetcher is recreated on renders; a naive dependency on it would
    // reset the timer every ~10 Hz frame and the poll would never fire.
    const first = vi.fn();
    const second = vi.fn();
    const { rerender } = renderHook(({ fn }) => usePollingRefresh(fn, 1000), {
      initialProps: { fn: first as () => void },
    });
    expect(first).toHaveBeenCalledTimes(1);

    rerender({ fn: second as () => void });
    vi.advanceTimersByTime(1000);

    // The latest callback is used, and mounting did not happen twice.
    expect(second).toHaveBeenCalledTimes(1);
    expect(first).toHaveBeenCalledTimes(1);
  });

  it("can be disabled, and runs immediately once re-enabled", () => {
    const fn = vi.fn();
    const { rerender } = renderHook(
      ({ enabled }) => usePollingRefresh(fn, 1000, enabled),
      { initialProps: { enabled: false } },
    );
    vi.advanceTimersByTime(5000);
    expect(fn).not.toHaveBeenCalled();

    rerender({ enabled: true });
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it("rejects a non-positive interval (DbC — a 0 ms poll would spin)", () => {
    expect(() => renderHook(() => usePollingRefresh(vi.fn(), 0))).toThrow(RangeError);
  });
});
