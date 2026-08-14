import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useTelemetryStream } from "./useTelemetryStream";
import { TAG_COUNT } from "../lib/tags";

/** Minimal controllable WebSocket stand-in. */
class MockWebSocket {
  static instances: MockWebSocket[] = [];
  onopen: (() => void) | null = null;
  onmessage: ((e: { data: string }) => void) | null = null;
  onclose: ((e?: { code?: number }) => void) | null = null;
  onerror: (() => void) | null = null;
  close = vi.fn();
  // The hook authenticates by sending the stored API key as its first frame
  // (#4007), so the stand-in must accept a send().
  send = vi.fn();
  constructor(public url: string) {
    MockWebSocket.instances.push(this);
  }
}

beforeEach(() => {
  MockWebSocket.instances = [];
  vi.stubGlobal("WebSocket", MockWebSocket as unknown as typeof WebSocket);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("useTelemetryStream", () => {
  it("opens a socket and reports connection on open", () => {
    const onConnect = vi.fn();
    const { result } = renderHook(() => useTelemetryStream({ onConnect }));

    expect(MockWebSocket.instances).toHaveLength(1);
    expect(result.current.isConnected).toBe(false);

    act(() => {
      MockWebSocket.instances[0].onopen?.();
    });

    expect(result.current.isConnected).toBe(true);
    expect(onConnect).toHaveBeenCalledTimes(1);
  });

  it("applies a validated telemetry frame", () => {
    const { result } = renderHook(() => useTelemetryStream());
    const tags = Array.from({ length: TAG_COUNT }, (_, i) => i);

    act(() => {
      MockWebSocket.instances[0].onmessage?.({
        data: JSON.stringify({ tags, e_stop_active: true }),
      });
    });

    expect(result.current.tagValues).toEqual(tags);
    expect(result.current.history).toHaveLength(1);
    expect(result.current.eStopActive).toBe(true);
  });

  it("ignores malformed JSON without throwing", () => {
    const { result } = renderHook(() => useTelemetryStream());
    act(() => {
      MockWebSocket.instances[0].onmessage?.({ data: "not json" });
    });
    expect(result.current.tagValues).toHaveLength(TAG_COUNT);
  });

  it("accepts a legacy bare-array frame", () => {
    const { result } = renderHook(() => useTelemetryStream());
    const tags = Array.from({ length: TAG_COUNT }, () => 1);
    act(() => {
      MockWebSocket.instances[0].onmessage?.({ data: JSON.stringify(tags) });
    });
    expect(result.current.tagValues).toEqual(tags);
  });

  it("keeps activeAlarms REF-STABLE when the frame is unchanged", () => {
    const { result } = renderHook(() => useTelemetryStream());
    const tags = Array.from({ length: TAG_COUNT }, () => 0);
    const alarms = {
      "1": {
        tag_id: "1",
        state: "HIGH",
        value: 42,
        severity: 2,
        acknowledged: false,
        timestamp: "2026-01-01T00:00:00Z",
      },
    };

    act(() => {
      MockWebSocket.instances[0].onmessage?.({
        data: JSON.stringify({ tags, active_alarms: alarms }),
      });
    });
    const first = result.current.activeAlarms;
    expect(first).toHaveLength(1);

    // A second, value-identical frame must NOT allocate a new array (so a
    // memoized consumer can bail out of re-rendering).
    act(() => {
      MockWebSocket.instances[0].onmessage?.({
        data: JSON.stringify({ tags, active_alarms: alarms }),
      });
    });
    expect(result.current.activeAlarms).toBe(first);

    // A changed alarm DOES produce a new reference.
    act(() => {
      MockWebSocket.instances[0].onmessage?.({
        data: JSON.stringify({
          tags,
          active_alarms: {
            "1": { ...alarms["1"], acknowledged: true },
          },
        }),
      });
    });
    expect(result.current.activeAlarms).not.toBe(first);
    expect(result.current.activeAlarms[0].acknowledged).toBe(true);
  });

  it("keeps alicats/tagsDict/status fields ref-stable on identical frames", () => {
    const { result } = renderHook(() => useTelemetryStream());
    const tags = Array.from({ length: TAG_COUNT }, () => 0);
    const frame = {
      tags,
      tags_dict: { FOO: 1, BAR: 2 },
      power_supply: { state: "on", voltage: 12 },
      temperature: { state: "on", temperature: 20 },
    };

    act(() => {
      MockWebSocket.instances[0].onmessage?.({ data: JSON.stringify(frame) });
    });
    const dict = result.current.tagsDict;
    const ps = result.current.powerSupplyStatus;
    const temp = result.current.temperatureStatus;

    act(() => {
      MockWebSocket.instances[0].onmessage?.({ data: JSON.stringify(frame) });
    });
    expect(result.current.tagsDict).toBe(dict);
    expect(result.current.powerSupplyStatus).toBe(ps);
    expect(result.current.temperatureStatus).toBe(temp);
  });

  it("bounds the live history buffer (MAX_HISTORY eviction)", () => {
    const { result } = renderHook(() => useTelemetryStream());
    const ws = MockWebSocket.instances[0];
    // 6000 = 10 min @ 10 Hz. Push well past the cap and confirm the buffer is
    // bounded and still tracks the most recent frame at its tail.
    const total = 6100;
    act(() => {
      for (let n = 0; n < total; n++) {
        const tags = Array.from({ length: TAG_COUNT }, () => n);
        ws.onmessage?.({ data: JSON.stringify({ tags }) });
      }
    });

    expect(result.current.history.length).toBe(6000);
    expect(result.current.historyTimes.length).toBe(6000);
    // Newest frame is retained at the tail.
    const last = result.current.history[result.current.history.length - 1];
    expect(last[0]).toBe(total - 1);
    // Oldest surviving frame is n = total - 6000 (older frames evicted).
    expect(result.current.history[0][0]).toBe(total - 6000);
  });
});
