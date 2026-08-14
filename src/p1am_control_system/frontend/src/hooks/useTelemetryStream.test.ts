import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useTelemetryStream } from "./useTelemetryStream";
import { DATA_STALE_MS } from "../lib/dataAge";
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
  it("opens a socket and announces the open, but an open socket is NOT data", () => {
    const onConnect = vi.fn();
    const { result } = renderHook(() => useTelemetryStream({ onConnect }));

    expect(MockWebSocket.instances).toHaveLength(1);
    expect(result.current.isConnected).toBe(false);

    act(() => {
      MockWebSocket.instances[0].onopen?.();
    });

    // The socket handshake succeeded — the operator is told so — but no frame
    // has arrived, so the HMI must NOT claim the process is live (#4010).
    expect(onConnect).toHaveBeenCalledTimes(1);
    expect(result.current.isConnected).toBe(false);
    expect(result.current.dataAgeMs).toBeUndefined();
    expect(result.current.freshness).toBe("offline");
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

  // --- #4010: an empty payload is not a live frame -------------------------

  it("does NOT count an empty payload as a live frame", () => {
    const { result } = renderHook(() => useTelemetryStream());

    act(() => {
      MockWebSocket.instances[0].onmessage?.({ data: JSON.stringify({}) });
    });

    // `{}` parses cleanly against the all-optional frame schema, which is
    // exactly how a dead backend (latest_frame = {} forever) used to keep the
    // HMI green while the trend appended a frozen value.
    expect(result.current.isConnected).toBe(false);
    expect(result.current.dataAgeMs).toBeUndefined();
    expect(result.current.freshness).toBe("offline");
  });

  it("does NOT count a payload of unrecognised fields as a live frame", () => {
    const { result } = renderHook(() => useTelemetryStream());
    act(() => {
      MockWebSocket.instances[0].onmessage?.({
        data: JSON.stringify({ some_other_key: 1 }),
      });
    });
    expect(result.current.isConnected).toBe(false);
  });

  it("counts a frame with any recognised field as live, with age 0", () => {
    const { result } = renderHook(() => useTelemetryStream());
    act(() => {
      MockWebSocket.instances[0].onmessage?.({
        data: JSON.stringify({ e_stop_active: false }),
      });
    });
    expect(result.current.isConnected).toBe(true);
    expect(result.current.dataAgeMs).toBe(0);
    expect(result.current.freshness).toBe("live");
  });

  it("ages a frozen stream out of CONNECTED instead of latching green forever", () => {
    vi.useFakeTimers();
    try {
      const { result } = renderHook(() => useTelemetryStream());
      const tags = Array.from({ length: TAG_COUNT }, () => 1);
      act(() => {
        MockWebSocket.instances[0].onmessage?.({ data: JSON.stringify({ tags }) });
      });
      expect(result.current.isConnected).toBe(true);

      // The poll loop dies on the backend: no further frames, HTTP 200 `{}` or
      // nothing at all. The age must climb and the UI must stop claiming live.
      act(() => {
        vi.advanceTimersByTime(DATA_STALE_MS + 2_000);
      });

      expect(result.current.dataAgeMs).toBeGreaterThanOrEqual(DATA_STALE_MS);
      expect(result.current.freshness).toBe("stale");
      expect(result.current.isConnected).toBe(false);
    } finally {
      vi.useRealTimers();
    }
  });

  // --- #4011: one malformed alarm must not erase the alarm list ------------

  it("keeps the GOOD alarms when one entry in the map is malformed", () => {
    const { result } = renderHook(() => useTelemetryStream());
    const good = {
      tag_id: "TAG_0",
      state: "HiHi",
      severity: 2,
      acknowledged: false,
      timestamp: "2026-01-01T00:00:00Z",
    };

    act(() => {
      MockWebSocket.instances[0].onmessage?.({
        data: JSON.stringify({
          active_alarms: {
            TAG_0: good,
            BROKEN: { nope: true },
            TAG_9: { ...good, tag_id: "TAG_9" },
          },
        }),
      });
    });

    // Previously the whole record collapsed to undefined, `if (frame.
    // active_alarms)` was false, setActiveAlarms was never called again, and
    // the header kept saying "All normal — no active alarms".
    expect(result.current.activeAlarms.map((a) => a.tag_id).sort()).toEqual([
      "TAG_0",
      "TAG_9",
    ]);
    expect(result.current.droppedAlarmCount).toBe(1);
  });

  it("clears the degraded-data count once a clean alarm map arrives", () => {
    const { result } = renderHook(() => useTelemetryStream());
    const good = {
      tag_id: "TAG_0",
      state: "HiHi",
      severity: 2,
      acknowledged: false,
      timestamp: "2026-01-01T00:00:00Z",
    };
    act(() => {
      MockWebSocket.instances[0].onmessage?.({
        data: JSON.stringify({ active_alarms: { BROKEN: { nope: true } } }),
      });
    });
    expect(result.current.droppedAlarmCount).toBe(1);

    act(() => {
      MockWebSocket.instances[0].onmessage?.({
        data: JSON.stringify({ active_alarms: { TAG_0: good } }),
      });
    });
    expect(result.current.droppedAlarmCount).toBe(0);
    expect(result.current.activeAlarms).toHaveLength(1);
  });

  it("applies an EMPTY alarm map as 'alarms cleared', not as 'no update'", () => {
    const { result } = renderHook(() => useTelemetryStream());
    const good = {
      tag_id: "TAG_0",
      state: "HiHi",
      severity: 2,
      acknowledged: false,
      timestamp: "2026-01-01T00:00:00Z",
    };
    act(() => {
      MockWebSocket.instances[0].onmessage?.({
        data: JSON.stringify({ active_alarms: { TAG_0: good } }),
      });
    });
    expect(result.current.activeAlarms).toHaveLength(1);

    act(() => {
      MockWebSocket.instances[0].onmessage?.({
        data: JSON.stringify({ active_alarms: {} }),
      });
    });
    expect(result.current.activeAlarms).toHaveLength(0);
  });
});
