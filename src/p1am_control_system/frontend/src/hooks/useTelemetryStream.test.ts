import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useTelemetryStream } from "./useTelemetryStream";
import { TAG_COUNT } from "../lib/tags";

/** Minimal controllable WebSocket stand-in. */
class MockWebSocket {
  static instances: MockWebSocket[] = [];
  onopen: (() => void) | null = null;
  onmessage: ((e: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  close = vi.fn();
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
});
