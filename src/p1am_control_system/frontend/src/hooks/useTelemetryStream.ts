import { useEffect, useRef, useState } from "react";
import { TAG_COUNT } from "../lib/tags";
import { MAX_TREND_SAMPLES } from "../lib/trendTime";
import { telemetryFrameSchema } from "../api/schemas";
import type { PowerSupplyStatus } from "../components/PowerSupplyControl";
import type { TemperatureStatus } from "../components/TemperatureControl";
import type { AlicatMFCState, ActiveAlarm } from "../api/schemas";

/**
 * Live telemetry from the `/api/stream` WebSocket (#3543).
 *
 * Extracted from App.tsx, which inlined the entire WS lifecycle plus the
 * frame-parsing/duck-typing logic. The frame is now validated with the
 * `telemetryFrameSchema` zod contract (#3545) instead of `as`-casting fields.
 *
 * Resilience: some hosts can't hold a long-lived WebSocket (e.g. an embedded VS
 * Code Simple Browser webview, or any client behind a proxy that drops idle
 * sockets), and a backend restart drops every socket. So this hook also polls
 * the `/api/snapshot` HTTP endpoint whenever no frame has arrived recently — the
 * UI keeps updating over plain HTTP even with no usable WebSocket. A healthy
 * WebSocket keeps the data fresh and the poll is a no-op.
 */

/** Treat the stream as stale (fall back to HTTP polling) after this long. */
const STALE_MS = 3000;
/** How often to check staleness / poll the snapshot fallback. */
const POLL_MS = 1500;
export interface TelemetryState {
  tagValues: number[];
  history: number[][];
  /** Epoch-ms timestamp of each `history` frame (same length as `history`). */
  historyTimes: number[];
  tagsDict: Record<string, number>;
  alicats: AlicatMFCState[];
  activeAlarms: ActiveAlarm[];
  eStopActive: boolean;
  powerSupplyStatus: PowerSupplyStatus | undefined;
  temperatureStatus: TemperatureStatus | undefined;
  isConnected: boolean;
}

// Buffer enough history for the longest selectable trend window (5 min @ 10 Hz).
const MAX_HISTORY = MAX_TREND_SAMPLES;

export interface UseTelemetryStreamOptions {
  /** Called once when the socket first opens (e.g. to show a banner). */
  onConnect?: () => void;
}

/**
 * Subscribe to the live SCADA telemetry stream. Reconnects automatically on
 * close. Returns the latest telemetry snapshot plus setters the host needs to
 * keep alicats/alarms in sync with REST mutations.
 */
export function useTelemetryStream(
  options: UseTelemetryStreamOptions = {},
): TelemetryState & {
  setAlicats: React.Dispatch<React.SetStateAction<AlicatMFCState[]>>;
  setActiveAlarms: React.Dispatch<React.SetStateAction<ActiveAlarm[]>>;
  setEStopActive: React.Dispatch<React.SetStateAction<boolean>>;
} {
  const [tagValues, setTagValues] = useState<number[]>(
    Array(TAG_COUNT).fill(0.0),
  );
  const [history, setHistory] = useState<number[][]>([]);
  // Wall-clock (epoch ms) of each history frame, kept in lockstep with
  // `history` so trends can window/scale by real time rather than assuming a
  // fixed sample rate (the Pi polls below the nominal 10 Hz under load).
  const [historyTimes, setHistoryTimes] = useState<number[]>([]);
  const [tagsDict, setTagsDict] = useState<Record<string, number>>({});
  const [alicats, setAlicats] = useState<AlicatMFCState[]>([]);
  const [activeAlarms, setActiveAlarms] = useState<ActiveAlarm[]>([]);
  const [eStopActive, setEStopActive] = useState<boolean>(false);
  const [powerSupplyStatus, setPowerSupplyStatus] = useState<
    PowerSupplyStatus | undefined
  >(undefined);
  const [temperatureStatus, setTemperatureStatus] = useState<
    TemperatureStatus | undefined
  >(undefined);
  const [isConnected, setIsConnected] = useState<boolean>(false);

  const wsRef = useRef<WebSocket | null>(null);
  const onConnectRef = useRef(options.onConnect);
  onConnectRef.current = options.onConnect;

  useEffect(() => {
    let disposed = false;
    let reconnectTimer: ReturnType<typeof setTimeout> | undefined;
    let lastFrameAt = 0; // epoch ms of the last applied frame (WS or poll)

    const pushTags = (values: number[]) => {
      setTagValues(values);
      const stamp = Date.now();
      setHistory((prev) => {
        const updated = [...prev, values];
        if (updated.length > MAX_HISTORY) {
          updated.shift();
        }
        return updated;
      });
      setHistoryTimes((prev) => {
        const updated = [...prev, stamp];
        if (updated.length > MAX_HISTORY) {
          updated.shift();
        }
        return updated;
      });
    };

    // Apply one telemetry frame from either transport (WS message or snapshot
    // poll). Returns true if it was a recognized frame.
    const applyFrame = (raw: unknown): boolean => {
      const parsed = telemetryFrameSchema.safeParse(raw);
      if (parsed.success) {
        const frame = parsed.data;
        if (frame.tags && frame.tags.length === TAG_COUNT) {
          pushTags(frame.tags);
        }
        if (frame.tags_dict) setTagsDict(frame.tags_dict);
        if (frame.alicats) setAlicats(frame.alicats);
        if (frame.active_alarms) {
          setActiveAlarms(Object.values(frame.active_alarms));
        }
        if (typeof frame.e_stop_active === "boolean") {
          setEStopActive(frame.e_stop_active);
        }
        if (frame.power_supply) {
          setPowerSupplyStatus(frame.power_supply as PowerSupplyStatus);
        }
        if (frame.temperature) {
          setTemperatureStatus(frame.temperature as TemperatureStatus);
        }
        lastFrameAt = Date.now();
        setIsConnected(true);
        return true;
      }
      // Legacy fallback: a bare array of tag values.
      if (Array.isArray(raw) && raw.length === TAG_COUNT) {
        pushTags(raw as number[]);
        lastFrameAt = Date.now();
        setIsConnected(true);
        return true;
      }
      return false;
    };

    const connect = () => {
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const ws = new WebSocket(
        `${protocol}//${window.location.host}/api/stream`,
      );
      wsRef.current = ws;

      ws.onopen = () => {
        setIsConnected(true);
        onConnectRef.current?.();
      };

      ws.onmessage = (event) => {
        try {
          applyFrame(JSON.parse(event.data));
        } catch {
          /* malformed frame — ignore */
        }
      };

      ws.onclose = () => {
        if (!disposed) {
          reconnectTimer = setTimeout(connect, 3000);
        }
      };

      ws.onerror = () => {
        ws.close();
      };
    };

    // HTTP fallback: when no frame has arrived for STALE_MS (WS down/flaky, or a
    // webview that can't hold a socket), pull the cached snapshot over plain HTTP.
    const pollIfStale = async () => {
      if (disposed || Date.now() - lastFrameAt < STALE_MS) return;
      try {
        const res = await fetch("/api/snapshot");
        if (res.ok && applyFrame(await res.json())) return;
        setIsConnected(false);
      } catch {
        setIsConnected(false);
      }
    };

    connect();
    void pollIfStale(); // immediate snapshot so a fresh mount shows data fast
    const pollTimer = setInterval(pollIfStale, POLL_MS);

    return () => {
      disposed = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      clearInterval(pollTimer);
      wsRef.current?.close();
    };
  }, []);

  return {
    tagValues,
    history,
    historyTimes,
    tagsDict,
    alicats,
    activeAlarms,
    eStopActive,
    powerSupplyStatus,
    temperatureStatus,
    isConnected,
    setAlicats,
    setActiveAlarms,
    setEStopActive,
  };
}
