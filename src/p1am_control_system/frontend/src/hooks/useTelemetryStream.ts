import { useEffect, useRef, useState } from "react";
import { TAG_COUNT } from "../lib/tags";
import { SAMPLES_PER_SECOND } from "../lib/trendTime";
import {
  hasTelemetryContent,
  partitionAlarmMap,
  telemetryFrameSchema,
} from "../api/schemas";
import {
  DATA_STALE_MS,
  dataFreshness,
  type DataFreshness,
} from "../lib/dataAge";
import type { PowerSupplyStatus } from "../components/PowerSupplyControl";
import type { TemperatureStatus } from "../types";
import type { AlicatMFCState, ActiveAlarm, CommsHealth } from "../api/schemas";
import { authHeaders, getApiKey, promptForApiKey } from "../api/credentials";

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
/**
 * How often the exposed data age is recomputed. The age is what the whole UI
 * greys out on, so it must keep climbing while NOTHING arrives — a clock driven
 * only by incoming frames would freeze exactly when it matters most (#4010).
 */
const AGE_TICK_MS = 500;

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
  /**
   * Milliseconds since the last frame that carried recognised telemetry, or
   * `undefined` when no such frame has ever arrived. This — not a boolean — is
   * what the UI must render and grey out on.
   */
  dataAgeMs: number | undefined;
  /** {@link dataAgeMs} classified into live / stale / offline. */
  freshness: DataFreshness;
  /**
   * Convenience alias for `freshness === "live"`. Retained so existing call
   * sites keep compiling, but it is now derived from the age: an open socket
   * with no data, or a frozen stream, is NOT connected.
   */
  isConnected: boolean;
  /** Per-link comms quality reported by the backend, when it supplies it. */
  commsHealth: CommsHealth | undefined;
  /**
   * How many entries of the most recent `active_alarms` map failed validation
   * and were dropped. Non-zero means the alarm list on screen is incomplete and
   * the operator must be told (#4011).
   */
  droppedAlarmCount: number;
}

// In-memory live buffer bound. The Pi's browser only needs the recent tail for
// the live TrendChart (which caps its own window at 300 s) and SignalDiagnostics;
// deeper windows are served on demand by useTrendBackfill from the historian, so
// there is no reason to hold an hour of frames in JS heap. 6000 = 10 min @ 10 Hz
// keeps allocation and per-frame slice cost Pi-sane.
const MAX_HISTORY = 10 * 60 * SAMPLES_PER_SECOND;

export interface UseTelemetryStreamOptions {
  /** Called once when the socket first opens (e.g. to show a banner). */
  onConnect?: () => void;
}

/** Shallow, one-level equality of two plain objects (status frames). */
function shallowObjEqual(
  a: Record<string, unknown> | undefined,
  b: Record<string, unknown> | undefined,
): boolean {
  if (a === b) return true;
  if (!a || !b) return false;
  const ak = Object.keys(a);
  if (ak.length !== Object.keys(b).length) return false;
  for (const k of ak) {
    if (a[k] !== b[k]) return false;
  }
  return true;
}

/** Shallow equality of two object lists: same length + per-element shape. */
function shallowListEqual<T extends Record<string, unknown>>(
  a: readonly T[],
  b: readonly T[],
): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (!shallowObjEqual(a[i], b[i])) return false;
  }
  return true;
}

/** Shallow equality of two flat numeric dictionaries (tags_dict frames). */
function numberDictEqual(
  a: Record<string, number>,
  b: Record<string, number>,
): boolean {
  return shallowObjEqual(a, b);
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
  /**
   * Reset the degraded-alarm-data count. The host calls this after replacing
   * the alarm list wholesale from `/api/alarms/active`: that list is complete
   * by construction, so the stream's stale drop count must not keep warning
   * about it (the next stream frame re-raises the warning if still degraded).
   */
  setDroppedAlarmCount: React.Dispatch<React.SetStateAction<number>>;
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
  // Epoch-ms of the last frame that actually carried telemetry (undefined =
  // never), plus a ticking clock so the derived age keeps climbing with no
  // traffic at all. Both are state (not refs) because the whole UI renders off
  // the resulting age.
  const [lastFrameAt, setLastFrameAt] = useState<number | undefined>(undefined);
  const [nowMs, setNowMs] = useState<number>(() => Date.now());
  const [droppedAlarmCount, setDroppedAlarmCount] = useState<number>(0);
  const [commsHealth, setCommsHealth] = useState<CommsHealth | undefined>(
    undefined,
  );
  const [isConnected, setIsConnected] = useState<boolean>(false);

  const wsRef = useRef<WebSocket | null>(null);
  const onConnectRef = useRef(options.onConnect);
  onConnectRef.current = options.onConnect;

  useEffect(() => {
    let disposed = false;
    let reconnectTimer: ReturnType<typeof setTimeout> | undefined;
    // Epoch ms of the last applied frame (WS or poll); undefined until one with
    // real content arrives. Mirrored into state by markLive() for rendering.
    let lastFrameAt: number | undefined;

    const pushTags = (values: number[]) => {
      setTagValues(values);
      const stamp = Date.now();
      // Bounded append: once at capacity, drop the single oldest frame with a
      // tail `slice` (O(MAX_HISTORY)) instead of `[...prev, v]` + `shift()`,
      // which reallocated and index-shifted the whole array every 100 ms.
      setHistory((prev) =>
        prev.length >= MAX_HISTORY
          ? [...prev.slice(prev.length - MAX_HISTORY + 1), values]
          : [...prev, values],
      );
      setHistoryTimes((prev) =>
        prev.length >= MAX_HISTORY
          ? [...prev.slice(prev.length - MAX_HISTORY + 1), stamp]
          : [...prev, stamp],
      );
    };

    /** Stamp the data-age clock. Only a frame with real content may do this. */
    const markLive = () => {
      const stamp = Date.now();
      lastFrameAt = stamp;
      setLastFrameAt(stamp);
      setNowMs(stamp);
    };

    // Apply one telemetry frame from either transport (WS message or snapshot
    // poll). Returns true if it was a recognized frame CARRYING telemetry.
    //
    // Parse success is deliberately NOT the liveness test: every field of the
    // frame schema is optional, so `{}` — which is exactly what the backend's
    // never-cleared `latest_frame` serves after its poll loop dies — parses
    // cleanly. Treating that as a live frame is what kept the HMI green, the
    // status pill CONNECTED, and the trend appending a frozen value forever.
    const applyFrame = (raw: unknown): boolean => {
      const parsed = telemetryFrameSchema.safeParse(raw);
      if (parsed.success && hasTelemetryContent(parsed.data)) {
        const frame = parsed.data;
        if (frame.tags && frame.tags.length === TAG_COUNT) {
          pushTags(frame.tags);
        }
        // Non-live fields: only setState when the value actually changed, so a
        // memoized consumer (AlarmsHeader, PowerSupplyControl, …) can skip a
        // re-render even though a new frame arrives every ~100 ms. React bails
        // out of an identical-reference update, so returning `prev` is a no-op.
        if (frame.tags_dict) {
          const next = frame.tags_dict;
          setTagsDict((prev) => (numberDictEqual(prev, next) ? prev : next));
        }
        if (frame.alicats) {
          const next = frame.alicats;
          setAlicats((prev) =>
            shallowListEqual(
              prev as Record<string, unknown>[],
              next as Record<string, unknown>[],
            )
              ? prev
              : next,
          );
        }
        if (frame.active_alarms) {
          // Per-entry resilience: malformed entries parse to `undefined` and
          // are dropped individually, so one bad alarm no longer erases the
          // whole list. The drop count is surfaced so the operator is told the
          // list on screen is incomplete rather than believing it is empty.
          const { alarms: next, droppedIds } = partitionAlarmMap(frame.active_alarms);
          setActiveAlarms((prev) =>
            shallowListEqual(
              prev as Record<string, unknown>[],
              next as Record<string, unknown>[],
            )
              ? prev
              : next,
          );
          setDroppedAlarmCount((prev) =>
            prev === droppedIds.length ? prev : droppedIds.length,
          );
        }
        if (typeof frame.e_stop_active === "boolean") {
          setEStopActive(frame.e_stop_active);
        }
        if (frame.power_supply) {
          const next = frame.power_supply as PowerSupplyStatus;
          setPowerSupplyStatus((prev) =>
            shallowObjEqual(
              prev as Record<string, unknown> | undefined,
              next as unknown as Record<string, unknown>,
            )
              ? prev
              : next,
          );
        }
        if (frame.temperature) {
          const next = frame.temperature as TemperatureStatus;
          setTemperatureStatus((prev) =>
            shallowObjEqual(
              prev as Record<string, unknown> | undefined,
              next as unknown as Record<string, unknown>,
            )
              ? prev
              : next,
          );
        }
        markLive();
        if (frame.comms_health) {
          const next = frame.comms_health;
          setCommsHealth((prev) =>
            shallowObjEqual(
              prev as unknown as Record<string, unknown> | undefined,
              next as unknown as Record<string, unknown>,
            )
              ? prev
              : next,
          );
        }
        lastFrameAt = Date.now();
        setIsConnected(true);
        return true;
      }
      // Legacy fallback: a bare array of tag values.
      if (Array.isArray(raw) && raw.length === TAG_COUNT) {
        pushTags(raw as number[]);
        markLive();
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
        // Authenticate with the FIRST frame rather than an `api_key` query
        // parameter: a query string lands in proxy/access logs and browser
        // history, a frame does not (#4007). Send an empty frame when we have
        // no key so the backend closes with 1008 straight away instead of
        // leaving a silently unauthorized socket open.
        ws.send(getApiKey() ?? "");
        // Announce the handshake, but do NOT claim the process is live: a
        // socket that opens and then delivers nothing is precisely the failure
        // this hook now has to make visible.
        onConnectRef.current?.();
      };

      ws.onmessage = (event) => {
        try {
          applyFrame(JSON.parse(event.data));
        } catch {
          /* malformed frame — ignore */
        }
      };

      ws.onclose = (event) => {
        // 1008 = policy violation: the backend rejected our credential. Ask for
        // it once before reconnecting, otherwise we reconnect-and-fail forever
        // and the operator only sees a dead display (#4007).
        if (!disposed && event?.code === 1008) {
          promptForApiKey();
        }
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
    //
    // A failed (or contentless) poll no longer needs to flip any flag: the age
    // clock below is already climbing, so doing nothing here IS reporting the
    // outage. That is the whole point of driving the UI from an age.
    const pollIfStale = async () => {
      if (disposed || Date.now() - (lastFrameAt ?? 0) < STALE_MS) return;
      try {
        // /api/snapshot is now credential-gated by default (#4037).
        const res = await fetch("/api/snapshot", { headers: authHeaders() });
        if (res.ok) applyFrame(await res.json());
      } catch {
        /* snapshot unavailable — the data age keeps climbing */
      }
    };

    connect();
    void pollIfStale(); // immediate snapshot so a fresh mount shows data fast
    const pollTimer = setInterval(pollIfStale, POLL_MS);
    // Advance the clock the age is measured against, independently of traffic.
    const ageTimer = setInterval(() => setNowMs(Date.now()), AGE_TICK_MS);

    return () => {
      disposed = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      clearInterval(pollTimer);
      clearInterval(ageTimer);
      wsRef.current?.close();
    };
  }, []);

  // Derive the age from the two clocks. `Math.max` guards a backwards system
  // clock so the age can never go negative (dataFreshness treats that as a
  // contract violation rather than silently reporting "live").
  const dataAgeMs =
    lastFrameAt === undefined ? undefined : Math.max(0, nowMs - lastFrameAt);
  const freshness = dataFreshness(dataAgeMs, DATA_STALE_MS);

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
    dataAgeMs,
    freshness,
    isConnected: freshness === "live",
    droppedAlarmCount,
    commsHealth,
    isConnected,
    setAlicats,
    setActiveAlarms,
    setEStopActive,
    setDroppedAlarmCount,
  };
}
