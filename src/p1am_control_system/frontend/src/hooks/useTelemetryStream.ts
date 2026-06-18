import { useEffect, useRef, useState } from "react";
import { TAG_COUNT } from "../lib/tags";
import { telemetryFrameSchema } from "../api/schemas";
import type { PowerSupplyStatus } from "../components/PowerSupplyControl";
import type { AlicatMFCState, ActiveAlarm } from "../api/schemas";

/**
 * Live telemetry from the `/api/stream` WebSocket (#3543).
 *
 * Extracted from App.tsx, which inlined the entire WS lifecycle plus the
 * frame-parsing/duck-typing logic. The frame is now validated with the
 * `telemetryFrameSchema` zod contract (#3545) instead of `as`-casting fields.
 */
export interface TelemetryState {
  tagValues: number[];
  history: number[][];
  tagsDict: Record<string, number>;
  alicats: AlicatMFCState[];
  activeAlarms: ActiveAlarm[];
  eStopActive: boolean;
  powerSupplyStatus: PowerSupplyStatus | undefined;
  isConnected: boolean;
}

const MAX_HISTORY = 1200;

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
  const [tagsDict, setTagsDict] = useState<Record<string, number>>({});
  const [alicats, setAlicats] = useState<AlicatMFCState[]>([]);
  const [activeAlarms, setActiveAlarms] = useState<ActiveAlarm[]>([]);
  const [eStopActive, setEStopActive] = useState<boolean>(false);
  const [powerSupplyStatus, setPowerSupplyStatus] = useState<
    PowerSupplyStatus | undefined
  >(undefined);
  const [isConnected, setIsConnected] = useState<boolean>(false);

  const wsRef = useRef<WebSocket | null>(null);
  const onConnectRef = useRef(options.onConnect);
  onConnectRef.current = options.onConnect;

  useEffect(() => {
    let disposed = false;
    let reconnectTimer: ReturnType<typeof setTimeout> | undefined;

    const pushTags = (values: number[]) => {
      setTagValues(values);
      setHistory((prev) => {
        const updated = [...prev, values];
        if (updated.length > MAX_HISTORY) {
          updated.shift();
        }
        return updated;
      });
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
        let raw: unknown;
        try {
          raw = JSON.parse(event.data);
        } catch {
          return;
        }

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
          return;
        }

        // Legacy fallback: a bare array of tag values.
        if (Array.isArray(raw) && raw.length === TAG_COUNT) {
          pushTags(raw as number[]);
        }
      };

      ws.onclose = () => {
        setIsConnected(false);
        if (!disposed) {
          reconnectTimer = setTimeout(connect, 3000);
        }
      };

      ws.onerror = () => {
        ws.close();
      };
    };

    connect();

    return () => {
      disposed = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      wsRef.current?.close();
    };
  }, []);

  return {
    tagValues,
    history,
    tagsDict,
    alicats,
    activeAlarms,
    eStopActive,
    powerSupplyStatus,
    isConnected,
    setAlicats,
    setActiveAlarms,
    setEStopActive,
  };
}
