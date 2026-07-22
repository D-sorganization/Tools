import React, { useEffect, useMemo, useState } from "react";
import { Activity, ChevronDown, Copy, Check } from "lucide-react";

import {
  effectiveHz,
  stalenessMs,
  formatStaleness,
  formatFeedLine,
} from "../lib/diagnostics";
import type { TemperatureStatus } from "./TemperatureControl";

// Curated "interesting" tags for the compact feed: thermocouples (0=K, 1=R),
// analog outputs (10,11), analog inputs (12,13), and the raw 0-5 V diagnostic
// channels (20-25). The full 32 are available via the toggle.
const KEY_TAGS = [0, 1, 10, 11, 12, 13, 20, 21, 22, 23, 24, 25];
const ALL_TAGS = Array.from({ length: 32 }, (_, i) => i);
const FEED_ROWS = 24;
const STALE_MS = 2000;

/**
 * Live-link diagnostics: a compact, always-visible health bar (connection,
 * effective rate, staleness, and the K/R/relay summary) that expands to a
 * scrollable feed of the most recent RAW PLC tag values, copyable for sharing.
 *
 * Shown on every HMI page so the data path (PLC → backend → HMI) can be
 * troubleshooted from wherever the operator is: is the stream live and fresh,
 * what rate is it really arriving at, and what are the raw values (a lone 0 is a
 * dropped read — shown verbatim here, unlike the deglitched trends).
 */
export const DiagnosticsFeed: React.FC<{
  history: number[][];
  historyTimes: number[];
  isConnected: boolean;
  temperature?: TemperatureStatus;
}> = ({ history, historyTimes, isConnected, temperature }) => {
  const [open, setOpen] = useState(false);
  const [showAll, setShowAll] = useState(false);
  const [copied, setCopied] = useState(false);

  // Re-render ~1 Hz so staleness keeps climbing even when the stream STALLS — a
  // stalled stream stops delivering frames, so nothing else would re-render it.
  const [, setTick] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setTick((t) => t + 1), 1000);
    return () => clearInterval(id);
  }, []);

  const now = Date.now();
  const stale = stalenessMs(historyTimes, now);
  const hz = effectiveHz(historyTimes);
  const indices = showAll ? ALL_TAGS : KEY_TAGS;

  const offline = !isConnected;
  const staleWarn = !offline && stale > STALE_MS;
  const statusColor = offline
    ? "var(--color-error)"
    : staleWarn
      ? "var(--color-warning)"
      : "var(--color-success)";
  const statusText = offline ? "OFFLINE" : staleWarn ? "STALE" : "LIVE";

  const k = temperature?.type_k_temp_c;
  const r = temperature?.type_r_temp_c;
  const delta =
    typeof k === "number" && typeof r === "number" ? Math.abs(k - r) : null;

  const feed = useMemo(() => {
    const start = Math.max(0, history.length - FEED_ROWS);
    const lines: string[] = [];
    for (let i = start; i < history.length; i++) {
      lines.push(formatFeedLine(historyTimes[i] ?? 0, history[i] ?? [], indices));
    }
    return lines;
  }, [history, historyTimes, indices]);

  const copyFeed = async (): Promise<void> => {
    const header =
      `# P1AM diagnostics  ${statusText}  ~${hz.toFixed(1)} Hz  ` +
      `fresh ${formatStaleness(stale)}  frames=${history.length}`;
    try {
      await navigator.clipboard?.writeText([header, ...feed].join("\n"));
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard unavailable — non-fatal */
    }
  };

  return (
    <section className="glass-panel" aria-label="Live link diagnostics">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        style={{
          display: "flex",
          alignItems: "center",
          gap: "0.6rem",
          width: "100%",
          background: "none",
          border: "none",
          cursor: "pointer",
          color: "var(--text-primary)",
          padding: 0,
          textAlign: "left",
          flexWrap: "wrap",
        }}
      >
        <ChevronDown
          size={14}
          aria-hidden
          style={{
            flexShrink: 0,
            transform: open ? "none" : "rotate(-90deg)",
            transition: "transform .15s",
            opacity: 0.7,
          }}
        />
        <Activity size={14} color={statusColor} aria-hidden />
        <span style={{ fontWeight: 700, color: statusColor }}>{statusText}</span>
        <span className="mono-text" style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>
          ~{hz.toFixed(1)} Hz · fresh {formatStaleness(stale)} · {history.length} frames
        </span>
        <span
          className="mono-text"
          style={{ fontSize: "0.72rem", marginLeft: "auto", color: "var(--text-muted)" }}
        >
          {typeof k === "number" ? `K ${k.toFixed(1)}°C` : "K —"} ·{" "}
          {typeof r === "number" ? `R ${r.toFixed(1)}°C` : "R —"}
          {delta !== null ? ` · Δ${delta.toFixed(1)}` : ""} · relay{" "}
          {temperature?.relay_on ? "ON" : "OFF"}
        </span>
      </button>

      {open && (
        <div style={{ marginTop: "0.6rem" }}>
          <div
            style={{
              display: "flex",
              gap: "1rem",
              flexWrap: "wrap",
              alignItems: "center",
              marginBottom: "0.4rem",
            }}
          >
            <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>
              Most recent raw PLC tag values — a lone <code>0</code> is likely a
              dropped read (the trends bridge these; here they're shown verbatim).
            </span>
            <label
              style={{
                fontSize: "0.72rem",
                display: "inline-flex",
                alignItems: "center",
                gap: "0.3rem",
              }}
            >
              <input
                type="checkbox"
                checked={showAll}
                onChange={(e) => setShowAll(e.target.checked)}
              />
              show all 32 tags
            </label>
            <button
              type="button"
              className="btn"
              style={{
                fontSize: "0.7rem",
                padding: "0.15rem 0.45rem",
                display: "inline-flex",
                alignItems: "center",
                gap: "0.25rem",
              }}
              onClick={copyFeed}
              title="Copy the diagnostics feed to share"
            >
              {copied ? <Check size={12} /> : <Copy size={12} />}
              {copied ? "Copied" : "Copy"}
            </button>
          </div>
          <pre
            aria-label="Recent raw tag values"
            style={{
              margin: 0,
              maxHeight: "11rem",
              overflow: "auto",
              fontFamily: "var(--font-mono)",
              fontSize: "0.68rem",
              lineHeight: 1.5,
              background: "var(--panel-bg)",
              border: "1px solid var(--panel-border)",
              borderRadius: "4px",
              padding: "0.4rem",
              whiteSpace: "pre",
            }}
          >
            {feed.length ? feed.join("\n") : "waiting for live data…"}
          </pre>
        </div>
      )}
    </section>
  );
};
