import React, { useState } from "react";
import { AlertOctagon, CheckCircle, ChevronDown, ChevronRight } from "lucide-react";
import { ActiveAlarm } from "../App";

interface AlarmsHeaderProps {
  activeAlarms: ActiveAlarm[];
  /**
   * Entries of the last `active_alarms` map that failed validation and were
   * dropped. Non-zero means the list below is INCOMPLETE (#4011).
   */
  droppedAlarmCount: number;
  onAcknowledgeAll: () => void;
}

/**
 * Subtle, compact system-status bar. Shows a one-line summary; the active
 * alarms expand into a scrollable list of clean rows (no per-row checkboxes).
 * Acknowledge-all is the only action, and only when something is unacked.
 *
 * When any alarm entry was dropped by the parser, a degraded-data banner is
 * raised and the reassuring "All normal" summary is suppressed: an incomplete
 * alarm list must never be presented as a clean one.
 */
const AlarmsHeaderImpl: React.FC<AlarmsHeaderProps> = ({
  activeAlarms,
  droppedAlarmCount,
  onAcknowledgeAll,
}) => {
  const [expanded, setExpanded] = useState(false);

  let unacked = 0;
  let highestSeverity = 0;
  for (let i = 0; i < activeAlarms.length; i++) {
    if (!activeAlarms[i].acknowledged) unacked++;
    if (activeAlarms[i].severity > highestSeverity) {
      highestSeverity = activeAlarms[i].severity;
    }
  }

  const count = activeAlarms.length;
  const degraded = droppedAlarmCount > 0;
  // A known-incomplete list is never "ok", even when everything that parsed is
  // clear — that combination is exactly how the defect presented.
  const level = degraded
    ? "is-warn"
    : count === 0
      ? "is-ok"
      : highestSeverity >= 2
        ? "is-crit"
        : "is-warn";

  const fmtTime = (iso: string): string => {
    const d = new Date(iso);
    return Number.isNaN(d.getTime()) ? "" : d.toLocaleTimeString();
  };

  return (
    <>
      <div className={`statusbar ${level}`}>
        <span className="statusbar-dot" />
        {count === 0 ? (
          <CheckCircle size={14} color="var(--color-success)" />
        ) : (
          <AlertOctagon size={14} color="var(--color-error)" />
        )}
        <span className="statusbar-label">System</span>
        <span className="statusbar-summary">
          {count === 0
            ? degraded
              ? "Alarm data degraded"
              : "All normal — no active alarms"
            : `${count} active alarm${count === 1 ? "" : "s"}` +
              (unacked > 0 ? ` · ${unacked} unacknowledged` : "")}
        </span>

        <span className="statusbar-spacer" />

        {unacked > 0 && (
          <button className="statusbar-ack" onClick={onAcknowledgeAll}>
            Acknowledge All
          </button>
        )}
        {count > 0 && (
          <button
            className="statusbar-toggle"
            onClick={() => setExpanded((v) => !v)}
            aria-expanded={expanded}
          >
            {expanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
            {expanded ? "Hide" : "View"}
          </button>
        )}
      </div>

      {degraded && (
        <div className="alarm-degraded" role="alert">
          <AlertOctagon size={13} color="var(--color-warning)" />
          <span>
            Alarm data incomplete — {droppedAlarmCount} alarm
            {droppedAlarmCount === 1 ? "" : "s"} could not be read from the last
            frame and {droppedAlarmCount === 1 ? "is" : "are"} NOT shown. Check the
            PLC alarm summary directly.
          </span>
        </div>
      )}

      {expanded && count > 0 && (
        <div className="alarm-list">
          {activeAlarms.map((a) => (
            <div className="alarm-row" key={a.tag_id}>
              <span className={`alarm-sev sev-${a.severity >= 2 ? 2 : 1}`} />
              <span className="alarm-tag">{a.tag_id}</span>
              <span className="alarm-state">
                {a.state}
                {a.acknowledged ? " · ack" : ""}
              </span>
              <span className="alarm-time">{fmtTime(a.timestamp)}</span>
            </div>
          ))}
        </div>
      )}
    </>
  );
};

/**
 * Memoized: the App tree re-renders on every ~10 Hz telemetry frame, but the
 * alarm list changes rarely. `React.memo` skips this subtree when `activeAlarms`
 * (ref-stable from useTelemetryStream) and `onAcknowledgeAll` are unchanged.
 */
export const AlarmsHeader = React.memo(AlarmsHeaderImpl);
