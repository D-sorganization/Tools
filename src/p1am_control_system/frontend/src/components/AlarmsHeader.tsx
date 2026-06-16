import React, { useState } from "react";
import { AlertOctagon, CheckCircle, ChevronDown, ChevronRight } from "lucide-react";
import { ActiveAlarm } from "../App";

interface AlarmsHeaderProps {
  activeAlarms: ActiveAlarm[];
  onAcknowledgeAll: () => void;
}

/**
 * Subtle, compact system-status bar. Shows a one-line summary; the active
 * alarms expand into a scrollable list of clean rows (no per-row checkboxes).
 * Acknowledge-all is the only action, and only when something is unacked.
 */
export const AlarmsHeader: React.FC<AlarmsHeaderProps> = ({
  activeAlarms,
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
  const level = count === 0 ? "is-ok" : highestSeverity >= 2 ? "is-crit" : "is-warn";

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
            ? "All normal — no active alarms"
            : `${count} active alarm${count === 1 ? "" : "s"}` +
              (unacked > 0 ? ` · ${unacked} unacknowledged` : "")}
        </span>

        <span className="statusbar-spacer" />

        {unacked > 0 && (
          <button className="statusbar-ack" onClick={onAcknowledgeAll}>
            Acknowledge all
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
