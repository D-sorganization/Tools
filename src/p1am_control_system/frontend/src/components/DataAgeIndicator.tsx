import React from "react";
import {
  dataStatus,
  formatDataAge,
  type DataFreshness,
  type DataStatus,
} from "../lib/dataAge";

/**
 * Header liveness pill driven by DATA AGE, not by a boolean (#4010).
 *
 * The previous pill latched CONNECTED on any successful frame parse. Because
 * every telemetry field is optional, a dead backend's never-cleared `{}`
 * satisfied that test forever: the HMI stayed green while the trend appended
 * the same frozen value, which an operator reads as a rock-steady process.
 *
 * Showing the age makes the failure self-evident — "STALE DATA · 20 m 4 s ago"
 * cannot be misread as a healthy process — and gives the rest of the UI a
 * single state to grey its process values on.
 */

const LABELS: Record<DataStatus, string> = {
  live: "CONNECTED",
  stale: "STALE DATA",
  offline: "OFFLINE",
  simulated: "SIMULATED",
  held: "HELD DATA",
  fault: "SCAN FAULT",
  unknown: "SOURCE UNKNOWN",
};

const TITLES: Record<DataStatus, string> = {
  live: "Live telemetry is arriving from the backend.",
  stale:
    "No telemetry has arrived recently. The values on screen are FROZEN, not steady — treat every process value as unverified until this clears.",
  offline:
    "No telemetry has ever arrived this session. Nothing on screen reflects the plant.",
  simulated:
    "Frames are arriving on time but come from a simulator, not the plant.",
  held:
    "Frames are arriving on time but the backend is HOLDING the last good values: they are not fresh readings.",
  fault:
    "The backend scan is in FAULT: the values on screen are not trustworthy readings.",
  unknown:
    "Frames are arriving but do not say where their data came from. Treat values as unverified.",
};

interface Props {
  /** Freshness band derived from the data age. */
  freshness: DataFreshness;
  /**
   * `data_source` from the latest frame, passed through unvalidated. Required
   * (may be undefined) so no caller can silently omit it and read as live.
   */
  dataSource: string | undefined;
  /** Milliseconds since the last real frame, or undefined if none ever. */
  ageMs: number | undefined;
}

const DataAgeIndicatorImpl: React.FC<Props> = ({
  freshness,
  dataSource,
  ageMs,
}) => {
  const status = dataStatus(freshness, dataSource);
  return (
    <div
      className={`data-age data-age-${status}`}
      data-freshness={freshness}
      data-status={status}
      role="status"
      aria-live="polite"
      title={TITLES[status]}
    >
      <span className="status-indicator" />
      <span className="data-age-label">{LABELS[status]}</span>
      <span className="data-age-value mono-text">
        {freshness === "offline" ? "no data" : `${formatDataAge(ageMs)} ago`}
      </span>
    </div>
  );
};

/**
 * Memoized: the App tree re-renders on every ~10 Hz frame, but this pill only
 * changes when the age crosses a whole second or the band changes.
 */
export const DataAgeIndicator = React.memo(DataAgeIndicatorImpl);
