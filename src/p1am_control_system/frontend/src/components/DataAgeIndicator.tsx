import React from "react";
import { formatDataAge, type DataFreshness } from "../lib/dataAge";

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

const LABELS: Record<DataFreshness, string> = {
  live: "CONNECTED",
  stale: "STALE DATA",
  offline: "OFFLINE",
};

const TITLES: Record<DataFreshness, string> = {
  live: "Live telemetry is arriving from the backend.",
  stale:
    "No telemetry has arrived recently. The values on screen are FROZEN, not steady — treat every process value as unverified until this clears.",
  offline:
    "No telemetry has ever arrived this session. Nothing on screen reflects the plant.",
};

interface Props {
  /** Freshness band derived from the data age. */
  freshness: DataFreshness;
  /** Milliseconds since the last real frame, or undefined if none ever. */
  ageMs: number | undefined;
}

const DataAgeIndicatorImpl: React.FC<Props> = ({ freshness, ageMs }) => (
  <div
    className={`data-age data-age-${freshness}`}
    data-freshness={freshness}
    role="status"
    aria-live="polite"
    title={TITLES[freshness]}
  >
    <span className="status-indicator" />
    <span className="data-age-label">{LABELS[freshness]}</span>
    <span className="data-age-value mono-text">
      {freshness === "offline" ? "no data" : `${formatDataAge(ageMs)} ago`}
    </span>
  </div>
);

/**
 * Memoized: the App tree re-renders on every ~10 Hz frame, but this pill only
 * changes when the age crosses a whole second or the band changes.
 */
export const DataAgeIndicator = React.memo(DataAgeIndicatorImpl);
