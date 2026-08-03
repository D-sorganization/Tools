import { useId } from "react";
import type { CommsHealth } from "../api/schemas";

export interface CommsQualityBadgeProps {
  transportConnected: boolean;
  health: CommsHealth | undefined;
}

const COLORS: Record<CommsHealth["quality"] | "offline" | "waiting", string> = {
  good: "var(--color-success)",
  uncertain: "var(--color-warning)",
  bad: "var(--color-error)",
  stale: "var(--color-error)",
  simulated: "var(--accent-cyan)",
  offline: "var(--color-error)",
  waiting: "var(--color-warning)",
};

function label(
  transportConnected: boolean,
  health: CommsHealth | undefined,
): string {
  if (!transportConnected) return "OFFLINE";
  if (!health) return "WAITING DATA";
  const labels: Record<CommsHealth["quality"], string> = {
    good: "DATA GOOD",
    uncertain: "DATA UNCERTAIN",
    bad: "DATA BAD",
    stale: "DATA STALE",
    simulated: "SIMULATED DATA",
  };
  return labels[health.quality];
}

function description(health: CommsHealth | undefined): string {
  if (!health) return "No qualified process-data frame has arrived.";
  const reason = health.diagnostic_reason ?? "no diagnostic reason";
  const sequence = health.sequence === null ? "no sequence" : `sequence ${health.sequence}`;
  return `${reason}; ${sequence}; source ${health.source}`;
}

export function CommsQualityBadge({
  transportConnected,
  health,
}: CommsQualityBadgeProps) {
  const descriptionId = useId();
  const state = !transportConnected ? "offline" : (health?.quality ?? "waiting");
  return (
    <span
      role="status"
      aria-live="polite"
      aria-describedby={descriptionId}
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: "0.35rem",
        color: COLORS[state],
        fontSize: "0.75rem",
        fontWeight: 800,
        letterSpacing: "0.03em",
      }}
    >
      <span aria-hidden="true">●</span>
      {label(transportConnected, health)}
      <span
        id={descriptionId}
        style={{
          position: "absolute",
          width: "1px",
          height: "1px",
          padding: 0,
          margin: "-1px",
          overflow: "hidden",
          clip: "rect(0, 0, 0, 0)",
          whiteSpace: "nowrap",
          border: 0,
        }}
      >
        {description(health)}
      </span>
    </span>
  );
}
