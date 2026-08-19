/** Exact pointer and keyboard commit boundary for Simulation impact time. */
import { type ContactMode } from "../model/contact";
import { type SimulationRunTs } from "../model/simulation";
import { FIELD_GUIDANCE } from "../model/units";

const COMMIT_KEYS = new Set([
  "ArrowLeft",
  "ArrowRight",
  "ArrowUp",
  "ArrowDown",
  "Home",
  "End",
  "PageUp",
  "PageDown",
]);

interface Props {
  contactMode: ContactMode;
  run: SimulationRunTs | null;
  swingDuration: number;
  tauMs: number | null;
  onPreview: (valueMs: number) => void;
  onCommit: (valueMs: number) => void;
}

export function SimulationImpactTimeControl({
  contactMode,
  run,
  swingDuration,
  tauMs,
  onPreview,
  onCommit,
}: Props) {
  const disabled = contactMode === "fixed_ball_contact";
  const displayedValue = tauMs ?? (run
    ? (run.impactTimeS ?? run.impactOutcome.candidateTimeS) * 1000
    : swingDuration * 500);
  return (
    <label className="mb-3 block text-sm" title={FIELD_GUIDANCE.impactTimeScrub}>
      <span className="mb-1 flex justify-between text-slate-300">
        <span>Impact Time τ</span>
        <span className="text-slate-500">
          {tauMs === null ? "auto" : `${tauMs.toFixed(0)} ms`}
        </span>
      </span>
      <input
        type="range"
        aria-label="Impact Time"
        min={0}
        max={swingDuration * 1000}
        step={1}
        value={displayedValue}
        disabled={disabled}
        title={FIELD_GUIDANCE.impactTimeScrub}
        onChange={(event) => onPreview(Number(event.currentTarget.value))}
        onPointerUp={(event) => onCommit(Number(event.currentTarget.value))}
        onKeyUp={(event) => {
          if (COMMIT_KEYS.has(event.key)) {
            onCommit(Number(event.currentTarget.value));
          }
        }}
        className="w-full disabled:cursor-not-allowed disabled:opacity-40"
      />
      {disabled && (
        <span className="mt-1 block text-xs text-amber-300/90">
          Impact time is detected from closest approach in this mode.
        </span>
      )}
    </label>
  );
}
