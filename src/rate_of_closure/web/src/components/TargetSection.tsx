/**
 * Target-region editor + containment stats for the flight view
 * (epic #4125 H7b) — web twin of the PyQt6 TargetPanel: kind picker,
 * geometry entries (the place/edit seam; the top-down canvas renders
 * the region live), and a containment readout for the latest landing
 * point. 'Optimize to Target' lives on the Solver section, which
 * receives the same region.
 */

import {
  signedDistance,
  type TargetKind,
  type TargetRegionTs,
} from "../model/targets";
import { DISTANCE_UNITS, formatDistanceM } from "../model/units";
import { DecimalInput } from "./DecimalInput";

interface Props {
  target: TargetRegionTs;
  onChange: (target: TargetRegionTs) => void;
  /** Latest landing point (carry, + right lateral) [m], if a run exists. */
  landing?: { carryM: number; lateralM: number };
  /** Distance display unit (#4125 H6, yards default); region stays SI. */
  unit?: string;
}

const inputClass =
  "no-spinner w-20 rounded border border-slate-700 bg-slate-800 px-2 " +
  "py-1 text-slate-100 focus:border-blue-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500";

export function TargetSection({ target, onChange, landing, unit = "yd" }: Props) {
  const factor = DISTANCE_UNITS[unit] ?? 1.0;
  const patch = (updates: Partial<TargetRegionTs>) =>
    onChange({ ...target, ...updates });

  const field = (
    label: string,
    value: number,
    title: string,
    key: keyof TargetRegionTs,
  ) => (
    <label className="flex items-center gap-1 text-slate-300" title={title}>
      {label} ({unit})
      <DecimalInput
        value={Number((value / factor).toFixed(1))}
        aria-label={`${label} ${unit}`}
        min={key === "lateralM" ? undefined : Number.EPSILON}
        title={title}
        onCommit={(parsed) => patch({ [key]: parsed * factor })}
        className={inputClass}
      />
    </label>
  );

  const distance = landing
    ? signedDistance(target, landing.carryM, landing.lateralM)
    : null;

  return (
    <div className="mb-3 flex flex-wrap items-center gap-3 text-sm">
      <label
        className="flex items-center gap-1 text-slate-300"
        title={
          "Green: a circle at a distance with a radius (approach target). " +
          "Fairway: a carry-distance band with a half-width about the " +
          "target line (tee-shot target)."
        }
      >
        Target
        <select
          value={target.kind}
          onChange={(e) => patch({ kind: e.target.value as TargetKind })}
          title="Target-region kind: green circle or fairway corridor"
          className="rounded border border-slate-700 bg-slate-800 px-2 py-1 text-slate-100"
        >
          <option value="green">Green (circle)</option>
          <option value="fairway">Fairway (corridor)</option>
        </select>
      </label>
      {field(
        "Distance",
        target.distanceM,
        "Downrange center of the target region [m]",
        "distanceM",
      )}
      {target.kind === "green" ? (
        <>
          {field("Radius", target.radiusM, "Green radius [m]", "radiusM")}
          {field(
            "Lateral",
            target.lateralM,
            "Lateral center offset [m], + right of the target line",
            "lateralM",
          )}
        </>
      ) : (
        <>
          {field(
            "Band ±",
            target.bandHalfLengthM,
            "Half-length of the acceptable carry band [m]",
            "bandHalfLengthM",
          )}
          {field(
            "Width ±",
            target.halfWidthM,
            "Fairway half-width about the target line [m]",
            "halfWidthM",
          )}
        </>
      )}
      {distance !== null && (
        <span
          aria-live="polite"
          title={
            "Signed distance from the latest landing point to the target " +
            "boundary: negative or zero means the shot holds the target."
          }
          className={
            distance <= 0 ? "text-emerald-300" : "text-amber-300/90"
          }
        >
          {distance <= 0
            ? `Holding the target (${formatDistanceM(-distance, unit)} inside)`
            : `Outside the target by ${formatDistanceM(distance, unit)}`}
        </span>
      )}
    </div>
  );
}
