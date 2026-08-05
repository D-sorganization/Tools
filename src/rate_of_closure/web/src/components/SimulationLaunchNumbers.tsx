import { useState } from "react";

import {
  type SimulationLaunchTs,
  type SimulationRunTs,
} from "../model/simulation";
import { formatDistanceM } from "../model/units";

const ROWS: Array<{ key: keyof SimulationLaunchTs; label: string; unit: string }> = [
  { key: "ballSpeedMph", label: "Ball Speed", unit: "mph" },
  { key: "launchAngleDeg", label: "Launch Angle", unit: "°" },
  { key: "launchAzimuthDeg", label: "Launch Azimuth", unit: "°" },
  { key: "spinRpm", label: "Total Spin", unit: "rpm" },
  { key: "carryM", label: "Carry", unit: "m" },
  { key: "maxHeightM", label: "Apex", unit: "m" },
  { key: "flightTimeS", label: "Flight Time", unit: "s" },
  { key: "landingAngleDeg", label: "Landing Angle", unit: "°" },
];

const GUIDANCE: Record<keyof SimulationLaunchTs, string> = {
  ballSpeedMph: "Ball speed immediately after separation. Reference frame: scalar speed magnitude in the ground-fixed app frame.",
  launchAngleDeg: "Vertical launch angle above the ground plane. Reference frame: positive from horizontal toward world +y (up).",
  launchAzimuthDeg: "Horizontal launch direction. Reference frame: 0° follows +x target line; positive points toward +z, right of target.",
  spinRpm: "Magnitude of the ball angular-velocity vector at launch; direction is reported separately by spin-axis tilt.",
  carryM: "Ground-projected distance from the tee to first landing. Reference frame: measured along +x target direction.",
  maxHeightM: "Maximum ball-center elevation during flight. Reference frame: world +y above the ground plane.",
  flightTimeS: "Elapsed time from separation until first ground contact in the flight integration.",
  landingAngleDeg: "Downward trajectory angle at first landing. Reference frame: magnitude below the horizontal ground plane.",
};

export function SimulationLaunchNumbers({
  run,
  distanceUnit,
}: {
  run: SimulationRunTs | null;
  distanceUnit: string;
}) {
  const [explained, setExplained] =
    useState<keyof SimulationLaunchTs>("ballSpeedMph");
  const launch = run?.launch;
  return (
    <div className="rounded-xl border border-slate-800/80 bg-slate-900/60 p-5 shadow-lg shadow-black/20 backdrop-blur">
      <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">Launch Numbers</h2>
      {run?.impactOutcome.status === "miss" && (
        <p role="status" className="mb-3 rounded-lg border border-amber-500/30 bg-amber-950/20 p-3 text-sm text-amber-200">
          No impact occurred. Launch and flight values are intentionally
          absent; the complete swing and closest-approach result are retained.
        </p>
      )}
      <div className="grid gap-2">
        {ROWS.map(({ key, label, unit }) => (
          <button
            key={key}
            type="button"
            onClick={() => setExplained(key)}
            aria-pressed={explained === key}
            title={GUIDANCE[key]}
            className="flex items-center justify-between rounded-lg border border-slate-800/80 bg-slate-900/50 px-3 py-2 text-left text-sm transition hover:border-sky-500/50 hover:bg-slate-800/70 focus-visible:outline focus-visible:outline-2 focus-visible:outline-sky-400 aria-pressed:border-sky-400/60 aria-pressed:bg-sky-500/10"
          >
            <span className="text-slate-300">{label} <span className="text-[10px] font-semibold uppercase text-sky-400">Details ›</span></span>
            <span className="font-semibold tabular-nums text-slate-100">
              {launch
                ? key === "carryM"
                  ? formatDistanceM(launch[key], distanceUnit)
                  : `${launch[key].toFixed(1)} ${unit}`
                : "—"}
            </span>
          </button>
        ))}
      </div>
      <p className="mt-3 rounded-lg border border-sky-500/25 bg-slate-950/60 p-3 text-xs text-slate-300">
        {GUIDANCE[explained]}
      </p>
      <p className="mt-3 text-xs text-slate-500">
        Physics parity-pinned against Python: double/triple pendulum RK4,
        rigid-body COR impact, and Waterloo/Penner flight.
      </p>
    </div>
  );
}
