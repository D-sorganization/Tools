/**
 * Standalone Ball-Flight Explorer section (epic #4120, V2 web parity).
 *
 * Direct entry of launch conditions (ball speed with a unit drop-down,
 * launch angle, azimuth, spin, spin-axis tilt — sourced hover guidance
 * on every control) integrated with the Waterloo/Penner model and
 * rendered in the flight profile canvases with result rows. No swing
 * required. The 7-model picker and delivery mode stay Python-side
 * until the P7 WASM kernels land.
 */

import { useState } from "react";

import { FlightCanvases } from "./FlightCanvases";
import {
  directLaunch,
  exploreFlight,
  type FlightExplorationTs,
} from "../model/flightExplorer";
import { FIELD_GUIDANCE, formatDistanceM } from "../model/units";

const SPEED_UNITS: Record<string, number> = { mph: 1.0, "m/s": 2.236936292054402 };

const RESULT_ROWS: Array<{
  key: keyof FlightExplorationTs["metrics"];
  label: string;
  unit: string;
}> = [
  { key: "carryM", label: "Carry Distance", unit: "m" },
  { key: "maxHeightM", label: "Apex Height", unit: "m" },
  { key: "flightTimeS", label: "Flight Time", unit: "s" },
  { key: "landingAngleDeg", label: "Landing Angle", unit: "°" },
  { key: "lateralM", label: "Lateral Landing Offset", unit: "m" },
];

interface FieldSpec {
  key: "launchAngleDeg" | "azimuthDeg" | "spinRpm" | "spinAxisTiltDeg";
  label: string;
  unit: string;
  guidance: string;
}

const FIELDS: FieldSpec[] = [
  { key: "launchAngleDeg", label: "Launch Angle", unit: "deg", guidance: "fxLaunchAngle" },
  { key: "azimuthDeg", label: "Launch Azimuth", unit: "deg", guidance: "fxAzimuth" },
  { key: "spinRpm", label: "Total Spin", unit: "rpm", guidance: "fxSpinRpm" },
  { key: "spinAxisTiltDeg", label: "Spin-Axis Tilt", unit: "deg", guidance: "fxSpinAxisTilt" },
];

export function FlightExplorerPanel({
  distanceUnit = "yd",
}: {
  /** Ball-flight distance display unit (#4125 H6): yards default. */
  distanceUnit?: string;
} = {}) {
  const [speed, setSpeed] = useState(167.0);
  const [speedUnit, setSpeedUnit] = useState("mph");
  const [fields, setFields] = useState({
    launchAngleDeg: 10.9,
    azimuthDeg: 0.0,
    spinRpm: 2686.0,
    spinAxisTiltDeg: 0.0,
  });
  const [result, setResult] = useState<FlightExplorationTs | null>(null);
  const [error, setError] = useState<string | null>(null);

  const run = () => {
    try {
      const exploration = exploreFlight(
        directLaunch({
          ballSpeedMph: speed * (SPEED_UNITS[speedUnit] / SPEED_UNITS.mph),
          ...fields,
        }),
      );
      setResult(exploration);
      setError(null);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    }
  };

  return (
    <div className="grid gap-6 lg:grid-cols-[340px_1fr]">
      <section aria-label="Flight explorer inputs" className="min-w-0 space-y-4">
        <div className="rounded-xl border border-slate-800/80 bg-slate-900/60 p-5 shadow-lg shadow-black/20 backdrop-blur">
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">
            Launch Entry (No Swing Required)
          </h2>
          <label className="mb-2 block text-sm" title={FIELD_GUIDANCE.fxBallSpeed}>
            <span className="mb-1 flex justify-between text-slate-300">
              <span className="truncate" title="Ball Speed">
                Ball Speed
              </span>
            </span>
            <span className="flex min-w-0 gap-2">
              <input
                type="number"
                inputMode="decimal"
                value={speed}
                title={FIELD_GUIDANCE.fxBallSpeed}
                onChange={(e) => {
                  const parsed = Number(e.target.value);
                  if (Number.isFinite(parsed)) setSpeed(parsed);
                }}
                className="no-spinner w-full min-w-16 rounded border border-slate-700 bg-slate-800 px-2 py-1.5 text-slate-100 focus:border-blue-500 focus:outline-none"
              />
              <select
                value={speedUnit}
                title={FIELD_GUIDANCE.fxSpeedUnit}
                onChange={(e) => {
                  const next = e.target.value;
                  // Convert the displayed value in place (canonical mph).
                  const mph = speed * (SPEED_UNITS[speedUnit] / SPEED_UNITS.mph);
                  setSpeed(Number((mph * (SPEED_UNITS.mph / SPEED_UNITS[next])).toFixed(2)));
                  setSpeedUnit(next);
                }}
                className="min-w-16 rounded border border-slate-700 bg-slate-800 px-2 py-1.5 text-slate-100"
                aria-label="Ball speed unit"
              >
                {Object.keys(SPEED_UNITS).map((unit) => (
                  <option key={unit} value={unit}>
                    {unit}
                  </option>
                ))}
              </select>
            </span>
          </label>
          {FIELDS.map(({ key, label, unit, guidance }) => (
            <label key={key} className="mb-2 block text-sm" title={FIELD_GUIDANCE[guidance]}>
              <span className="mb-1 flex justify-between text-slate-300">
                <span className="truncate" title={label}>
                  {label}
                </span>
                <span className="text-slate-500">{unit}</span>
              </span>
              <input
                type="number"
                inputMode="decimal"
                value={fields[key]}
                title={FIELD_GUIDANCE[guidance]}
                onChange={(e) => {
                  const parsed = Number(e.target.value);
                  if (Number.isFinite(parsed)) {
                    setFields((f) => ({ ...f, [key]: parsed }));
                  }
                }}
                className="no-spinner w-full min-w-16 rounded border border-slate-700 bg-slate-800 px-2 py-1.5 text-slate-100 focus:border-blue-500 focus:outline-none"
              />
            </label>
          ))}
          <button
            type="button"
            onClick={run}
            title="Integrate the ball flight for the entered launch conditions"
            className="mt-1 w-full rounded-lg border border-sky-400/60 bg-sky-500/10 px-3 py-2 text-sm font-semibold text-sky-300 transition-all hover:bg-sky-500/20"
          >
            Run Flight
          </button>
          {error && (
            <p className="mt-2 text-xs text-rose-400" role="alert">
              {error}
            </p>
          )}
          <p className="mt-3 text-xs text-slate-500">
            Waterloo/Penner flight physics, parity-banded against the
            Python explorer (which adds the full 7-model literature picker
            and an impact-delivery entry mode; both arrive here with the
            P7 WASM kernels).
          </p>
        </div>

        <div className="rounded-xl border border-slate-800/80 bg-slate-900/60 p-5 shadow-lg shadow-black/20 backdrop-blur">
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">
            Flight Numbers
          </h2>
          <div className="grid gap-2">
            {RESULT_ROWS.map(({ key, label, unit }) => (
              <div
                key={key}
                className="flex min-w-0 items-center justify-between rounded-lg border border-slate-800/80 bg-slate-900/50 px-3 py-2 text-sm"
              >
                <span className="truncate text-slate-400" title={label}>
                  {label}
                </span>
                <span className="ml-2 min-w-16 text-right font-semibold tabular-nums text-slate-100">
                  {result
                    ? key === "carryM" || key === "lateralM"
                      ? `${result.metrics[key] >= 0 ? "+" : "-"}${formatDistanceM(
                          Math.abs(result.metrics[key]),
                          distanceUnit,
                        )}`
                      : `${result.metrics[key] >= 0 ? "+" : ""}${result.metrics[key].toFixed(1)} ${unit}`
                    : "—"}
                </span>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="min-w-0 space-y-3">
        <div className="rounded-xl border border-slate-800/80 bg-slate-900/60 p-4 shadow-lg shadow-black/20 backdrop-blur">
          <FlightCanvases
            points={result?.points ?? []}
            emptyText="Enter launch conditions and press Run Flight."
          />
        </div>
      </section>
    </div>
  );
}
