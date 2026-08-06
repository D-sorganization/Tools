/**
 * Standalone Ball-Flight Explorer section (epic #4120, V2 web parity).
 *
 * Direct entry of launch conditions (ball speed with a unit drop-down,
 * launch angle, launch direction, spin, spin-axis tilt — sourced guidance
 * on every control) integrated with the Waterloo/Penner model and
 * rendered in the flight profile canvases with result rows. No swing
 * required. The 7-model picker and delivery mode stay Python-side
 * until the P7 WASM kernels land.
 */

import { useState } from "react";

import { DecimalInput } from "./DecimalInput";
import { FieldInfo } from "./FieldInfo";
import { FlightCanvases } from "./FlightCanvases";
import {
  compareWind,
  directLaunch,
  exploreFlight,
  type FlightExplorationTs,
  type WindComparisonTs,
} from "../model/flightExplorer";
import {
  LAUNCH_DIRECTION_DEFINITIONS,
  launchDirectionSignLabels,
  type LaunchDirectionConvention,
} from "../model/launchDirection";
import { FIELD_GUIDANCE, formatDistanceM } from "../model/units";
import { meteorologicalWind } from "../model/wind";

const SPEED_UNITS: Record<string, number> = { mph: 1.0, "m/s": 2.236936292054402 };

const DIRECTION_CONVENTIONS: Array<{
  value: LaunchDirectionConvention;
  label: string;
}> = [
  { value: "app_native", label: "App Native (+ Right)" },
  {
    value: "trackman_comparable",
    label: "TrackMan-Comparable (+ Right)",
  },
];

const RESULT_ROWS: Array<{
  key: keyof WindComparisonTs["deltas"];
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
  key: "launchAngleDeg" | "launchDirectionDeg" | "spinRpm" | "spinAxisTiltDeg";
  label: string;
  unit: string;
  guidance: string;
}

const FIELDS: FieldSpec[] = [
  { key: "launchAngleDeg", label: "Launch Angle", unit: "deg", guidance: "fxLaunchAngle" },
  { key: "launchDirectionDeg", label: "Launch Direction", unit: "deg", guidance: "fxLaunchDirection" },
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
  const [directionConvention, setDirectionConvention] =
    useState<LaunchDirectionConvention>("app_native");
  const [fields, setFields] = useState({
    launchAngleDeg: 10.9,
    launchDirectionDeg: 0.0,
    spinRpm: 2686.0,
    spinAxisTiltDeg: 0.0,
  });
  const [windEnabled, setWindEnabled] = useState(false);
  const [windSpeedMph, setWindSpeedMph] = useState(10.0);
  const [windFromDeg, setWindFromDeg] = useState(0.0);
  const [result, setResult] = useState<FlightExplorationTs | null>(null);
  const [windComparison, setWindComparison] = useState<WindComparisonTs | null>(null);
  const [error, setError] = useState<string | null>(null);
  const directionSigns = launchDirectionSignLabels(directionConvention);

  const run = () => {
    try {
      const launch = directLaunch({
        ballSpeedMph: speed * (SPEED_UNITS[speedUnit] / SPEED_UNITS.mph),
        launchDirectionConvention: directionConvention,
        ...fields,
      });
      const comparison = windEnabled
        ? compareWind(launch, meteorologicalWind(windSpeedMph / SPEED_UNITS["m/s"], windFromDeg))
        : null;
      const exploration = comparison?.wind ?? exploreFlight(launch);
      setResult(exploration);
      setWindComparison(comparison);
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
              <DecimalInput
                value={speed}
                aria-label="Ball Speed"
                title={FIELD_GUIDANCE.fxBallSpeed}
                min={0.1}
                onCommit={setSpeed}
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
          <label className="mb-2 block text-sm text-slate-300">
            <span className="mb-1 block">Direction Convention</span>
            <select
              aria-label="Launch Direction Convention"
              value={directionConvention}
              onChange={(event) =>
                setDirectionConvention(event.target.value as LaunchDirectionConvention)
              }
              className="w-full rounded border border-slate-700 bg-slate-800 px-2 py-1.5 text-slate-100"
            >
              {DIRECTION_CONVENTIONS.map(({ value, label }) => (
                <option key={value} value={value}>{label}</option>
              ))}
            </select>
            <span className="mt-1 block text-xs text-slate-500" data-testid="direction-sign-example">
              0° = straight · + = {directionSigns.positive} · − = {directionSigns.negative} · {LAUNCH_DIRECTION_DEFINITIONS[directionConvention].quantityStatus}
            </span>
          </label>
          {FIELDS.map(({ key, label, unit, guidance }) => (
            <label key={key} className="mb-2 block text-sm" title={FIELD_GUIDANCE[guidance]}>
              <span className="mb-1 flex justify-between text-slate-300">
                <span className="flex items-center truncate" title={label}>
                  {label}<FieldInfo label={label} guidance={FIELD_GUIDANCE[guidance]} />
                </span>
                <span className="text-slate-500">{unit}</span>
              </span>
              <DecimalInput
                value={fields[key]}
                aria-label={label}
                title={FIELD_GUIDANCE[guidance]}
                min={key === "spinRpm" ? 0 : undefined}
                onCommit={(value) => setFields((f) => ({ ...f, [key]: value }))}
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
            Wind Comparison
          </h2>
          <label
            className="mb-3 flex items-center gap-2 text-sm text-slate-300"
            title="Run identical launch conditions with no wind and with the selected steady wind. Source: canonical wind-scenario/v1 paired-comparison contract."
          >
            <input
              type="checkbox"
              checked={windEnabled}
              onChange={(event) => setWindEnabled(event.target.checked)}
            />
            Compare No Wind and Selected Wind
          </label>
          <label className="mb-2 block text-sm" title="Horizontal wind speed in miles per hour. Source: canonical wind-scenario/v1 meteorological adapter.">
            <span className="mb-1 flex justify-between text-slate-300">
              <span>Wind Speed</span><span className="text-slate-500">mph</span>
            </span>
            <DecimalInput
              value={windSpeedMph}
              min={0}
              aria-label="Wind Speed"
              title="Horizontal wind speed in miles per hour. Source: canonical wind-scenario/v1 meteorological adapter."
              onCommit={setWindSpeedMph}
              className="no-spinner w-full rounded border border-slate-700 bg-slate-800 px-2 py-1.5 text-slate-100 focus:border-blue-500 focus:outline-none"
            />
          </label>
          <label className="mb-2 block text-sm" title="Meteorological bearing the wind comes from, clockwise from the target line. Source: canonical wind-scenario/v1 meteorological adapter.">
            <span className="mb-1 flex justify-between text-slate-300">
              <span>Wind From Bearing</span><span className="text-slate-500">deg</span>
            </span>
            <DecimalInput
              value={windFromDeg}
              aria-label="Wind From Bearing"
              title="0° is a headwind from the target, 90° comes from the player's right, and 180° is a tailwind"
              onCommit={setWindFromDeg}
              className="no-spinner w-full rounded border border-slate-700 bg-slate-800 px-2 py-1.5 text-slate-100 focus:border-blue-500 focus:outline-none"
            />
          </label>
          <div
            className="mt-3 flex items-center gap-3 rounded-lg border border-slate-800 bg-slate-950/50 px-3 py-2 text-xs text-slate-300"
            role="img"
            aria-label={`Wind ${windSpeedMph.toFixed(1)} miles per hour from ${windFromDeg.toFixed(1)} degrees`}
            title="Arrow points where the air moves; the numeric bearing states where it comes from"
          >
            <span
              aria-hidden="true"
              className="inline-block text-xl text-sky-300"
              style={{ transform: `rotate(${windFromDeg + 180}deg)` }}
            >
              ↑
            </span>
            <span>
              {windSpeedMph.toFixed(1)} mph from {windFromDeg.toFixed(1)}°;
              arrow shows the wind-to direction.
            </span>
          </div>
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
          {windComparison && (
            <div className="mt-4 border-t border-slate-800 pt-3" aria-label="Wind effect deltas">
              <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-sky-300">
                Selected Wind Minus No Wind
              </p>
              {RESULT_ROWS.map(({ key, label, unit }) => (
                <div key={`delta-${key}`} className="flex justify-between text-xs text-slate-400">
                  <span>Δ {label}</span>
                  <span className="tabular-nums text-slate-200">
                    {windComparison.deltas[key] >= 0 ? "+" : ""}
                    {windComparison.deltas[key].toFixed(2)} {unit}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      </section>

      <section className="min-w-0 space-y-3">
        <div className="rounded-xl border border-slate-800/80 bg-slate-900/60 p-4 shadow-lg shadow-black/20 backdrop-blur">
          <FlightCanvases
            points={result?.points ?? []}
            comparisonPoints={windComparison?.calm.points ?? []}
            emptyText="Enter launch conditions and press Run Flight."
          />
        </div>
      </section>
    </div>
  );
}
