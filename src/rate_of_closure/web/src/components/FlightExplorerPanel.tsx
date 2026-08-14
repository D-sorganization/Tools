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

import { useRef, useState } from "react";

import { DecimalInput } from "./DecimalInput";
import { FieldInfo } from "./FieldInfo";
import { FlightCanvases } from "./FlightCanvases";
import { FlightPlayback3D } from "./FlightPlayback3D";
import {
  buildAcceptedFlightStudy, type AcceptedFlightStudy,
} from "./flightAcceptedStudy";
import { SpatialTargetSection } from "./SpatialTargetSection";
import {
  compareWind,
  directLaunch,
  exploreFlight,
} from "../model/flightExplorer";
import {
  LAUNCH_DIRECTION_DEFINITIONS,
  launchDirectionSignLabels,
  type LaunchDirectionConvention,
} from "../model/launchDirection";
import { FIELD_GUIDANCE, formatDistanceM } from "../model/units";
import { meteorologicalWind } from "../model/wind";
import { scheduleMeaningfulVisualReveal } from "../model/variationVisualProminence";
import type { FlightSampleSelection } from "../model/flightSampleInspector";
import type { SpatialTargetTs } from "../model/spatialTarget";
import {
  boundedFlightError, DIRECTION_CONVENTIONS, FLIGHT_FIELDS, RESULT_ROWS,
  SPEED_UNITS,
} from "./flightExplorerContract";

interface Props {
  /** Ball-flight distance display unit (#4125 H6): yards default. */
  distanceUnit?: string;
  spatialTarget: SpatialTargetTs;
  onSpatialTargetChange: (target: SpatialTargetTs) => void;
  executeFlight?: typeof exploreFlight;
}

export function FlightExplorerPanel({
  distanceUnit = "yd",
  spatialTarget,
  onSpatialTargetChange,
  executeFlight = exploreFlight,
}: Props) {
  const [speedMph, setSpeedMph] = useState(167.0);
  const [speedUnit, setSpeedUnit] = useState("mph");
  const displayedSpeed = speedMph / SPEED_UNITS[speedUnit];
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
  const [accepted, setAccepted] = useState<AcceptedFlightStudy | null>(null);
  const [selection, setSelection] = useState<(
    FlightSampleSelection & { generation: number; commandId: number }
  ) | null>(null);
  const [error, setError] = useState<string | null>(null);
  const generation = useRef(0);
  const selectionCommand = useRef(0);
  const visualCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const result = accepted?.exploration ?? null;
  const windComparison = accepted?.comparison ?? null;
  const currentWind = windEnabled
    ? meteorologicalWind(windSpeedMph / SPEED_UNITS["m/s"], windFromDeg) : null;
  const inputsChanged = accepted !== null && (
    accepted.context.ballSpeedMph !== speedMph ||
    accepted.context.launchAngleDeg !== fields.launchAngleDeg ||
    accepted.context.launchDirectionDeg !== fields.launchDirectionDeg ||
    accepted.context.spinRpm !== fields.spinRpm ||
    accepted.context.spinAxisTiltDeg !== fields.spinAxisTiltDeg ||
    accepted.context.directionConvention !== directionConvention ||
    JSON.stringify(accepted.context.windScenario) !== JSON.stringify(currentWind)
  );
  const directionSigns = launchDirectionSignLabels(directionConvention);

  const run = (allowAutomaticReveal = false) => {
    try {
      const launch = directLaunch({
        ballSpeedMph: speedMph,
        launchDirectionConvention: directionConvention,
        ...fields,
      });
      const windScenario = windEnabled
        ? meteorologicalWind(windSpeedMph / SPEED_UNITS["m/s"], windFromDeg) : null;
      const comparison = windScenario
        ? compareWind(launch, windScenario, executeFlight) : null;
      const exploration = comparison?.wind ?? executeFlight(launch);
      const nextGeneration = generation.current + 1;
      const candidate = buildAcceptedFlightStudy(nextGeneration, {
        entryMode: "direct",
        ballSpeedMph: speedMph,
        launchAngleDeg: fields.launchAngleDeg,
        launchDirectionDeg: fields.launchDirectionDeg,
        spinRpm: fields.spinRpm,
        spinAxisTiltDeg: fields.spinAxisTiltDeg,
        directionConvention,
        windScenario,
        model: "waterloo_penner",
        kernelRevision: "web-rk4-10ms-sampled-v1",
      }, exploration, comparison);
      generation.current = nextGeneration;
      setAccepted(candidate);
      setSelection(null);
      setError(null);
      if (allowAutomaticReveal) {
        scheduleMeaningfulVisualReveal(() => visualCanvasRef.current);
      }
    } catch (exc) {
      setError(boundedFlightError(exc, accepted !== null));
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
                value={displayedSpeed}
                aria-label="Ball Speed"
                title={FIELD_GUIDANCE.fxBallSpeed}
                min={1 / SPEED_UNITS[speedUnit]}
                max={250 / SPEED_UNITS[speedUnit]}
                onCommit={(value) => setSpeedMph(value * SPEED_UNITS[speedUnit])}
                className="no-spinner w-full min-w-16 rounded border border-slate-700 bg-slate-800 px-2 py-1.5 text-slate-100 focus:border-blue-500 focus:outline-none"
              />
              <select
                value={speedUnit}
                title={FIELD_GUIDANCE.fxSpeedUnit}
                onChange={(event) => setSpeedUnit(event.target.value)}
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
              {DIRECTION_CONVENTIONS.map(({ value, label, disabled, title }) => (
                <option key={label} value={value} disabled={disabled} title={title}>
                  {label}
                </option>
              ))}
            </select>
            <span className="mt-1 block text-xs text-slate-500" data-testid="direction-sign-example">
              0° = straight · + = {directionSigns.positive} · − = {directionSigns.negative} · {LAUNCH_DIRECTION_DEFINITIONS[directionConvention].quantityStatus}
            </span>
          </label>
          {FLIGHT_FIELDS.map(({ key, label, unit, guidance }) => (
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
                min={key === "spinRpm" ? 0 : key === "launchAngleDeg" ? -89 :
                  key === "launchDirectionDeg" ? -45 : -60}
                max={key === "spinRpm" ? 15000 : key === "launchAngleDeg" ? 89 :
                  key === "launchDirectionDeg" ? 45 : 60}
                onCommit={(value) => setFields((f) => ({ ...f, [key]: value }))}
                className="no-spinner w-full min-w-16 rounded border border-slate-700 bg-slate-800 px-2 py-1.5 text-slate-100 focus:border-blue-500 focus:outline-none"
              />
            </label>
          ))}
          <button
            type="button"
            onClick={(event) => run(event.detail > 0)}
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
          {accepted && (
            <p className="mt-2 text-xs text-slate-400" role="status"
              aria-label="Displayed flight context">
              {inputsChanged ? "Prior result — inputs changed: " : "Displayed flight: "}
              {accepted.contextLabel}
            </p>
          )}
          <p className="mt-3 text-xs text-slate-500">
            Waterloo/Penner flight physics, parity-banded against the
            Python explorer (which adds the full 7-model literature picker
            and an impact-delivery entry mode; both arrive here with the
            P7 WASM kernels).
          </p>
        </div>

        <SpatialTargetSection target={spatialTarget} onChange={onSpatialTargetChange}
          flightPoints={result?.points ?? []} />

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
              max={150}
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

      <section aria-label="Flight accepted visual inspector"
        className="order-first min-w-0 space-y-3 lg:order-none">
        <div className="rounded-xl border border-slate-800/80 bg-slate-900/60 p-4 shadow-lg shadow-black/20 backdrop-blur">
          <FlightCanvases
            points={result?.points ?? []}
            comparisonPoints={windComparison?.calm.points ?? []}
            emptyText="Enter launch conditions and press Run Flight."
            spatialTarget={spatialTarget}
            plan={accepted?.plan}
            selection={selection?.generation === accepted?.generation ? selection : null}
            onSelectionChange={(next) => setSelection(
              next && accepted ? {
                ...next, generation: accepted.generation,
                commandId: ++selectionCommand.current,
              } : null,
            )}
            prominenceRef={visualCanvasRef}
          />
          <p className="mt-2 min-h-5 text-xs text-slate-300" role="status"
            aria-label="Selected flight sample" aria-live="polite">
            {selection && accepted && selection.generation === accepted.generation
              ? (() => {
                const sample = accepted.plan.rawSample(selection.rawIndex);
                return `Current primary flight, source sample ${sample.rawIndex + 1}/${accepted.plan.rawCount}; ` +
                  `t ${sample.timeS.toFixed(3)} s; downrange ${sample.downrangeM.toFixed(3)} m; ` +
                  `height ${sample.heightM.toFixed(3)} m; right ${sample.rightM.toFixed(3)} m; ${sample.phase}.`;
              })()
              : "Select the current primary trajectory; the calm dashed ghost is comparison-only."}
          </p>
          <div className="mt-4 border-t border-slate-800 pt-4">
            <FlightPlayback3D
              points={result?.points ?? []}
              comparisonPoints={windComparison?.calm.points ?? []}
              spatialTarget={spatialTarget}
              selectedTimeS={selection && accepted && selection.generation === accepted.generation
                ? accepted.plan.rawSample(selection.rawIndex).timeS : null}
              selectedCommandId={selection?.commandId}
            />
          </div>
        </div>
      </section>
    </div>
  );
}
