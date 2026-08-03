/**
 * Rate of Closure Impact Explorer — shareable web version.
 *
 * Mirrors the PyQt6 tool: scenario inputs on the left, results and the
 * animated 3D clubhead on the right. All physics lives in model/impact.ts,
 * which is pinned test-for-test against the Python implementation.
 */

import { useMemo, useState } from "react";

import { ClubCanvas } from "./components/ClubCanvas";
import {
  BOUNDS,
  DEFAULT_SCENARIO,
  solve,
  type ImpactScenario,
} from "./model/impact";

interface FieldSpec {
  key: keyof ImpactScenario;
  label: string;
  unit: string;
  step: number;
}

const FIELDS: FieldSpec[] = [
  { key: "clubheadSpeedMph", label: "Clubhead speed", unit: "mph", step: 1 },
  {
    key: "omegaPlaneDps",
    label: "In-plane rotation",
    unit: "deg/s",
    step: 50,
  },
  {
    key: "omegaShaftDps",
    label: "About-shaft rotation",
    unit: "deg/s",
    step: 50,
  },
  { key: "lieAngleDeg", label: "Shaft lie at impact", unit: "deg", step: 1 },
  {
    key: "comToFaceMm",
    label: "Reference to face center",
    unit: "mm",
    step: 1,
  },
  { key: "impactOffsetToeMm", label: "Impact toward toe", unit: "mm", step: 1 },
  {
    key: "impactOffsetHighMm",
    label: "Impact above center",
    unit: "mm",
    step: 1,
  },
  { key: "contactDurationUs", label: "Contact duration", unit: "µs", step: 10 },
];

const RESULT_ROWS: { key: string; label: string; unit: string }[] = [
  {
    key: "pathDeviationDeg",
    label: "Impact-point path vs reference",
    unit: "°",
  },
  { key: "aoaDeviationDeg", label: "Attack-angle change", unit: "°" },
  {
    key: "tangentialSpeedMph",
    label: "Rotation-induced velocity",
    unit: " mph",
  },
  { key: "speedDeltaMph", label: "Delivered speed change", unit: " mph" },
  { key: "closureRateDps", label: "Closure rate (CCV)", unit: " °/s" },
  {
    key: "normalizedClosureDegPerFt",
    label: "Normalized closure",
    unit: " °/ft",
  },
  {
    key: "closureDuringContactDeg",
    label: "Face closure during contact",
    unit: "°",
  },
  {
    key: "loftGainDuringContactDeg",
    label: "Dynamic loft gained in contact",
    unit: "°",
  },
];

export default function App() {
  const [scenario, setScenario] = useState<ImpactScenario>(DEFAULT_SCENARIO);
  const result = useMemo(() => solve(scenario), [scenario]);

  const update = (key: keyof ImpactScenario, raw: string) => {
    const value = Number(raw);
    if (!Number.isFinite(value)) return;
    const [low, high] = BOUNDS[key];
    setScenario((s) => ({
      ...s,
      [key]: Math.min(high, Math.max(low, value)),
    }));
  };

  return (
    <div className="min-h-screen bg-slate-950 p-6 text-slate-100">
      <header className="mb-6">
        <h1 className="text-2xl font-semibold">
          Rate of Closure Impact Explorer
        </h1>
        <p className="mt-1 max-w-3xl text-sm text-slate-400">
          A rotating clubhead is a rigid body: the velocity of the impact
          point is v(P) = v(ref) + ω × r. Launch monitors track the reference
          point; the ball only feels the impact point. This explorer shows
          how far apart those two deliveries are.
        </p>
      </header>

      <div className="grid gap-6 lg:grid-cols-[340px_1fr]">
        <section
          aria-label="Scenario inputs"
          className="rounded-lg border border-slate-800 bg-slate-900 p-4"
        >
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">
            Scenario
          </h2>
          {FIELDS.map(({ key, label, unit, step }) => (
            <label key={key} className="mb-3 block text-sm">
              <span className="mb-1 flex justify-between text-slate-300">
                <span>{label}</span>
                <span className="text-slate-500">{unit}</span>
              </span>
              <input
                type="number"
                step={step}
                value={scenario[key]}
                min={BOUNDS[key][0]}
                max={BOUNDS[key][1]}
                onChange={(e) => update(key, e.target.value)}
                className="w-full rounded border border-slate-700 bg-slate-800 px-2 py-1.5 text-slate-100 focus:border-blue-500 focus:outline-none"
              />
            </label>
          ))}
        </section>

        <section className="space-y-6">
          <div
            aria-label="Results"
            className="rounded-lg border border-slate-800 bg-slate-900 p-4"
          >
            <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">
              Impact-Point Deviation
            </h2>
            <dl className="grid gap-x-8 gap-y-2 sm:grid-cols-2">
              {RESULT_ROWS.map(({ key, label, unit }) => {
                const value = result[key as keyof typeof result] as number;
                return (
                  <div key={key} className="flex justify-between text-sm">
                    <dt className="text-slate-400">{label}</dt>
                    <dd
                      className={
                        key === "pathDeviationDeg"
                          ? "font-semibold text-amber-300"
                          : "text-slate-100"
                      }
                    >
                      {value >= 0 ? "+" : ""}
                      {value.toFixed(2)}
                      {unit}
                    </dd>
                  </div>
                );
              })}
            </dl>
            <p className="mt-3 text-xs text-slate-500">
              Sign convention follows TrackMan: club path positive =
              in-to-out (right of target); negative path deviation = the
              impact point travels left of the reported geometric-center
              path. Defaults are dossier-sourced (Cheetham 2014 tour HTV
              1,307 ± 304 °/s about the shaft; CCV ≈ 2,100 °/s; 40 mm
              GC-to-face offset) — enter your own measured values.
            </p>
          </div>

          <ClubCanvas scenario={scenario} />
        </section>
      </div>
    </div>
  );
}
