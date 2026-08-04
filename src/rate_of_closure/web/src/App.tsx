/**
 * Rate of Closure Impact Explorer — shareable web version.
 *
 * Mirrors the PyQt6 tool: scenario inputs on the left, clickable results
 * with explanations and the animated 3D clubhead on the right, and a
 * Derivation & Traceability tab typesetting the whole calculation with
 * live numbers. All physics lives in model/impact.ts, which is pinned
 * test-for-test against the Python implementation.
 */

import { useMemo, useState } from "react";

import { ClubCanvas } from "./components/ClubCanvas";
import { Derivation } from "./components/Derivation";
import { RESULT_EXPLANATIONS } from "./model/derivation";
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
  { key: "clubheadSpeedMph", label: "Clubhead Speed", unit: "mph", step: 1 },
  {
    key: "omegaPlaneDps",
    label: "In-Plane Rotation (SPV)",
    unit: "deg/s",
    step: 50,
  },
  {
    key: "omegaShaftDps",
    label: "About-Shaft Rotation (HTV)",
    unit: "deg/s",
    step: 50,
  },
  { key: "lieAngleDeg", label: "Shaft Lie at Impact", unit: "deg", step: 1 },
  { key: "comToFaceMm", label: "GC to Face Center", unit: "mm", step: 1 },
  { key: "impactOffsetToeMm", label: "Impact Toward Toe", unit: "mm", step: 1 },
  {
    key: "impactOffsetHighMm",
    label: "Impact Above Center",
    unit: "mm",
    step: 1,
  },
  { key: "contactDurationUs", label: "Contact Duration", unit: "µs", step: 10 },
];

const RESULT_ROWS: { key: string; label: string; unit: string }[] = [
  {
    key: "pathDeviationDeg",
    label: "Impact-Point Path vs Reference",
    unit: "°",
  },
  { key: "aoaDeviationDeg", label: "Attack-Angle Change", unit: "°" },
  {
    key: "tangentialSpeedMph",
    label: "Rotation-Induced Velocity",
    unit: " mph",
  },
  { key: "speedDeltaMph", label: "Delivered Speed Change", unit: " mph" },
  { key: "closureRateDps", label: "Closure Rate (CCV)", unit: " °/s" },
  {
    key: "normalizedClosureDegPerFt",
    label: "Normalized Closure",
    unit: " °/ft",
  },
  {
    key: "closureDuringContactDeg",
    label: "Face Closure During Contact",
    unit: "°",
  },
  {
    key: "loftGainDuringContactDeg",
    label: "Dynamic Loft Gained During Contact",
    unit: "°",
  },
];

const TABS = ["Explorer", "Derivation & Traceability"] as const;

export default function App() {
  const [scenario, setScenario] = useState<ImpactScenario>(DEFAULT_SCENARIO);
  const [tab, setTab] = useState<(typeof TABS)[number]>(TABS[0]);
  const [explained, setExplained] = useState<string>("pathDeviationDeg");
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

  const explainedLabel = RESULT_ROWS.find((r) => r.key === explained)?.label;

  return (
    <div className="min-h-screen bg-slate-950 p-6 text-slate-100">
      <header className="mb-4">
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

      <nav aria-label="Views" className="mb-5 flex gap-2">
        {TABS.map((name) => (
          <button
            key={name}
            type="button"
            onClick={() => setTab(name)}
            aria-current={tab === name}
            className={
              "rounded-md border px-3 py-1.5 text-sm transition-colors " +
              (tab === name
                ? "border-blue-500 bg-blue-500/10 text-blue-300"
                : "border-slate-700 bg-slate-900 text-slate-300 hover:border-slate-500")
            }
          >
            {name}
          </button>
        ))}
      </nav>

      {tab === TABS[1] ? (
        <Derivation scenario={scenario} />
      ) : (
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
                Impact-Point Deviation — Click a Value for Its Explanation
              </h2>
              <div className="grid gap-2 sm:grid-cols-2">
                {RESULT_ROWS.map(({ key, label, unit }) => {
                  const value = result[key as keyof typeof result] as number;
                  const active = explained === key;
                  return (
                    <button
                      key={key}
                      type="button"
                      onClick={() => setExplained(key)}
                      aria-pressed={active}
                      className={
                        "flex items-center justify-between rounded-md border px-3 py-2 text-left text-sm transition-colors " +
                        (active
                          ? "border-blue-500 bg-blue-500/10"
                          : "border-slate-800 bg-slate-900 hover:border-slate-600")
                      }
                    >
                      <span className="text-slate-400">{label}</span>
                      <span
                        className={
                          key === "pathDeviationDeg"
                            ? "font-semibold text-amber-300"
                            : "font-semibold text-slate-100"
                        }
                      >
                        {value >= 0 ? "+" : ""}
                        {value.toFixed(2)}
                        {unit}
                      </span>
                    </button>
                  );
                })}
              </div>
              {explainedLabel && (
                <div
                  aria-live="polite"
                  className="mt-3 rounded-md border border-slate-800 bg-slate-950/60 p-3 text-xs text-slate-400"
                >
                  <span className="font-semibold text-slate-200">
                    {explainedLabel}.{" "}
                  </span>
                  {RESULT_EXPLANATIONS[explained]}
                </div>
              )}
              <p className="mt-3 text-xs text-slate-500">
                Sign convention follows standard launch-monitor definitions:
                club path positive = in-to-out (right of target); negative
                path deviation = the impact point travels left of the
                reported geometric-center path. Defaults are dossier-sourced
                (Cheetham 2014 tour HTV 1,307 ± 304 °/s about the shaft;
                CCV ≈ 2,100 °/s; 40 mm GC-to-face offset) — enter your own
                measured values.
              </p>
            </div>

            <ClubCanvas scenario={scenario} />
          </section>
        </div>
      )}
    </div>
  );
}
