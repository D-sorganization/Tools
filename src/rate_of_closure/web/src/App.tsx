/**
 * Rate of Closure Impact Explorer — shareable web version.
 *
 * Mirrors the PyQt6 tool: scenario inputs on the left (typed entries
 * with hover guidance, unit drop-downs in the UpstreamDrift style),
 * clickable results with explanations, common literature closure
 * metrics, the animated 3D clubhead, and a Derivation & Traceability
 * tab typesetting the whole calculation with live numbers. All physics
 * lives in model/impact.ts, pinned test-for-test against Python.
 */

import { useMemo, useState } from "react";

import { ClubCanvas } from "./components/ClubCanvas";
import { DecimalInput } from "./components/DecimalInput";
import { FieldInfo } from "./components/FieldInfo";
import { FlightExplorerPanel } from "./components/FlightExplorerPanel";
import { PlotsPanel } from "./components/PlotsPanel";
import { PuttingPanel } from "./components/PuttingPanel";
import { DEFAULT_TARGET, type TargetRegionTs } from "./model/targets";
import { SimulationPanel } from "./components/SimulationPanel";
import { VariationPanel } from "./components/VariationPanel";
import { ClubPanel } from "./components/ClubPanel";
import {
  generatedHeadFor,
  type GeneratedHead,
} from "./model/clubHeadGeneration";
import { Derivation } from "./components/Derivation";
import { GlossaryPanel } from "./components/GlossaryPanel";
import { getClub, type ClubSpec } from "./model/club";
import {
  METRIC_EXPLANATIONS,
  RESULT_EXPLANATIONS,
} from "./model/derivation";
import { FIELD_TO_TERM } from "./model/glossary";
import { HELP_TEXTS } from "./model/helptext";
import {
  BOUNDS,
  closureMetrics,
  DEFAULT_SCENARIO,
  solve,
  type ImpactScenario,
} from "./model/impact";
import {
  FIELD_GUIDANCE,
  fromCanonical,
  QUANTITY_UNITS,
  toCanonical,
  type Quantity,
} from "./model/units";

interface FieldSpec {
  key: keyof ImpactScenario;
  label: string;
  quantity?: Quantity;
  fixedUnit?: string;
  step: number;
}

const FIELDS: FieldSpec[] = [
  { key: "clubheadSpeedMph", label: "Clubhead Speed", quantity: "speed", step: 1 },
  {
    key: "omegaPlaneDps",
    label: "In-Plane Rotation (SPV)",
    quantity: "rotation",
    step: 50,
  },
  {
    key: "omegaShaftDps",
    label: "About-Shaft Rotation (HTV)",
    quantity: "rotation",
    step: 50,
  },
  { key: "lieAngleDeg", label: "Shaft Lie at Impact", fixedUnit: "deg", step: 1 },
  { key: "comToFaceMm", label: "GC to Face Center", quantity: "length", step: 1 },
  {
    key: "impactOffsetToeMm",
    label: "Impact Toward Toe",
    quantity: "length",
    step: 1,
  },
  {
    key: "impactOffsetHighMm",
    label: "Impact Above Center",
    quantity: "length",
    step: 1,
  },
  { key: "contactDurationUs", label: "Contact Duration", fixedUnit: "µs", step: 10 },
];

interface RowSpec {
  key: string;
  label: string;
  unit?: string;
  quantity?: Quantity;
}

const RESULT_ROWS: RowSpec[] = [
  { key: "pathDeviationDeg", label: "Impact-Point Path vs Reference", unit: "°" },
  { key: "aoaDeviationDeg", label: "Attack-Angle Change", unit: "°" },
  { key: "tangentialSpeedMph", label: "Rotation-Induced Velocity", quantity: "speed" },
  { key: "speedDeltaMph", label: "Delivered Speed Change", quantity: "speed" },
  { key: "closureRateDps", label: "Closure Rate (CCV)", quantity: "rotation" },
  { key: "normalizedClosureDegPerFt", label: "Normalized Closure", unit: " °/ft" },
  { key: "closureDuringContactDeg", label: "Face Closure During Contact", unit: "°" },
  {
    key: "loftGainDuringContactDeg",
    label: "Dynamic Loft Gained During Contact",
    unit: "°",
  },
];

const METRIC_ROWS: RowSpec[] = [
  { key: "ccvDps", label: "Club Closure Velocity (CCV)", quantity: "rotation" },
  { key: "closureDegPerFt", label: "Closure per Foot of Travel", unit: " °/ft" },
  { key: "closureDegPerInch", label: "Closure per Inch of Travel", unit: " °/in" },
  { key: "closureDegPerMs", label: "Closure per Millisecond", unit: " °/ms" },
  { key: "rIsaFt", label: "Distance to Screw Axis (R_ISA)", unit: " ft" },
  { key: "rIsaM", label: "Distance to Screw Axis (Metric)", unit: " m" },
  {
    key: "timeToSquareFrom1DegOpenMs",
    label: "Time to Square From 1° Open",
    unit: " ms",
  },
  {
    key: "toeHeelSpeedDeltaMph",
    label: "Toe vs Heel Speed Difference",
    quantity: "speed",
  },
];

const UNIT_LABELS: Record<Quantity, string> = {
  speed: "Speed",
  rotation: "Rotation",
  length: "Length",
  distance: "Distance",
};

const TABS = [
  "Explorer",
  "Calculation Description",
  "Simulation",
  "Plots",
  "Flight Explorer",
  "Variation",
  "Putting",
  "Glossary",
] as const;

export default function App() {
  const defaultDriver = useMemo(() => getClub("Driver 10.5°"), []);
  const [scenario, setScenario] = useState<ImpactScenario>(DEFAULT_SCENARIO);
  // Target region (#4125 H7b): shared by the Simulation flight view /
  // solver and the Variation landing overlay (hold-% headline).
  const [target, setTarget] = useState<TargetRegionTs>(DEFAULT_TARGET);
  const [tab, setTab] = useState<(typeof TABS)[number]>(TABS[0]);
  const [explained, setExplained] = useState<string>("pathDeviationDeg");
  const [units, setUnits] = useState<Record<Quantity, string>>({
    speed: "mph",
    rotation: "deg/s",
    length: "mm",
    // Ball-flight distances (#4125 H6): yards by default.
    distance: "yd",
  });
  const [generatedHead, setGeneratedHead] = useState<GeneratedHead>(() =>
    generatedHeadFor(defaultDriver),
  );
  const [clubSpec, setClubSpec] = useState<ClubSpec>(defaultDriver);
  const [glossaryTerm, setGlossaryTerm] = useState<string | undefined>(undefined);
  const result = useMemo(() => solve(scenario), [scenario]);
  const metrics = useMemo(() => closureMetrics(scenario), [scenario]);

  // Scenario plumbing: GC-to-face and lie follow the selected club's
  // spec (the CG sits within a few mm of the geometric center); both
  // stay fully editable afterwards, preserving user overrides.
  const driveScenarioFromClub = (comToFaceMm: number, lieAngleDeg: number) => {
    setScenario((s) => ({ ...s, comToFaceMm, lieAngleDeg }));
  };

  const update = (key: keyof ImpactScenario, quantity: Quantity | undefined, raw: string) => {
    const displayed = Number(raw);
    if (!Number.isFinite(displayed)) return;
    const canonical = quantity
      ? toCanonical(quantity, units[quantity], displayed)
      : displayed;
    const [low, high] = BOUNDS[key];
    setScenario((s) => ({
      ...s,
      [key]: Math.min(high, Math.max(low, canonical)),
    }));
  };

  const formatRow = (
    spec: RowSpec,
    value: number,
  ): string => {
    if (!Number.isFinite(value)) return "∞ (not closing)";
    if (spec.quantity) {
      const unit = units[spec.quantity];
      const displayed = fromCanonical(spec.quantity, unit, value);
      return `${displayed >= 0 ? "+" : ""}${displayed.toFixed(2)} ${unit}`;
    }
    return `${value >= 0 ? "+" : ""}${value.toFixed(2)}${spec.unit ?? ""}`;
  };

  const allRows = [...RESULT_ROWS, ...METRIC_ROWS];
  const explainedLabel = allRows.find((r) => r.key === explained)?.label;
  const explanation =
    RESULT_EXPLANATIONS[explained] ?? METRIC_EXPLANATIONS[explained];

  const rowButton = (spec: RowSpec, value: number) => {
    const active = explained === spec.key;
    return (
      <button
        key={spec.key}
        type="button"
        onClick={() => setExplained(spec.key)}
        aria-pressed={active}
        title={
          RESULT_EXPLANATIONS[spec.key] ?? METRIC_EXPLANATIONS[spec.key]
        }
        className={
          "flex items-center justify-between rounded-lg border px-3 py-2 text-left text-sm transition-all " +
          (active
            ? "border-sky-400 bg-sky-500/20 ring-1 ring-sky-400/60 shadow-[0_0_14px_rgba(56,189,248,0.25)]"
            : "border-slate-800/80 bg-slate-900/50 hover:border-slate-600 hover:bg-slate-800/50")
        }
      >
        <span className="flex items-center gap-2 text-slate-300">
          {spec.label}
          <span aria-hidden="true" className="text-[10px] font-semibold uppercase tracking-wide text-sky-400">
            Details ›
          </span>
        </span>
        <span
          className={
            spec.key === "pathDeviationDeg"
              ? "font-semibold tabular-nums text-amber-300"
              : "font-semibold tabular-nums text-slate-100"
          }
        >
          {formatRow(spec, value)}
        </span>
      </button>
    );
  };

  return (
    <div className="mx-auto min-h-screen max-w-7xl p-5 text-slate-100 sm:p-8">
      <header className="mb-4">
        <h1 className="bg-gradient-to-r from-sky-300 via-teal-200 to-emerald-300 bg-clip-text text-3xl font-bold tracking-tight text-transparent">
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
            title={`Switch to the ${name} view`}
            className={
              "rounded-full border px-4 py-1.5 text-sm font-medium transition-all " +
              (tab === name
                ? "border-sky-400/60 bg-sky-500/10 text-sky-300 shadow-[0_0_18px_rgba(56,189,248,0.25)]"
                : "border-slate-700/80 bg-slate-900/60 text-slate-300 hover:border-slate-500 hover:text-slate-100")
            }
          >
            {name}
          </button>
        ))}
      </nav>

      <details
        className="mb-5 rounded-xl border border-slate-800/80 bg-slate-900/60 px-5 py-3 text-sm shadow-lg shadow-black/20 backdrop-blur"
        title="Usage instructions for this page"
      >
        <summary className="cursor-pointer font-semibold text-slate-300 hover:text-slate-100">
          {HELP_TEXTS[tab].title}
        </summary>
        {HELP_TEXTS[tab].paragraphs.map((paragraph, index) => (
          <p key={index} className="mt-2 max-w-3xl text-slate-400">
            {paragraph}
          </p>
        ))}
      </details>

      {tab === TABS[7] ? (
        <GlossaryPanel key={glossaryTerm ?? "none"} initialTerm={glossaryTerm} />
      ) : tab === TABS[6] ? (
        <PuttingPanel
          distanceUnit={units.distance}
          onGlossary={(term) => {
            setGlossaryTerm(term);
            setTab(TABS[7]);
          }}
        />
      ) : tab === TABS[5] ? (
        <VariationPanel target={target} distanceUnit={units.distance} />
      ) : tab === TABS[4] ? (
        <FlightExplorerPanel distanceUnit={units.distance} />
      ) : tab === TABS[3] ? (
        // Static loft mirrors the desktop default driver (same note as
        // the Simulation tab; the full club picker joins with P7 WASM).
        <PlotsPanel scenario={scenario} loftDeg={10.5} />
      ) : tab === TABS[2] ? (
        // Static loft mirrors the desktop default driver; the full club
        // picker joins the web simulation with the P7 WASM port.
        <SimulationPanel
          scenario={scenario}
          loftDeg={10.5}
          clubSpec={clubSpec}
          onScenarioChange={(updates) =>
            setScenario((s) => ({ ...s, ...updates }))
          }
          target={target}
          onTargetChange={setTarget}
          distanceUnit={units.distance}
        />
      ) : tab === TABS[1] ? (
        <Derivation scenario={scenario} />
      ) : (
        <div className="grid gap-6 lg:grid-cols-[340px_1fr]">
          <section
            aria-label="Scenario inputs"
            className="space-y-4"
          >
            <div className="rounded-xl border border-slate-800/80 bg-slate-900/60 p-5 shadow-lg shadow-black/20 backdrop-blur">
              <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">
                Units
              </h2>
              {(Object.keys(QUANTITY_UNITS) as Quantity[]).map((quantity) => (
                <label
                  key={quantity}
                  className="mb-2 flex items-center justify-between text-sm"
                >
                  <span className="text-slate-300">{UNIT_LABELS[quantity]}</span>
                  <select
                    value={units[quantity]}
                    title={`Display unit for ${UNIT_LABELS[quantity].toLowerCase()} values`}
                    onChange={(e) =>
                      setUnits((u) => ({ ...u, [quantity]: e.target.value }))
                    }
                    className="rounded border border-slate-700 bg-slate-800 px-2 py-1 text-slate-100 focus:border-blue-500 focus:outline-none"
                  >
                    {Object.keys(QUANTITY_UNITS[quantity]).map((unit) => (
                      <option key={unit} value={unit}>
                        {unit}
                      </option>
                    ))}
                  </select>
                </label>
              ))}
            </div>

            <ClubPanel
              onDriveScenario={driveScenarioFromClub}
              onGenerate={setGeneratedHead}
              onSpecChange={setClubSpec}
            />

            <div className="rounded-xl border border-slate-800/80 bg-slate-900/60 p-5 shadow-lg shadow-black/20 backdrop-blur">
              <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">
                Scenario
              </h2>
              {FIELDS.map(({ key, label, quantity, fixedUnit, step }) => {
                const unit = quantity ? units[quantity] : fixedUnit ?? "";
                const displayed = quantity
                  ? fromCanonical(quantity, units[quantity], scenario[key])
                  : scenario[key];
                return (
                  <label
                    key={key}
                    title={FIELD_GUIDANCE[key]}
                    className="mb-3 block text-sm"
                  >
                    <span className="mb-1 flex justify-between text-slate-300">
                      <span className="flex items-center">
                        {label}
                        <FieldInfo label={label} guidance={FIELD_GUIDANCE[key]} />
                      </span>
                      <span className="text-slate-500">{unit}</span>
                    </span>
                    <DecimalInput
                      step={step}
                      value={Number(displayed.toFixed(4))}
                      aria-label={`${label} ${unit}`.trim()}
                      onCommit={(value) => update(key, quantity, String(value))}
                      title={FIELD_GUIDANCE[key]}
                      className="no-spinner w-full rounded border border-slate-700 bg-slate-800 px-2 py-1.5 text-slate-100 focus:border-blue-500 focus:outline-none"
                    />
                  </label>
                );
              })}
            </div>
          </section>

          <section className="space-y-6">
            <div
              aria-label="Results"
              className="rounded-xl border border-slate-800/80 bg-slate-900/60 p-5 shadow-lg shadow-black/20 backdrop-blur"
            >
              <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">
                Impact-Point Deviation — Click a Value for Its Explanation
              </h2>
              <div className="grid gap-2 sm:grid-cols-2">
                {RESULT_ROWS.map((spec) =>
                  rowButton(
                    spec,
                    result[spec.key as keyof typeof result] as number,
                  ),
                )}
              </div>
              <h2 className="mb-3 mt-5 text-sm font-semibold uppercase tracking-wide text-slate-400">
                Common Closure Metrics
              </h2>
              <div className="grid gap-2 sm:grid-cols-2">
                {METRIC_ROWS.map((spec) =>
                  rowButton(
                    spec,
                    metrics[spec.key as keyof typeof metrics],
                  ),
                )}
              </div>
              {explainedLabel && (
                <div
                  aria-live="polite"
                  className="mt-3 rounded-md border border-slate-800 bg-slate-950/60 p-3 text-xs text-slate-400"
                >
                  <h3 className="mb-1 text-sm font-bold text-sky-200">
                    {explainedLabel}
                  </h3>
                  {explanation}
                  <button
                    type="button"
                    onClick={() => {
                      setGlossaryTerm(FIELD_TO_TERM[explained]);
                      setTab("Glossary");
                    }}
                    title="Open the glossary, pre-selecting the matching term"
                    className="mt-2 block text-sky-400 underline-offset-2 hover:underline"
                  >
                    Glossary →
                  </button>
                </div>
              )}
              <p className="mt-3 text-xs text-slate-500">
                Sign convention follows standard launch-monitor definitions:
                club path positive = in-to-out (right of target); negative
                path deviation = the impact point travels left of the
                reported geometric-center path. Defaults are dossier-sourced
                (Cheetham 2014 tour HTV 1,307 ± 304 °/s about the shaft;
                CCV ≈ 2,100 °/s; 40 mm GC-to-face offset) — hover any input
                for its suggested range and source, and enter your own
                measured values.
              </p>
            </div>

            <ClubCanvas
              scenario={scenario}
              externalMesh={generatedHead?.mesh ?? null}
              hoselPoint={generatedHead?.hosel ?? null}
              cogPoint={generatedHead?.cog ?? null}
            />
          </section>
        </div>
      )}
      <footer className="mt-10 border-t border-slate-800/60 pt-4 text-xs text-slate-500">
        Companion tool to the{" "}
        <a
          href="https://www.affinedrift.com"
          target="_blank"
          rel="noreferrer"
          className="text-sky-400 underline-offset-2 hover:underline"
        >
          AffineDrift
        </a>{" "}
        launch-monitor research. Physics parity-tested against the canonical
        Python implementation; rate data from openly published sources
        (Cheetham 2014; published launch-monitor material). MIT licensed.
      </footer>
    </div>
  );
}
