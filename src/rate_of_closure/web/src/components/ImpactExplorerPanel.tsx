import { useMemo, type Dispatch, type SetStateAction } from "react";

import type { ClubSpec } from "../model/club";
import type { GeneratedHead } from "../model/clubHeadGeneration";
import { METRIC_EXPLANATIONS, RESULT_EXPLANATIONS } from "../model/derivation";
import { FIELD_TO_TERM } from "../model/glossary";
import {
  BOUNDS,
  closureMetrics,
  solve,
  type ImpactScenario,
} from "../model/impact";
import {
  FIELD_GUIDANCE,
  fromCanonical,
  QUANTITY_UNITS,
  toCanonical,
  type Quantity,
} from "../model/units";
import { ClubCanvas } from "./ClubCanvas";
import type { ClubCamera } from "../model/clubCamera";
import {
  generatedMeshSource,
  type ClubMeshSource,
} from "../model/clubMeshSource";
import { ClubPanel } from "./ClubPanel";
import { DecimalInput } from "./DecimalInput";
import { FieldInfo } from "./FieldInfo";

export type UnitSelections = Record<Quantity, string>;

interface FieldSpec {
  readonly key: keyof ImpactScenario;
  readonly label: string;
  readonly quantity?: Quantity;
  readonly fixedUnit?: string;
  readonly step: number;
}

interface RowSpec {
  readonly key: string;
  readonly label: string;
  readonly unit?: string;
  readonly quantity?: Quantity;
}

const FIELDS: readonly FieldSpec[] = [
  { key: "clubheadSpeedMph", label: "Clubhead Speed", quantity: "speed", step: 1 },
  { key: "omegaPlaneDps", label: "In-Plane Rotation (SPV)", quantity: "rotation", step: 50 },
  { key: "omegaShaftDps", label: "About-Shaft Rotation (HTV)", quantity: "rotation", step: 50 },
  { key: "lieAngleDeg", label: "Shaft Lie at Impact", fixedUnit: "deg", step: 1 },
  { key: "comToFaceMm", label: "GC to Face Center", quantity: "length", step: 1 },
  { key: "impactOffsetToeMm", label: "Impact Toward Toe", quantity: "length", step: 1 },
  { key: "impactOffsetHighMm", label: "Impact Above Center", quantity: "length", step: 1 },
  { key: "contactDurationUs", label: "Contact Duration", fixedUnit: "µs", step: 10 },
];

const RESULT_ROWS: readonly RowSpec[] = [
  { key: "pathDeviationDeg", label: "Impact-Point Path vs Reference", unit: "°" },
  { key: "aoaDeviationDeg", label: "Attack-Angle Change", unit: "°" },
  { key: "tangentialSpeedMph", label: "Rotation-Induced Velocity", quantity: "speed" },
  { key: "speedDeltaMph", label: "Delivered Speed Change", quantity: "speed" },
  { key: "closureRateDps", label: "Closure Rate (CCV)", quantity: "rotation" },
  { key: "normalizedClosureDegPerFt", label: "Normalized Closure", unit: " °/ft" },
  { key: "closureDuringContactDeg", label: "Face Closure During Contact", unit: "°" },
  { key: "loftGainDuringContactDeg", label: "Dynamic Loft Gained During Contact", unit: "°" },
];

const METRIC_ROWS: readonly RowSpec[] = [
  { key: "ccvDps", label: "Club Closure Velocity (CCV)", quantity: "rotation" },
  { key: "closureDegPerFt", label: "Closure per Foot of Travel", unit: " °/ft" },
  { key: "closureDegPerInch", label: "Closure per Inch of Travel", unit: " °/in" },
  { key: "closureDegPerMs", label: "Closure per Millisecond", unit: " °/ms" },
  { key: "rIsaFt", label: "Distance to Screw Axis (R_ISA)", unit: " ft" },
  { key: "rIsaM", label: "Distance to Screw Axis (Metric)", unit: " m" },
  { key: "timeToSquareFrom1DegOpenMs", label: "Time to Square From 1° Open", unit: " ms" },
  { key: "toeHeelSpeedDeltaMph", label: "Toe vs Heel Speed Difference", quantity: "speed" },
];

const UNIT_LABELS: Record<Quantity, string> = {
  speed: "Speed", rotation: "Rotation", length: "Length", distance: "Distance",
};

function formattedRow(spec: RowSpec, value: number, units: UnitSelections): string {
  if (!Number.isFinite(value)) return "∞ (not closing)";
  if (spec.quantity) {
    const unit = units[spec.quantity];
    const displayed = fromCanonical(spec.quantity, unit, value);
    return `${displayed >= 0 ? "+" : ""}${displayed.toFixed(2)} ${unit}`;
  }
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)}${spec.unit ?? ""}`;
}

function ResultButton(props: {
  readonly spec: RowSpec;
  readonly value: number;
  readonly units: UnitSelections;
  readonly selected: boolean;
  readonly onSelect: () => void;
}) {
  const { spec, value, units, selected, onSelect } = props;
  const explanation = RESULT_EXPLANATIONS[spec.key] ?? METRIC_EXPLANATIONS[spec.key];
  return (
    <button type="button" onClick={onSelect} aria-pressed={selected} title={explanation}
      className={`flex items-center justify-between rounded-lg border px-3 py-2 text-left text-sm transition-all ${selected
        ? "border-sky-400 bg-sky-500/20 ring-1 ring-sky-400/60 shadow-[0_0_14px_rgba(56,189,248,0.25)]"
        : "border-slate-800/80 bg-slate-900/50 hover:border-slate-600 hover:bg-slate-800/50"}`}>
      <span className="flex items-center gap-2 text-slate-300">
        {spec.label}
        <span aria-hidden="true" className="text-[10px] font-semibold uppercase tracking-wide text-sky-400">Details ›</span>
      </span>
      <span className={`font-semibold tabular-nums ${spec.key === "pathDeviationDeg" ? "text-amber-300" : "text-slate-100"}`}>
        {formattedRow(spec, value, units)}
      </span>
    </button>
  );
}

function UnitsCard(props: {
  readonly units: UnitSelections;
  readonly onChange: Dispatch<SetStateAction<UnitSelections>>;
}) {
  return (
    <div className="rounded-xl border border-slate-800/80 bg-slate-900/60 p-5 shadow-lg shadow-black/20 backdrop-blur">
      <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">Units</h2>
      {(Object.keys(QUANTITY_UNITS) as Quantity[]).map((quantity) => (
        <label key={quantity} className="mb-2 flex items-center justify-between text-sm">
          <span className="text-slate-300">{UNIT_LABELS[quantity]}</span>
          <select value={props.units[quantity]} title={`Display unit for ${UNIT_LABELS[quantity].toLowerCase()} values`}
            onChange={(event) => props.onChange((units) => ({ ...units, [quantity]: event.target.value }))}
            className="rounded border border-slate-700 bg-slate-800 px-2 py-1 text-slate-100 focus:border-blue-500 focus:outline-none">
            {Object.keys(QUANTITY_UNITS[quantity]).map((unit) => <option key={unit}>{unit}</option>)}
          </select>
        </label>
      ))}
    </div>
  );
}

function ScenarioCard(props: {
  readonly scenario: ImpactScenario;
  readonly units: UnitSelections;
  readonly onUpdate: (key: keyof ImpactScenario, quantity: Quantity | undefined, raw: string) => void;
}) {
  return (
    <div className="rounded-xl border border-slate-800/80 bg-slate-900/60 p-5 shadow-lg shadow-black/20 backdrop-blur">
      <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">Scenario</h2>
      {FIELDS.map(({ key, label, quantity, fixedUnit, step }) => {
        const unit = quantity ? props.units[quantity] : fixedUnit ?? "";
        const displayed = quantity
          ? fromCanonical(quantity, props.units[quantity], props.scenario[key])
          : props.scenario[key];
        return (
          <label key={key} title={FIELD_GUIDANCE[key]} className="mb-3 block text-sm">
            <span className="mb-1 flex justify-between text-slate-300">
              <span className="flex items-center">{label}<FieldInfo label={label} guidance={FIELD_GUIDANCE[key]} /></span>
              <span className="text-slate-500">{unit}</span>
            </span>
            <DecimalInput step={step} value={Number(displayed.toFixed(4))}
              aria-label={`${label} ${unit}`.trim()} title={FIELD_GUIDANCE[key]}
              onCommit={(value) => props.onUpdate(key, quantity, String(value))}
              className="no-spinner w-full rounded border border-slate-700 bg-slate-800 px-2 py-1.5 text-slate-100 focus:border-blue-500 focus:outline-none" />
          </label>
        );
      })}
    </div>
  );
}

function ResultsCard(props: {
  readonly scenario: ImpactScenario;
  readonly units: UnitSelections;
  readonly explained: string;
  readonly onExplainedChange: (key: string) => void;
  readonly onOpenGlossary: (term: string | undefined) => void;
}) {
  const result = useMemo(() => solve(props.scenario), [props.scenario]);
  const metrics = useMemo(() => closureMetrics(props.scenario), [props.scenario]);
  const rows = [...RESULT_ROWS, ...METRIC_ROWS];
  const label = rows.find((row) => row.key === props.explained)?.label;
  const explanation = RESULT_EXPLANATIONS[props.explained] ?? METRIC_EXPLANATIONS[props.explained];
  const renderRows = (specs: readonly RowSpec[], valueFor: (key: string) => number) => specs.map((spec) => (
    <ResultButton key={spec.key} spec={spec} value={valueFor(spec.key)} units={props.units}
      selected={props.explained === spec.key} onSelect={() => props.onExplainedChange(spec.key)} />
  ));
  return (
    <div aria-label="Results" className="rounded-xl border border-slate-800/80 bg-slate-900/60 p-5 shadow-lg shadow-black/20 backdrop-blur">
      <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">Impact-Point Deviation — Click a Value for Its Explanation</h2>
      <div className="grid gap-2 sm:grid-cols-2">{renderRows(
        RESULT_ROWS,
        (key) => result[key as keyof typeof result] as number,
      )}</div>
      <h2 className="mb-3 mt-5 text-sm font-semibold uppercase tracking-wide text-slate-400">Common Closure Metrics</h2>
      <div className="grid gap-2 sm:grid-cols-2">{renderRows(
        METRIC_ROWS,
        (key) => metrics[key as keyof typeof metrics],
      )}</div>
      {label && <div aria-live="polite" className="mt-3 rounded-md border border-slate-800 bg-slate-950/60 p-3 text-xs text-slate-400">
        <h3 className="mb-1 text-sm font-bold text-sky-200">{label}</h3>{explanation}
        <button type="button" onClick={() => props.onOpenGlossary(FIELD_TO_TERM[props.explained])}
          title="Open the glossary, pre-selecting the matching term"
          className="mt-2 block text-sky-400 underline-offset-2 hover:underline">Glossary →</button>
      </div>}
      <p className="mt-3 text-xs text-slate-500">Sign convention follows standard launch-monitor definitions: club path positive = in-to-out (right of target); negative path deviation = the impact point travels left of the reported geometric-center path. Defaults are dossier-sourced (Cheetham 2014 tour HTV 1,307 ± 304 °/s about the shaft; CCV ≈ 2,100 °/s; 40 mm GC-to-face offset) — hover any input for its suggested range and source, and enter your own measured values.</p>
    </div>
  );
}

interface ImpactExplorerPanelProps {
  readonly scenario: ImpactScenario;
  readonly setScenario: Dispatch<SetStateAction<ImpactScenario>>;
  readonly units: UnitSelections;
  readonly setUnits: Dispatch<SetStateAction<UnitSelections>>;
  readonly clubSpec: ClubSpec;
  readonly setClubSpec: (spec: ClubSpec) => void;
  readonly generatedHead: GeneratedHead;
  readonly setGeneratedHead: (head: GeneratedHead) => void;
  readonly clubMeshSource: ClubMeshSource;
  readonly setClubMeshSource: Dispatch<SetStateAction<ClubMeshSource>>;
  readonly clubCamera: ClubCamera;
  readonly setClubCamera: Dispatch<SetStateAction<ClubCamera>>;
  readonly explained: string;
  readonly onExplainedChange: (key: string) => void;
  readonly onOpenGlossary: (term: string | undefined) => void;
}

export function ImpactExplorerPanel(props: ImpactExplorerPanelProps) {
  const update = (key: keyof ImpactScenario, quantity: Quantity | undefined, raw: string) => {
    const displayed = Number(raw);
    if (!Number.isFinite(displayed)) return;
    const canonical = quantity ? toCanonical(quantity, props.units[quantity], displayed) : displayed;
    const [low, high] = BOUNDS[key];
    props.setScenario((scenario) => ({ ...scenario, [key]: Math.min(high, Math.max(low, canonical)) }));
  };
  const driveScenario = (comToFaceMm: number, lieAngleDeg: number) => {
    props.setScenario((scenario) => ({ ...scenario, comToFaceMm, lieAngleDeg }));
  };
  const adoptGenerated = (head: GeneratedHead) => {
    props.setGeneratedHead(head);
    props.setClubMeshSource((prior) =>
      generatedMeshSource(head, head.label, prior.generation + 1));
  };
  return (
    <div className="grid gap-6 lg:grid-cols-[340px_1fr]">
      <section aria-label="Scenario inputs" className="space-y-4">
        <UnitsCard units={props.units} onChange={props.setUnits} />
        <ClubPanel initialSpec={props.clubSpec} onDriveScenario={driveScenario}
          onGenerate={adoptGenerated} onSpecChange={props.setClubSpec} />
        <ScenarioCard scenario={props.scenario} units={props.units} onUpdate={update} />
      </section>
      <section className="order-first space-y-6 lg:order-none">
        <ClubCanvas scenario={props.scenario} source={props.clubMeshSource}
          onSourceChange={props.setClubMeshSource} camera={props.clubCamera}
          onCameraChange={props.setClubCamera} />
        <ResultsCard scenario={props.scenario} units={props.units}
          explained={props.explained} onExplainedChange={props.onExplainedChange}
          onOpenGlossary={props.onOpenGlossary} />
      </section>
    </div>
  );
}
