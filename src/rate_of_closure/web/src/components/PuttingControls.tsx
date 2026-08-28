/**
 * Putt setup controls — putter, stroke (#4800 P1), green (#4800 P2).
 *
 * React parity for the Qt Putting tab's control column (#4800 P6/P7):
 * every parameter the shared impact and green models accept is
 * editable here, with the same bounds the Python models validate, so a
 * refusal is shown rather than silently clamped away from the model.
 */

import { DecimalInput } from "./DecimalInput";
import {
  GREEN_FIELDS,
  STROKE_FIELDS,
  type FieldSpec,
  type PaceMode,
  type PuttSetup,
} from "./puttingSetup";
import type { PutterHeadDocument } from "../model/putterHead";
import type { CaptureModel } from "../model/puttingGreen";

const CARD =
  "rounded-xl border border-slate-800/80 bg-slate-900/60 p-5 shadow-lg " +
  "shadow-black/20 backdrop-blur";
const HEADING =
  "mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400";
const SELECT =
  "rounded border border-slate-700 bg-slate-800 px-2 py-1 text-slate-100 " +
  "focus:border-blue-500 focus-visible:outline-none focus-visible:ring-2 " +
  "focus-visible:ring-blue-500";
const INPUT =
  "w-24 rounded border border-slate-700 bg-slate-800 px-2 py-1 text-right " +
  "text-slate-100 focus:border-blue-500 focus-visible:outline-none " +
  "focus-visible:ring-2 focus-visible:ring-blue-500";

interface NumberFieldProps {
  readonly spec: FieldSpec;
  readonly value: number;
  readonly onCommit: (value: number) => void;
}

function NumberField({ spec, value, onCommit }: NumberFieldProps) {
  return (
    <label className="mb-2 flex items-center justify-between gap-2 text-sm">
      <span className="text-slate-300">{spec.label}</span>
      <span className="flex items-center gap-1">
        <DecimalInput
          value={value}
          step={spec.step}
          min={spec.bounds[0]}
          max={spec.bounds[1]}
          aria-label={`${spec.label} ${spec.suffix}`.trim()}
          title={spec.title}
          onCommit={onCommit}
          className={INPUT}
        />
        <span className="text-slate-400">{spec.suffix}</span>
      </span>
    </label>
  );
}

interface PuttingControlsProps {
  readonly setup: PuttSetup;
  readonly onChange: (patch: Partial<PuttSetup>) => void;
  readonly putters: readonly PutterHeadDocument[];
}

const PACE_SPEED: FieldSpec = {
  key: "speed",
  label: "Clubhead speed",
  suffix: "m/s",
  step: 0.05,
  bounds: [0.2, 6],
  title:
    "Clubhead speed at impact; 0.5-3 m/s covers putts inside 15 m (swing_sim.putting.impact)",
};

const PACE_BACKSTROKE: FieldSpec = {
  key: "backstrokeCm",
  label: "Backstroke",
  suffix: "cm",
  step: 1,
  bounds: [5, 100],
  title:
    "Backstroke arc length, converted with the simple-pendulum proxy v = A·sqrt(g/L); 10-60 cm typical",
};

const DISTANCE: FieldSpec = {
  key: "distance",
  label: "Distance to hole",
  suffix: "m",
  step: 0.1,
  bounds: [0.1, 40],
  title:
    "Ball-to-hole distance along the target line; 1-15 m typical (swing_sim.putting.green)",
};

export function PuttingControls({
  setup,
  onChange,
  putters,
}: PuttingControlsProps) {
  const field = (spec: FieldSpec) => (
    <NumberField
      key={spec.key}
      spec={spec}
      value={setup[spec.key] as number}
      onCommit={(value) => onChange({ [spec.key]: value } as Partial<PuttSetup>)}
    />
  );
  return (
    <>
      <div className={CARD}>
        <h2 className={HEADING}>Putt Setup</h2>
        <label className="mb-2 flex items-center justify-between gap-2 text-sm">
          <span className="text-slate-300">Putter</span>
          <select
            value={setup.putterName}
            title="Putter head used for the impact model (library putters when available, otherwise the swing_sim minimal specs); head mass, loft and any measured inertia tensor drive ball speed, launch spin and face twist"
            onChange={(event) => onChange({ putterName: event.target.value })}
            className={SELECT}
          >
            {putters.map((putter) => (
              <option key={putter.name} value={putter.name}>
                {putter.name}
              </option>
            ))}
          </select>
        </label>
        <label className="mb-2 flex items-center justify-between gap-2 text-sm">
          <span className="text-slate-300">Pace input</span>
          <select
            value={setup.paceMode}
            title="Set the stroke pace directly as clubhead speed, or as a pendulum backstroke length (v = A·sqrt(g/L))"
            onChange={(event) =>
              onChange({ paceMode: event.target.value as PaceMode })
            }
            className={SELECT}
          >
            <option value="speed">Clubhead speed</option>
            <option value="backstroke">Backstroke length</option>
          </select>
        </label>
        {field(setup.paceMode === "speed" ? PACE_SPEED : PACE_BACKSTROKE)}
        {field(DISTANCE)}
      </div>

      <div className={CARD}>
        <h2 className={HEADING}>Stroke</h2>
        {STROKE_FIELDS.map(field)}
      </div>

      <div className={CARD}>
        <h2 className={HEADING}>Green</h2>
        {GREEN_FIELDS.map(field)}
        <label className="mb-2 flex items-center justify-between gap-2 text-sm">
          <span className="text-slate-300">Hole capture</span>
          <select
            value={setup.captureModel}
            title="Effective radius: the published model shrinking the mouth as R·sqrt(1-(v/vc)²) (Holmes 1991, Penner 2002). Speed threshold: the historic bound-only test kept for regression comparison"
            onChange={(event) =>
              onChange({ captureModel: event.target.value as CaptureModel })
            }
            className={SELECT}
          >
            <option value="effective_radius">Effective radius</option>
            <option value="speed_threshold">Speed threshold</option>
          </select>
        </label>
      </div>
    </>
  );
}
