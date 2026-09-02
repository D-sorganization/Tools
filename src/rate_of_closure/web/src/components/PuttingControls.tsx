/**
 * Putt setup controls — putter, stroke (#4800 P1), green (#4800 P2).
 *
 * React parity for the Qt Putting tab's control column (#4800 P6/P7):
 * every parameter the shared impact and green models accept is
 * editable here, with the same bounds the Python models validate, so a
 * refusal is shown rather than silently clamped away from the model.
 */

import { type ChangeEvent } from "react";

import { DecimalInput } from "./DecimalInput";
import {
  GREEN_FIELDS,
  STROKE_FIELDS,
  type FieldSpec,
  type PaceMode,
  type PuttSetup,
} from "./puttingSetup";
import { readFileText } from "./variationUi";
import type { PutterHeadDocument } from "../model/putterHead";
import type { CaptureModel, GreenSurface } from "../model/puttingGreen";
import { greenSurfaceFromDocument } from "../model/puttingGreenImport";

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
  readonly disabled?: boolean;
}

function NumberField({ spec, value, onCommit, disabled = false }: NumberFieldProps) {
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
          disabled={disabled}
          className={INPUT}
        />
        <span className="text-slate-400">{spec.suffix}</span>
      </span>
    </label>
  );
}

/** The green surface actually integrated, if it replaces the planar one. */
export interface ImportedGreenState {
  readonly surface: GreenSurface;
  /** Provenance label: file name plus the wire it was read through. */
  readonly source: string;
}

//: Displayed when no heightfield has been imported (matches the Qt
//: `PuttingGreenControls._PLANAR_SOURCE` label).
export const PLANAR_GREEN_SOURCE = "planar grade/aspect";

interface PuttingControlsProps {
  readonly setup: PuttSetup;
  readonly onChange: (patch: Partial<PuttSetup>) => void;
  readonly putters: readonly PutterHeadDocument[];
  readonly importedGreen: ImportedGreenState | null;
  readonly importErrorMessage: string;
  readonly onGreenImported: (state: ImportedGreenState) => void;
  readonly onGreenImportError: (message: string) => void;
  readonly onUsePlanarGreen: () => void;
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
  importedGreen,
  importErrorMessage,
  onGreenImported,
  onGreenImportError,
  onUsePlanarGreen,
}: PuttingControlsProps) {
  const field = (spec: FieldSpec, disabled = false) => (
    <NumberField
      key={spec.key}
      spec={spec}
      value={setup[spec.key] as number}
      onCommit={(value) => onChange({ [spec.key]: value } as Partial<PuttSetup>)}
      disabled={disabled}
    />
  );
  const imported = importedGreen !== null;

  const handleImportGreenDocument = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    readFileText(file)
      .then((text) => {
        const { surface, wire } = greenSurfaceFromDocument(text);
        onGreenImported({ surface, source: `${file.name} via ${wire}` });
      })
      .catch((err: unknown) => {
        const message = err instanceof Error ? err.message : String(err);
        onGreenImportError(`Green import refused (${file.name}): ${message}`);
      });
  };

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
        {STROKE_FIELDS.map((spec) => field(spec))}
      </div>

      <div className={CARD}>
        <h2 className={HEADING}>Green</h2>
        {GREEN_FIELDS.map((spec) =>
          field(spec, imported && (spec.key === "grade" || spec.key === "aspect")),
        )}
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
        <div className="mt-3 flex flex-col gap-2 border-t border-slate-800 pt-3 text-sm">
          <p
            aria-label="Green surface source"
            className="text-slate-400"
            title="The green geometry actually integrated. A heightfield loaded through swing_sim.green_surface/1 or an UpstreamDrift topography (#4800 P9) replaces the planar grade and aspect above."
          >
            {imported ? importedGreen.source : PLANAR_GREEN_SOURCE}
          </p>
          {importErrorMessage ? (
            <p role="alert" className="text-red-400">
              {importErrorMessage}
            </p>
          ) : null}
          <div className="flex items-center gap-2">
            <label className="flex-1 text-slate-300">
              <span className="sr-only">Import Green Document</span>
              <input
                aria-label="Import Green Document File"
                type="file"
                accept=".json"
                title="Load a green heightfield: a swing_sim.green_surface/1 document, or an UpstreamDrift putting_green topography (#4800 P2/P9). The reader is chosen by the document's declared format and refuses anything it does not fully understand."
                onChange={handleImportGreenDocument}
                className="text-xs"
              />
            </label>
            <button
              type="button"
              aria-label="Use Planar Green"
              title="Discard an imported heightfield and return to the planar grade/aspect green above."
              disabled={!imported}
              onClick={onUsePlanarGreen}
              className="rounded border border-slate-700 bg-slate-800 px-2 py-1 text-xs text-slate-200 disabled:cursor-not-allowed disabled:opacity-50"
            >
              Use planar green
            </button>
          </div>
        </div>
      </div>
    </>
  );
}
