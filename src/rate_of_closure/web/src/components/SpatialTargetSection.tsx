import { useEffect, useRef, useState } from "react";

import {
  boxTolerance,
  createSpatialTarget,
  sphereTolerance,
  surfaceCircleTolerance,
  surfaceCorridorTolerance,
  targetPointFromFrame,
  targetPointInFrame,
  type AcceptanceGeometryTs,
  type SpatialTargetKind,
  type SpatialTargetTs,
  type TargetFrame,
} from "../model/spatialTarget";
import {
  spatialTargetFromJson,
  spatialTargetToJson,
} from "../model/spatialTargetSerialization";
import {
  draftFromSpatialTarget,
  spatialTargetAssessment,
  spatialTargetSummary,
  type SpatialTargetDraft,
} from "./spatialTargetPresentation";
import type { FlightPoint } from "../model/flight";

interface Props {
  target: SpatialTargetTs;
  onChange: (target: SpatialTargetTs) => void;
  flightPoints?: readonly FlightPoint[];
}

type DraftField = keyof SpatialTargetDraft;

class TargetDraftError extends Error {
  constructor(readonly field: DraftField, message: string) {
    super(message);
  }
}

const INPUT_CLASS =
  "w-full rounded border border-slate-700 bg-slate-800 px-2 py-1.5 text-slate-100 " +
  "focus:border-sky-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 aria-[invalid=true]:border-rose-400";
const ERROR_ID = "spatial-target-validation-error";

function finite(raw: string, field: DraftField, label: string): number {
  if (raw.trim() === "") throw new TargetDraftError(field, `${label} must be a finite number.`);
  const value = Number(raw);
  if (!Number.isFinite(value)) {
    throw new TargetDraftError(field, `${label} must be a finite number.`);
  }
  return value;
}

function positive(raw: string, field: DraftField, label: string): number {
  const value = finite(raw, field, label);
  if (value <= 0) throw new TargetDraftError(field, `${label} must be greater than zero.`);
  return value;
}

function toleranceFromDraft(draft: SpatialTargetDraft): AcceptanceGeometryTs {
  switch (draft.toleranceKind) {
    case "sphere":
      return sphereTolerance(positive(draft.radius, "radius", "Radius"));
    case "surface_circle":
      return surfaceCircleTolerance(positive(draft.radius, "radius", "Radius"));
    case "box":
      return boxTolerance([
        positive(draft.halfDownrange, "halfDownrange", "Downrange half extent"),
        positive(draft.halfElevation, "halfElevation", "Elevation half extent"),
        positive(draft.halfRight, "halfRight", "Right half extent"),
      ]);
    case "surface_corridor":
      return surfaceCorridorTolerance(
        positive(draft.halfDownrange, "halfDownrange", "Downrange half length"),
        positive(draft.halfRight, "halfRight", "Right half width"),
      );
  }
}

function targetFromDraft(draft: SpatialTargetDraft): SpatialTargetTs {
  if (draft.label.trim() !== draft.label || draft.label.length === 0) {
    throw new TargetDraftError("label", "Label must be non-empty and trimmed.");
  }
  const point = targetPointFromFrame([
    finite(draft.downrange, "downrange", "Downrange"),
    finite(draft.elevation, "elevation", draft.sourceFrame === "app" ? "Elevation" : "Left"),
    finite(draft.right, "right", draft.sourceFrame === "app" ? "Right offset" : "Elevation"),
  ], draft.sourceFrame);
  return createSpatialTarget({
    label: draft.label,
    kind: draft.kind,
    point,
    tolerance: toleranceFromDraft(draft),
    elevationSource: draft.kind === "landing_area" ? "course_surface" : "absolute",
    groundSource: draft.kind === "landing_area" ? "course.surface/user" : null,
  });
}

function allowedTolerances(kind: SpatialTargetKind) {
  return kind === "landing_area"
    ? [["surface_circle", "Circle"], ["surface_corridor", "Corridor"]] as const
    : [["sphere", "Sphere"], ["box", "Box"]] as const;
}

export function SpatialTargetSection({ target, onChange, flightPoints = [] }: Props) {
  const [draft, setDraft] = useState(() => draftFromSpatialTarget(target));
  const [jsonText, setJsonText] = useState(() => spatialTargetToJson(target));
  const [error, setError] = useState<string | null>(null);
  const [invalidField, setInvalidField] = useState<DraftField | "json" | null>(null);
  const fieldRefs = useRef(new Map<DraftField | "json", HTMLElement>());

  useEffect(() => {
    setDraft(draftFromSpatialTarget(target));
    setJsonText(spatialTargetToJson(target));
  }, [target]);

  const patch = (updates: Partial<SpatialTargetDraft>) => {
    setDraft((current) => ({ ...current, ...updates }));
    setError(null);
    setInvalidField(null);
  };
  const selectKind = (kind: SpatialTargetKind) => patch({
    kind,
    toleranceKind: kind === "landing_area" ? "surface_circle" : "sphere",
  });
  const reportError = (cause: unknown, fallback: DraftField | "json" | null) => {
    const field = cause instanceof TargetDraftError ? cause.field : fallback;
    setError(cause instanceof Error ? cause.message : String(cause));
    setInvalidField(field);
    if (field !== null) fieldRefs.current.get(field)?.focus();
  };
  const selectFrame = (sourceFrame: TargetFrame) => {
    try {
      const coordinates = [
        finite(draft.downrange, "downrange", "Downrange"),
        finite(draft.elevation, "elevation", draft.sourceFrame === "app" ? "Elevation" : "Left"),
        finite(draft.right, "right", draft.sourceFrame === "app" ? "Right offset" : "Elevation"),
      ] as const;
      const point = targetPointFromFrame(coordinates, draft.sourceFrame);
      const converted = targetPointInFrame(point, sourceFrame);
      patch({
        sourceFrame,
        downrange: String(Number(converted[0].toFixed(6))),
        elevation: String(Number(converted[1].toFixed(6))),
        right: String(Number(converted[2].toFixed(6))),
      });
    } catch (cause) {
      reportError(cause, null);
    }
  };
  const apply = () => {
    try {
      const next = targetFromDraft(draft);
      onChange(next);
      setJsonText(spatialTargetToJson(next));
      setError(null);
      setInvalidField(null);
    } catch (cause) {
      reportError(cause, null);
    }
  };
  const loadJson = () => {
    try {
      const next = spatialTargetFromJson(jsonText);
      onChange(next);
      setDraft(draftFromSpatialTarget(next));
      setError(null);
      setInvalidField(null);
    } catch (cause) {
      reportError(cause, "json");
    }
  };
  const input = (field: DraftField, label: string) => (
    <label className="block text-xs text-slate-300">
      {label}
      <input type="text" inputMode="decimal" aria-label={`Target ${label.toLowerCase()} m`}
        aria-invalid={invalidField === field} value={String(draft[field])}
        aria-errormessage={invalidField === field ? ERROR_ID : undefined}
        aria-describedby={invalidField === field ? ERROR_ID : undefined}
        ref={(element) => { if (element) fieldRefs.current.set(field, element); }}
        onChange={(event) => patch({ [field]: event.target.value })}
        className={INPUT_CLASS} />
    </label>
  );
  const sourceLabels = draft.sourceFrame === "app"
    ? ["downrange", "elevation", "right offset"] as const
    : ["downrange", "left offset", "elevation"] as const;

  return (
    <section aria-label="Spatial target" className="rounded-xl border border-sky-500/25 bg-slate-900/60 p-4">
      <h2 className="mb-1 text-sm font-semibold uppercase tracking-wide text-sky-300">Spatial Target</h2>
      <p className="mb-3 text-xs text-slate-400">
        Set the desired landing area or an aerial waypoint. Values are validated only when applied;
        invalid text never changes the active target.
      </p>
      <form aria-label="Spatial target editor" onSubmit={(event) => { event.preventDefault(); apply(); }}>
      <div className="grid gap-2 sm:grid-cols-2">
        <label className="block text-xs text-slate-300">Label
          <input aria-label="Target label" aria-invalid={invalidField === "label"}
            aria-errormessage={invalidField === "label" ? ERROR_ID : undefined}
            aria-describedby={invalidField === "label" ? ERROR_ID : undefined}
            ref={(element) => { if (element) fieldRefs.current.set("label", element); }}
            value={draft.label} onChange={(event) => patch({ label: event.target.value })}
            className={INPUT_CLASS} />
        </label>
        <label className="block text-xs text-slate-300">Target type
          <select aria-label="Target type" value={draft.kind}
            onChange={(event) => selectKind(event.target.value as SpatialTargetKind)}
            className={INPUT_CLASS}>
            <option value="landing_area">Landing area</option>
            <option value="aerial_waypoint">Aerial waypoint</option>
          </select>
        </label>
        <label className="block text-xs text-slate-300">Entry frame
          <select aria-label="Target coordinate frame" value={draft.sourceFrame}
            onChange={(event) => selectFrame(event.target.value as TargetFrame)}
            className={INPUT_CLASS}>
            <option value="app">App: x downrange, y up, z right</option>
            <option value="flight">Flight: x downrange, y left, z up</option>
          </select>
        </label>
        <label className="block text-xs text-slate-300">Acceptance shape
          <select aria-label="Target acceptance shape" value={draft.toleranceKind}
            onChange={(event) => patch({ toleranceKind: event.target.value as AcceptanceGeometryTs["kind"] })}
            className={INPUT_CLASS}>
            {allowedTolerances(draft.kind).map(([value, label]) =>
              <option key={value} value={value}>{label}</option>)}
          </select>
        </label>
        {input("downrange", sourceLabels[0])}
        {input("elevation", sourceLabels[1])}
        {input("right", sourceLabels[2])}
        {(draft.toleranceKind === "sphere" || draft.toleranceKind === "surface_circle") &&
          input("radius", "radius")}
        {(draft.toleranceKind === "box" || draft.toleranceKind === "surface_corridor") &&
          input("halfDownrange", "downrange half extent")}
        {draft.toleranceKind === "box" && input("halfElevation", "elevation half extent")}
        {(draft.toleranceKind === "box" || draft.toleranceKind === "surface_corridor") &&
          input("halfRight", "right half extent")}
      </div>
      <button type="submit"
        title="Validate the draft and apply it as the active canonical spatial target"
        className="mt-3 min-h-11 rounded-md bg-sky-700 px-4 py-2 text-sm font-semibold hover:bg-sky-600">
        Apply spatial target
      </button>
      {error && <p id={ERROR_ID} role="alert" className="mt-2 text-sm text-rose-300">{error}</p>}
      </form>
      <p role="status" aria-label="Current spatial target" className="mt-3 rounded-md bg-slate-950/70 p-3 text-xs text-slate-300">
        <strong className="text-sky-200">Current target: </strong>{spatialTargetSummary(target)}
      </p>
      <p role="status" aria-label="Spatial target assessment" className="mt-2 text-xs text-slate-300">
        {spatialTargetAssessment(target, flightPoints)}
      </p>
      <details className="mt-3 text-xs text-slate-400">
        <summary className="cursor-pointer text-sky-300">Canonical JSON import/export</summary>
        <textarea aria-label="Spatial target JSON" aria-invalid={invalidField === "json"}
          aria-errormessage={invalidField === "json" ? ERROR_ID : undefined}
          aria-describedby={invalidField === "json" ? ERROR_ID : undefined}
          ref={(element) => { if (element) fieldRefs.current.set("json", element); }}
          value={jsonText} onChange={(event) => { setJsonText(event.target.value); setError(null); setInvalidField(null); }}
          rows={5} className={`${INPUT_CLASS} mt-2 font-mono`} />
        <div className="mt-2 flex flex-wrap gap-2">
          <button type="button" onClick={loadJson}
            title="Validate and load the versioned spatial-target JSON"
            className="min-h-11 rounded border border-slate-600 px-3 py-2">
            Load target JSON
          </button>
          <button type="button" onClick={() => setJsonText(spatialTargetToJson(target))}
            title="Replace the JSON editor text with the current canonical target"
            className="min-h-11 rounded border border-slate-600 px-3 py-2">
            Export current target
          </button>
        </div>
      </details>
    </section>
  );
}
