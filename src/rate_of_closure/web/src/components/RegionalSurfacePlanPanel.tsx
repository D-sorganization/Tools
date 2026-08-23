import { type ChangeEvent, useRef, useState } from "react";

import {
  MAX_REGIONAL_SURFACE_EDITOR_ROWS,
  editorDraftFromGroundRegionalSurfacePlanRequest,
  illustrativeRegionalSurfacePlanDraft,
  regionalSurfacePlanRequestForDraft,
  type RegionalOverlayDraft,
  type RegionalSurfacePlanDraft,
  type SurfaceMaterialDraft,
} from "../model/regionalSurfacePlan";
import {
  downloadRegionalSurfacePlanRequest,
  readRegionalSurfacePlanFile,
} from "../model/regionalSurfacePlanFiles";
import type { GroundRegionalMaterialPlanRequest } from "../model/groundRegionalPlan";
import { RegionalExecutionEvidencePanel } from "./RegionalExecutionEvidencePanel";

type NumericMaterialKey = Exclude<keyof SurfaceMaterialDraft, "surface_id">;

const MATERIAL_FIELDS: readonly [NumericMaterialKey, string, number][] = [
  ["normal_restitution", "Normal restitution (fraction)", 0.01],
  ["static_friction", "Static friction (coefficient)", 0.01],
  ["kinetic_friction", "Kinetic friction (coefficient)", 0.01],
  ["rolling_resistance", "Rolling resistance (coefficient)", 0.01],
  ["firmness_pa", "Firmness (Pa)", 1000],
  ["hardness_fraction", "Hardness (fraction)", 0.01],
  ["grass_height_m", "Grass height (m)", 0.001],
  ["compressibility_fraction", "Compressibility (fraction)", 0.01],
  ["compression_damping_fraction", "Compression damping (fraction)", 0.01],
  ["turf_density_kg_m3", "Turf density (kg/m³)", 1],
  ["moisture_fraction", "Moisture (fraction)", 0.01],
];

function NumericInput(props: {
  readonly label: string;
  readonly value: number;
  readonly step?: number;
  readonly invalid?: boolean;
  readonly onChange: (value: number) => void;
}) {
  return (
    <label className="flex min-w-0 flex-col gap-1 text-xs text-slate-400">
      <span>{props.label}</span>
      <input type="number" value={Number.isNaN(props.value) ? "" : props.value}
        step={props.step ?? "any"} aria-label={props.label}
        aria-invalid={props.invalid || undefined}
        aria-describedby={props.invalid ? "regional-surface-plan-error" : undefined}
        onChange={(event) => props.onChange(event.target.value === ""
          ? Number.NaN : Number(event.target.value))}
        className="rounded-md border border-slate-700 bg-slate-950 px-2 py-1.5 text-slate-100 focus-visible:outline-none focus:border-sky-500 aria-[invalid=true]:border-rose-500" />
    </label>
  );
}

function MaterialEditor(props: {
  readonly prefix: string;
  readonly value: SurfaceMaterialDraft;
  readonly onChange: (value: SurfaceMaterialDraft) => void;
}) {
  const updateNumber = (field: NumericMaterialKey, value: number) =>
    props.onChange({ ...props.value, [field]: value });
  return (
    <div className="mt-3 grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
      <label className="flex flex-col gap-1 text-xs text-slate-400">
        <span>Surface ID</span>
        <input value={props.value.surface_id}
          aria-label={`${props.prefix} surface ID`}
          onChange={(event) => props.onChange({
            ...props.value, surface_id: event.target.value,
          })}
          className="rounded-md border border-slate-700 bg-slate-950 px-2 py-1.5 text-slate-100 focus-visible:outline-none focus:border-sky-500" />
      </label>
      {MATERIAL_FIELDS.map(([field, label, step]) => (
        <NumericInput key={field} label={`${props.prefix} ${label}`}
          value={props.value[field]} step={step}
          onChange={(value) => updateNumber(field, value)} />
      ))}
    </div>
  );
}

function OverlayEditor(props: {
  readonly index: number;
  readonly value: RegionalOverlayDraft;
  readonly validationAttempted: boolean;
  readonly removable: boolean;
  readonly onChange: (value: RegionalOverlayDraft) => void;
  readonly onRemove: () => void;
}) {
  const ordinal = props.index + 1;
  const invalidInterval = props.validationAttempted &&
    props.value.lower_coordinate_m >= props.value.upper_coordinate_m;
  return (
    <fieldset className="rounded-xl border border-slate-700/80 bg-slate-900/60 p-4"
      aria-label={`Regional overlay ${ordinal}`}>
      <div className="flex items-center justify-between gap-3">
        <legend className="font-semibold text-slate-200">Regional overlay {ordinal}</legend>
        <button type="button" disabled={!props.removable} onClick={props.onRemove}
          className="rounded-md border border-slate-700 px-2 py-1 text-xs text-slate-300 disabled:opacity-40">
          Remove overlay {ordinal}
        </button>
      </div>
      <div className="mt-3 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <label className="flex flex-col gap-1 text-xs text-slate-400">
          <span>Region ID</span>
          <input value={props.value.region_id} aria-label={`Overlay ${ordinal} region ID`}
            onChange={(event) => props.onChange({ ...props.value, region_id: event.target.value })}
            className="rounded-md border border-slate-700 bg-slate-950 px-2 py-1.5 text-slate-100" />
        </label>
        <NumericInput label={`Overlay ${ordinal} precedence`}
          value={props.value.precedence} step={1}
          onChange={(value) => props.onChange({ ...props.value, precedence: value })} />
        <NumericInput label={`Overlay ${ordinal} lower coordinate (m)`}
          value={props.value.lower_coordinate_m} invalid={invalidInterval}
          onChange={(value) => props.onChange({ ...props.value, lower_coordinate_m: value })} />
        <NumericInput label={`Overlay ${ordinal} upper coordinate (m)`}
          value={props.value.upper_coordinate_m} invalid={invalidInterval}
          onChange={(value) => props.onChange({ ...props.value, upper_coordinate_m: value })} />
      </div>
      <MaterialEditor prefix={`Overlay ${ordinal}`} value={props.value.surface}
        onChange={(surface) => props.onChange({ ...props.value, surface })} />
    </fieldset>
  );
}

function nextOverlay(draft: RegionalSurfacePlanDraft): RegionalOverlayDraft {
  const ordinal = draft.regions.length + 1;
  const template = illustrativeRegionalSurfacePlanDraft().regions[0];
  const lower = Math.min(280, 120 + (ordinal - 1) * 20);
  return {
    ...template,
    region_id: `illustrative-region-${ordinal}`,
    precedence: ordinal * 10,
    lower_coordinate_m: lower,
    upper_coordinate_m: Math.min(295, lower + 15),
    surface: { ...template.surface, surface_id: `illustrative-surface-${ordinal}` },
  };
}

export interface RegionalSurfacePlanPanelProps {
  readonly draft: RegionalSurfacePlanDraft;
  readonly importedRequest: GroundRegionalMaterialPlanRequest | null;
  readonly onDraftChange: (draft: RegionalSurfacePlanDraft) => void;
  readonly onImport: (request: GroundRegionalMaterialPlanRequest) => void;
}

export function RegionalSurfacePlanPanel({
  draft,
  importedRequest,
  onDraftChange,
  onImport,
}: RegionalSurfacePlanPanelProps) {
  const [error, setError] = useState<string | null>(null);
  const [readback, setReadback] = useState<string | null>(null);
  const [fileStatus, setFileStatus] = useState<string | null>(null);
  const [validationAttempted, setValidationAttempted] = useState(false);
  const [executionRevision, setExecutionRevision] = useState(0);
  const fileInput = useRef<HTMLInputElement>(null);
  const updateDraft = (
    transform: (current: RegionalSurfacePlanDraft) => RegionalSurfacePlanDraft,
  ) => {
    onDraftChange(transform(draft));
    setError(null);
    setReadback(null);
    setFileStatus(null);
    setValidationAttempted(false);
    setExecutionRevision((value) => value + 1);
  };
  const updateRegion = (index: number, value: RegionalOverlayDraft) =>
    updateDraft((current) => ({
      ...current,
      regions: current.regions.map((region, row) => row === index ? value : region),
    }));
  const validate = () => {
    setValidationAttempted(true);
    try {
      const request = regionalSurfacePlanRequestForDraft(draft, importedRequest);
      setError(null);
      setReadback(
        `${request.schema_version} · ${request.unit_system} · ` +
        `${request.base_surface.surface_id} · ${request.regions.length} overlay(s) · ` +
        `source ${request.provenance.source_revision} · ` +
        `input ${request.provenance.input_sha256}`,
      );
    } catch (caught) {
      setReadback(null);
      setError(caught instanceof Error ? caught.message : "Surface plan is invalid");
    }
  };
  const importFile = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.currentTarget.files?.[0];
    event.currentTarget.value = "";
    if (file === undefined) return;
    try {
      const request = await readRegionalSurfacePlanFile(file);
      editorDraftFromGroundRegionalSurfacePlanRequest(request);
      onImport(request);
      setValidationAttempted(false);
      setError(null);
      setReadback(`${request.schema_version} · exact imported provenance ${request.provenance.input_sha256}`);
      setFileStatus(`Imported ${file.name}. No physics executed.`);
      setExecutionRevision((value) => value + 1);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Regional plan import failed");
      setFileStatus("Import failed; the editor and prior validated readback were preserved.");
    }
  };
  const download = () => {
    try {
      const request = regionalSurfacePlanRequestForDraft(draft, importedRequest);
      downloadRegionalSurfacePlanRequest(request);
      setError(null);
      setFileStatus("Download prepared. Your browser controls the destination and overwrite behavior.");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Regional plan download failed");
      setFileStatus("Download failed; no browser filesystem state was changed.");
    }
  };
  return (
    <section aria-labelledby="regional-surface-plan-title" className="space-y-4">
      <header>
        <h2 id="regional-surface-plan-title" className="text-xl font-semibold text-slate-100">
          Regional surface plan
        </h2>
        <p className="mt-1 text-sm text-slate-400">
          Define one static coplanar SI base surface and up to {MAX_REGIONAL_SURFACE_EDITOR_ROWS} material overlays.
        </p>
      </header>
      <div role="note" aria-label="Regional surface qualification"
        className="rounded-lg border border-amber-500/40 bg-amber-500/10 p-3 text-sm text-amber-100">
        Illustrative, unvalidated values are loaded for discovery. They are not measured course data.
        This slice validates a material plan only; it does not run physics or playback.
      </div>
      <p className="text-xs text-slate-500">
        Import/download persists this canonical request only; workspace persistence remains separate.
        Browser downloads cannot promise a native path, atomic replacement, or recent-file access.
      </p>
      <fieldset className="rounded-xl border border-slate-700/80 bg-slate-900/60 p-4">
        <legend className="font-semibold text-slate-200">Plan identity and provenance</legend>
        <div className="mt-3 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          <label className="flex flex-col gap-1 text-xs text-slate-400">
            <span>Request ID</span>
            <input value={draft.request_id} aria-label="Regional plan request ID"
              onChange={(event) => updateDraft((current) => ({
                ...current, request_id: event.target.value,
              }))}
              className="rounded-md border border-slate-700 bg-slate-950 px-2 py-1.5 text-slate-100" />
          </label>
          <label className="flex flex-col gap-1 text-xs text-slate-400">
            <span>Source revision</span>
            <input value={draft.source_revision} aria-label="Regional plan source revision"
              onChange={(event) => updateDraft((current) => ({
                ...current, source_revision: event.target.value,
              }))}
              className="rounded-md border border-slate-700 bg-slate-950 px-2 py-1.5 text-slate-100" />
          </label>
          <label className="flex flex-col gap-1 text-xs text-slate-400">
            <span>Calibration</span>
            <select value={draft.calibration_kind} disabled aria-label="Regional plan calibration">
              <option value="unvalidated">Unvalidated</option>
            </select>
          </label>
          <p className="self-end text-xs text-slate-500">
            Frame: target x-downrange, y-up, z-right. Geometry: flat, static, coplanar.
          </p>
        </div>
      </fieldset>
      <fieldset className="rounded-xl border border-slate-700/80 bg-slate-900/60 p-4">
        <legend className="font-semibold text-slate-200">Base surface and domain</legend>
        <div className="mt-3 grid gap-3 sm:grid-cols-2">
          <NumericInput label="Base domain lower coordinate (m)"
            value={draft.lower_coordinate_m}
            onChange={(value) => updateDraft((current) => ({
              ...current, lower_coordinate_m: value,
            }))} />
          <NumericInput label="Base domain upper coordinate (m)"
            value={draft.upper_coordinate_m}
            onChange={(value) => updateDraft((current) => ({
              ...current, upper_coordinate_m: value,
            }))} />
        </div>
        <MaterialEditor prefix="Base" value={draft.base_surface}
          onChange={(base_surface) => updateDraft((current) => ({
            ...current, base_surface,
          }))} />
      </fieldset>
      {draft.regions.map((region, index) => (
        <OverlayEditor key={`regional-overlay-${index}`} index={index} value={region}
          validationAttempted={validationAttempted} removable={draft.regions.length > 1}
          onChange={(value) => updateRegion(index, value)}
          onRemove={() => updateDraft((current) => ({
            ...current,
            regions: current.regions.filter((_item, row) => row !== index),
          }))} />
      ))}
      <div className="flex flex-wrap gap-3">
        <input ref={fileInput} type="file" accept=".json,application/json"
          aria-label="Import regional surface plan JSON file" className="sr-only"
          onChange={(event) => { void importFile(event); }} />
        <button type="button" aria-label="Add regional overlay"
          disabled={draft.regions.length >= MAX_REGIONAL_SURFACE_EDITOR_ROWS}
          onClick={() => updateDraft((current) => ({
            ...current, regions: [...current.regions, nextOverlay(current)],
          }))}
          className="rounded-md border border-sky-500/60 px-3 py-2 text-sm text-sky-200 disabled:opacity-40">
          Add overlay
        </button>
        <button type="button" aria-label="Import regional surface plan JSON"
          onClick={() => fileInput.current?.click()}
          className="rounded-md border border-sky-500/60 px-3 py-2 text-sm text-sky-200">
          Import JSON
        </button>
        <button type="button" aria-label="Download regional surface plan JSON"
          onClick={download}
          className="rounded-md border border-sky-500/60 px-3 py-2 text-sm text-sky-200">
          Download JSON
        </button>
        <button type="button" aria-label="Validate surface plan" onClick={validate}
          className="rounded-md bg-sky-500 px-3 py-2 text-sm font-semibold text-slate-950">
          Validate and preview
        </button>
      </div>
      {error !== null && <p id="regional-surface-plan-error" role="alert"
        className="rounded-md border border-rose-500/60 bg-rose-500/10 p-3 text-sm text-rose-200">
        {error}
      </p>}
      {readback !== null && <p role="status" aria-label="Regional surface plan readback"
        className="rounded-md border border-emerald-500/50 bg-emerald-500/10 p-3 text-sm text-emerald-100">
        Validated readback: {readback}
      </p>}
      {fileStatus !== null && <p role="status" aria-label="Regional surface plan file status"
        className="text-xs text-slate-400">{fileStatus}</p>}
      <RegionalExecutionEvidencePanel key={executionRevision}
        currentPlan={() => regionalSurfacePlanRequestForDraft(draft, importedRequest)} />
    </section>
  );
}
