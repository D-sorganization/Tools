/**
 * Wire `golf_club.putter_head/1` — TypeScript mirror of the wire half
 * of `shared/python/golf_club/putter_head.py` (epic #4800, P3), split
 * out like `puttingGreenWire.ts` (P2 precedent).
 *
 * Versioned, fail-closed JSON with the package posture: sorted keys,
 * compact separators, finite values only, unknown fields refused,
 * byte-identical round-trips within a runtime. Float formatting is
 * runtime-local (JS shortest-round-trip vs Python repr);
 * cross-runtime interchange is by JSON value. Mesh-sourced documents
 * carry `cg_m` + `inertia_at_cg_kg_m2`; library-fallback documents
 * omit both (that absence *is* the fallback semantics).
 */

import type { Vec3 } from "./club";
import {
  PUTTER_HEAD_FORMAT,
  validatePutterHead,
  type Matrix3,
  type PutterHeadDocument,
  type PutterHeadProvenance,
} from "./putterHead";

function finiteNumber(value: unknown, name: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${name} must be a finite number`);
  }
  return value;
}

function canonicalJson(value: unknown): string {
  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      throw new Error("canonical JSON requires finite numbers");
    }
    return JSON.stringify(value);
  }
  if (typeof value === "string") return JSON.stringify(value);
  if (Array.isArray(value)) {
    return `[${value.map(canonicalJson).join(",")}]`;
  }
  if (typeof value === "object" && value !== null) {
    const record = value as Record<string, unknown>;
    const parts = Object.keys(record)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${canonicalJson(record[key])}`);
    return `{${parts.join(",")}}`;
  }
  throw new Error("unsupported canonical JSON value");
}

/** Serialize with deterministic key ordering (runtime-local bytes). */
export function putterHeadToJson(document: PutterHeadDocument): string {
  validatePutterHead(document);
  const provenance: Record<string, unknown> = {
    source_kind: document.provenance.source_kind,
  };
  for (const field of [
    "mesh_sha256",
    "density_kg_m3",
    "target_mass_kg",
    "library_name",
  ] as const) {
    if (document.provenance[field] !== undefined) {
      provenance[field] = document.provenance[field];
    }
  }
  const payload: Record<string, unknown> = {
    format: PUTTER_HEAD_FORMAT,
    name: document.name,
    head_mass_kg: document.head_mass_kg,
    loft_deg: document.loft_deg,
    cor: document.cor,
    provenance,
  };
  if (document.cg_m !== undefined) {
    payload.cg_m = document.cg_m;
  }
  if (document.inertia_at_cg_kg_m2 !== undefined) {
    payload.inertia_at_cg_kg_m2 = document.inertia_at_cg_kg_m2;
  }
  return canonicalJson(payload);
}

const DOCUMENT_FIELDS = new Set([
  "format",
  "name",
  "head_mass_kg",
  "loft_deg",
  "cor",
  "cg_m",
  "inertia_at_cg_kg_m2",
  "provenance",
]);
const PROVENANCE_FIELDS = new Set([
  "source_kind",
  "mesh_sha256",
  "density_kg_m3",
  "target_mass_kg",
  "library_name",
]);

function rejectUnknownFields(
  record: Record<string, unknown>,
  allowed: Set<string>,
  name: string,
): void {
  const unknown = Object.keys(record).filter((key) => !allowed.has(key));
  if (unknown.length > 0) {
    throw new Error(
      `${name} contains unknown fields: ${unknown.sort().join(", ")}`,
    );
  }
}

function asRecord(value: unknown, name: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${name} must be an object`);
  }
  return value as Record<string, unknown>;
}

function optionalVec3(value: unknown, name: string): Vec3 | undefined {
  if (value === undefined) return undefined;
  if (!Array.isArray(value) || value.length !== 3) {
    throw new Error(`${name} must have three components`);
  }
  return value.map((item, index) =>
    finiteNumber(item, `${name}[${index}]`),
  ) as Vec3;
}

function optionalMatrix3(value: unknown, name: string): Matrix3 | undefined {
  if (value === undefined) return undefined;
  if (!Array.isArray(value) || value.length !== 3) {
    throw new Error(`${name} must be 3x3`);
  }
  const rows = value.map((row, i) => optionalVec3(row, `${name}[${i}]`));
  return rows as unknown as Matrix3;
}

/** Parse and validate; unknown fields and wrong formats are refused. */
export function putterHeadFromJson(text: string): PutterHeadDocument {
  const data = asRecord(JSON.parse(text), "putter head document");
  rejectUnknownFields(data, DOCUMENT_FIELDS, "putter head document");
  if (data.format !== PUTTER_HEAD_FORMAT) {
    throw new Error(`format must be ${PUTTER_HEAD_FORMAT}`);
  }
  const provenanceData = asRecord(data.provenance, "provenance");
  rejectUnknownFields(provenanceData, PROVENANCE_FIELDS, "provenance");
  const document: PutterHeadDocument = {
    name: typeof data.name === "string" ? data.name : "",
    head_mass_kg: finiteNumber(data.head_mass_kg, "head_mass_kg"),
    loft_deg: finiteNumber(data.loft_deg, "loft_deg"),
    cor: finiteNumber(data.cor, "cor"),
    provenance: provenanceData as unknown as PutterHeadProvenance,
    ...(data.cg_m !== undefined
      ? { cg_m: optionalVec3(data.cg_m, "cg_m") }
      : {}),
    ...(data.inertia_at_cg_kg_m2 !== undefined
      ? {
          inertia_at_cg_kg_m2: optionalMatrix3(
            data.inertia_at_cg_kg_m2,
            "inertia_at_cg_kg_m2",
          ),
        }
      : {}),
  };
  return validatePutterHead(document);
}
